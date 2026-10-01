use bytes::{Buf, Bytes, BytesMut};
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;
const READ_BUFFER_CAPACITY: usize = 16 * 1024;

#[derive(Debug, Clone)]
pub struct StreamingClient {
    client: Client,
    chat_url: Url,
}

#[derive(Clone)]
pub struct StreamingClientBuilder {
    base_url: String,
    connect_timeout: Duration,
    request_timeout: Option<Duration>,
    bearer_token: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChunk {
    pub id: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(alias = "reasoning_content", alias = "reasoning")]
    pub reasoning: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Chunk(ChatChunk),
    Done,
}

#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub started: Option<Instant>,
    pub first_output: Option<Instant>,
    pub finished: Option<Instant>,
    pub chunks: u64,
    pub content_bytes: u64,
    pub reasoning_bytes: u64,
    pub output_events: u64,
}

impl StreamStats {
    pub fn ttft(&self) -> Option<Duration> {
        Some(self.first_output?.duration_since(self.started?))
    }

    pub fn elapsed(&self) -> Option<Duration> {
        Some(self.finished?.duration_since(self.started?))
    }

    pub fn output_bytes_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed()?.as_secs_f64();
        (seconds > 0.0).then_some(self.content_bytes as f64 / seconds)
    }

    pub fn total_output_bytes(&self) -> u64 {
        self.content_bytes + self.reasoning_bytes
    }

    pub fn total_output_bytes_per_second(&self) -> Option<f64> {
        let seconds = self.elapsed()?.as_secs_f64();
        (seconds > 0.0).then_some(self.total_output_bytes() as f64 / seconds)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReconnectPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(4),
        }
    }
}

#[derive(Debug, Error)]
pub enum StreamingError {
    #[error("stream cancelled")]
    Cancelled,
    #[error("invalid base URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned HTTP {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },
    #[error("invalid SSE payload: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SSE event exceeded {MAX_EVENT_BYTES} bytes")]
    EventTooLarge,
    #[error("SSE stream ended before [DONE]")]
    UnexpectedEof,
    #[error("invalid HTTP header value")]
    InvalidHeader,
}

impl Default for StreamingClientBuilder {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: None,
            bearer_token: None,
        }
    }
}

impl StreamingClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub fn build(self) -> Result<StreamingClient, StreamingError> {
        let mut base = Url::parse(&self.base_url)?;
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        let chat_url = base.join("chat/completions")?;

        let mut builder = Client::builder()
            .connect_timeout(self.connect_timeout)
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .http2_keep_alive_while_idle(true)
            .pool_idle_timeout(Duration::from_secs(90));

        if let Some(timeout) = self.request_timeout {
            builder = builder.timeout(timeout);
        }

        if let Some(token) = self.bearer_token {
            let mut headers = reqwest::header::HeaderMap::new();
            let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| StreamingError::InvalidHeader)?;
            headers.insert(reqwest::header::AUTHORIZATION, value);
            builder = builder.default_headers(headers);
        }

        Ok(StreamingClient {
            client: builder.build()?,
            chat_url,
        })
    }
}

impl StreamingClient {
    pub fn builder() -> StreamingClientBuilder {
        StreamingClientBuilder::new()
    }

    pub async fn stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<ChatStream<impl Stream<Item = Result<Bytes, reqwest::Error>> + Send>, StreamingError>
    {
        self.stream_with_options(&mut request, None, None).await
    }

    pub async fn stream_with_cancellation(
        &self,
        mut request: ChatRequest,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<ChatStream<impl Stream<Item = Result<Bytes, reqwest::Error>> + Send>, StreamingError>
    {
        self.stream_with_options(&mut request, Some(cancellation), None)
            .await
    }

    pub async fn stream_with_resume(
        &self,
        mut request: ChatRequest,
        last_event_id: impl AsRef<str>,
    ) -> Result<ChatStream<impl Stream<Item = Result<Bytes, reqwest::Error>> + Send>, StreamingError>
    {
        self.stream_with_options(&mut request, None, Some(last_event_id.as_ref().to_owned()))
            .await
    }

    pub async fn stream_with_reconnect(
        &self,
        request: ChatRequest,
        policy: ReconnectPolicy,
    ) -> Result<impl Stream<Item = Result<StreamEvent, StreamingError>>, StreamingError> {
        self.stream_with_reconnect_internal(request, policy, None)
            .await
    }

    pub async fn stream_with_reconnect_and_cancellation(
        &self,
        request: ChatRequest,
        policy: ReconnectPolicy,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> Result<impl Stream<Item = Result<StreamEvent, StreamingError>>, StreamingError> {
        self.stream_with_reconnect_internal(request, policy, Some(cancellation))
            .await
    }

    async fn stream_with_reconnect_internal(
        &self,
        request: ChatRequest,
        policy: ReconnectPolicy,
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<impl Stream<Item = Result<StreamEvent, StreamingError>>, StreamingError> {
        let client = self.clone();
        let stream = async_stream::try_stream! {
        let mut last_event_id: Option<String> = None;
        let mut retries = 0u32;
        let mut delay = std::cmp::min(policy.initial_backoff, policy.max_backoff);

        loop {
            let result = match cancellation.as_ref() {
                Some(token) => tokio::select! {
                    _ = token.cancelled() => Err(StreamingError::Cancelled),
                    result = client.stream_boxed_resume(
                        request.clone(),
                        last_event_id.as_deref().filter(|id| !id.is_empty()),
                        cancellation.clone(),
                    ) => result,
                },
                None => {
                    client
                        .stream_boxed_resume(
                            request.clone(),
                            last_event_id.as_deref().filter(|id| !id.is_empty()),
                            None,
                        )
                        .await
                }
            };

            let mut current = Box::pin(match result {
                Ok(stream) => stream,
                Err(error) => {
                    let retryable = is_retryable_stream_error(&error);
                    let resumable = last_event_id
                        .as_deref()
                        .is_some_and(|id| !id.is_empty());

                    if !retryable || !resumable || retries >= policy.max_retries {
                        Err(error)?;
                    }

                    retries += 1;
                    let cancelled = match cancellation.as_ref() {
                        Some(token) => tokio::select! {
                            _ = token.cancelled() => true,
                            _ = tokio::time::sleep(delay) => false,
                        },
                        None => {
                            tokio::time::sleep(delay).await;
                            false
                        }
                    };

                    if cancelled {
                        Err(StreamingError::Cancelled)?;
                    }

                    delay = std::cmp::min(
                        delay.saturating_mul(2),
                        policy.max_backoff,
                    );
                    continue;
                }
            });

            loop {
                match current.next().await {
                    Some(Ok(event)) => {
                        if let Some(id) = current.last_event_id() {
                            last_event_id = Some(id.to_owned());
                        }

                        let done = matches!(&event, StreamEvent::Done);
                        yield event;

                        if done {
                            return;
                        }

                        retries = 0;
                        delay = std::cmp::min(
                            policy.initial_backoff,
                            policy.max_backoff,
                        );
                    }
                    Some(Err(error)) => {
                        let retryable = is_retryable_stream_error(&error);
                        let resumable = last_event_id
                            .as_deref()
                            .is_some_and(|id| !id.is_empty());

                        if !retryable || !resumable || retries >= policy.max_retries {
                            Err(error)?;
                        }

                        retries += 1;
                        let cancelled = match cancellation.as_ref() {
                            Some(token) => tokio::select! {
                                _ = token.cancelled() => true,
                                _ = tokio::time::sleep(delay) => false,
                            },
                            None => {
                                tokio::time::sleep(delay).await;
                                false
                            }
                        };

                        if cancelled {
                            Err(StreamingError::Cancelled)?;
                        }

                        delay = std::cmp::min(
                            delay.saturating_mul(2),
                            policy.max_backoff,
                        );
                        break;
                    }
                    None => return,
                }
            }
        }
        };

        Ok(stream)
    }

    async fn stream_boxed_resume(
        &self,
        request: ChatRequest,
        last_event_id: Option<&str>,
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> Result<ChatStream<DynBody>, StreamingError> {
        let mut request = request;
        request.stream = true;
        let started = Instant::now();

        let mut builder = self
            .client
            .post(self.chat_url.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .header(reqwest::header::CACHE_CONTROL, "no-cache")
            .header(reqwest::header::ACCEPT_ENCODING, "identity");

        if let Some(id) = last_event_id {
            builder = builder.header("Last-Event-ID", id);
        }

        let response = match cancellation.as_ref() {
            Some(token) => tokio::select! {
                _ = token.cancelled() => return Err(StreamingError::Cancelled),
                result = builder.json(&request).send() => result?,
            },
            None => builder.json(&request).send().await?,
        };
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StreamingError::HttpStatus { status, body });
        }

        let cancellation = cancellation.unwrap_or_else(tokio_util::sync::CancellationToken::new);
        let cancelled = cancellation.clone().cancelled_owned();

        Ok(ChatStream {
            body: Box::pin(response.bytes_stream()),
            parser: SseParser::new(),
            stats: StreamStats {
                started: Some(started),
                ..Default::default()
            },
            cancellation,
            cancelled,
            finished: false,
            last_event_id: last_event_id.filter(|id| !id.is_empty()).map(str::to_owned),
        })
    }

    async fn stream_with_options(
        &self,
        request: &mut ChatRequest,
        cancellation: Option<tokio_util::sync::CancellationToken>,
        last_event_id: Option<String>,
    ) -> Result<ChatStream<impl Stream<Item = Result<Bytes, reqwest::Error>> + Send>, StreamingError>
    {
        request.stream = true;
        let started = Instant::now();

        let mut builder = self
            .client
            .post(self.chat_url.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .header(reqwest::header::CACHE_CONTROL, "no-cache")
            .header(reqwest::header::ACCEPT_ENCODING, "identity");

        if let Some(id) = last_event_id.as_deref().filter(|id| !id.is_empty()) {
            builder = builder.header("Last-Event-ID", id);
        }

        let response = match cancellation.as_ref() {
            Some(token) => tokio::select! {
                _ = token.cancelled() => return Err(StreamingError::Cancelled),
                result = builder.json(request).send() => result?,
            },
            None => builder.json(request).send().await?,
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StreamingError::HttpStatus { status, body });
        }

        let cancellation = cancellation.unwrap_or_else(tokio_util::sync::CancellationToken::new);
        let cancelled = cancellation.clone().cancelled_owned();

        Ok(ChatStream {
            body: response.bytes_stream(),
            parser: SseParser::new(),
            stats: StreamStats {
                started: Some(started),
                ..Default::default()
            },
            cancellation,
            cancelled,
            finished: false,
            last_event_id: last_event_id.filter(|id| !id.is_empty()),
        })
    }
}

type DynBody = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

fn is_retryable_stream_error(error: &StreamingError) -> bool {
    match error {
        StreamingError::Http(error) => {
            error.is_connect() || error.is_timeout() || error.is_request()
        }
        StreamingError::HttpStatus { status, .. } => {
            matches!(status.as_u16(), 408 | 425 | 429 | 500..=599)
        }
        StreamingError::UnexpectedEof => true,
        _ => false,
    }
}

pin_project_lite::pin_project! {
    pub struct ChatStream<S> {
        #[pin]
        body: S,
        parser: SseParser,
        stats: StreamStats,
        cancellation: tokio_util::sync::CancellationToken,
        #[pin]
        cancelled: tokio_util::sync::WaitForCancellationFutureOwned,
        finished: bool,
        last_event_id: Option<String>,
    }
}

impl<S> ChatStream<S> {
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl<S> Stream for ChatStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>>,
{
    type Item = Result<StreamEvent, StreamingError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        if *this.finished {
            return Poll::Ready(None);
        }

        if this.cancelled.as_mut().poll(cx).is_ready() {
            this.stats.finished = Some(Instant::now());
            *this.finished = true;
            return Poll::Ready(Some(Err(StreamingError::Cancelled)));
        }

        loop {
            let parsed = this.parser.next_event();

            if let Some(id) = this.parser.last_event_id() {
                *this.last_event_id = (!id.is_empty()).then_some(id.to_owned());
            }

            if let Some(result) = parsed {
                match result {
                    Ok(StreamEvent::Done) => {
                        this.stats.finished = Some(Instant::now());
                        *this.finished = true;
                        return Poll::Ready(Some(Ok(StreamEvent::Done)));
                    }
                    Ok(StreamEvent::Chunk(chunk)) => {
                        this.stats.chunks += 1;

                        for choice in &chunk.choices {
                            if let Some(text) = &choice.delta.content {
                                if !text.is_empty() {
                                    this.stats.first_output.get_or_insert_with(Instant::now);
                                    this.stats.content_bytes += text.len() as u64;
                                    this.stats.output_events += 1;
                                }
                            }

                            if let Some(text) = &choice.delta.reasoning {
                                if !text.is_empty() {
                                    this.stats.first_output.get_or_insert_with(Instant::now);
                                    this.stats.reasoning_bytes += text.len() as u64;
                                }
                            }
                        }

                        return Poll::Ready(Some(Ok(StreamEvent::Chunk(chunk))));
                    }
                    Err(error) => {
                        this.stats.finished = Some(Instant::now());
                        *this.finished = true;
                        return Poll::Ready(Some(Err(error)));
                    }
                }
            }

            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Err(error) = this.parser.push(&bytes) {
                        this.stats.finished = Some(Instant::now());
                        *this.finished = true;
                        return Poll::Ready(Some(Err(error)));
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    this.stats.finished = Some(Instant::now());
                    *this.finished = true;
                    return Poll::Ready(Some(Err(StreamingError::Http(error))));
                }
                Poll::Ready(None) => {
                    this.stats.finished = Some(Instant::now());
                    *this.finished = true;
                    return Poll::Ready(Some(Err(StreamingError::UnexpectedEof)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pub struct SseParser {
    buffer: BytesMut,
    scan_pos: usize,
    last_event_id: Option<String>,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(READ_BUFFER_CAPACITY),
            scan_pos: 0,
            last_event_id: None,
        }
    }

    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<(), StreamingError> {
        let old_len = self.buffer.len();
        self.buffer.extend_from_slice(bytes);

        if self.trailing_incomplete_event_len() > MAX_EVENT_BYTES {
            self.buffer.truncate(old_len);
            return Err(StreamingError::EventTooLarge);
        }

        Ok(())
    }

    fn trailing_incomplete_event_len(&self) -> usize {
        find_last_boundary_end(&self.buffer).map_or(self.buffer.len(), |end| {
            self.buffer.len().saturating_sub(end)
        })
    }

    pub fn next_event(&mut self) -> Option<Result<StreamEvent, StreamingError>> {
        loop {
            let (frame_len, separator_len) = find_boundary(&self.buffer, &mut self.scan_pos)?;
            let frame = self.buffer.split_to(frame_len);
            self.buffer.advance(separator_len);
            self.scan_pos = 0;

            if let Some(event_id) = parse_event_id(&frame) {
                self.last_event_id = Some(event_id);
            }

            match parse_frame(&frame) {
                None => continue,
                Some(result) => return Some(result),
            }
        }
    }
}

fn parse_frame(frame: &[u8]) -> Option<Result<StreamEvent, StreamingError>> {
    let mut data_lines = 0usize;
    let mut single_data: &[u8] = &[];
    let mut data_len = 0usize;

    for line in SseLineIter::new(frame) {
        if line.is_empty() || line[0] == b':' {
            continue;
        }

        let Some(colon) = line.iter().position(|b| *b == b':') else {
            continue;
        };

        if &line[..colon] != b"data" {
            continue;
        }

        let value = line.get(colon + 1..).unwrap_or_default();
        let value = value.strip_prefix(b" ").unwrap_or(value);
        data_lines += 1;
        data_len += value.len();

        if data_lines == 1 {
            single_data = value;
        }
    }

    if data_lines == 0 {
        return None;
    }

    if data_lines == 1 {
        if single_data == b"[DONE]" {
            return Some(Ok(StreamEvent::Done));
        }

        return Some(
            serde_json::from_slice::<ChatChunk>(single_data)
                .map(StreamEvent::Chunk)
                .map_err(StreamingError::Json),
        );
    }

    let mut data = Vec::with_capacity(data_len + data_lines - 1);
    let mut first = true;

    for line in SseLineIter::new(frame) {
        if line.is_empty() || line[0] == b':' {
            continue;
        }

        let Some(colon) = line.iter().position(|b| *b == b':') else {
            continue;
        };

        if &line[..colon] != b"data" {
            continue;
        }

        let value = line.get(colon + 1..).unwrap_or_default();
        let value = value.strip_prefix(b" ").unwrap_or(value);

        if !first {
            data.push(b'\n');
        }
        first = false;
        data.extend_from_slice(value);
    }

    if data == b"[DONE]" {
        return Some(Ok(StreamEvent::Done));
    }

    match serde_json::from_slice::<ChatChunk>(&data) {
        Ok(chunk) => Some(Ok(StreamEvent::Chunk(chunk))),
        Err(error) => Some(Err(StreamingError::Json(error))),
    }
}

fn parse_event_id(frame: &[u8]) -> Option<String> {
    let mut event_id = None;

    for line in SseLineIter::new(frame) {
        let (field, value) = match line.iter().position(|b| *b == b':') {
            Some(colon) => (&line[..colon], line.get(colon + 1..).unwrap_or_default()),
            None => (line, &[][..]),
        };

        if field != b"id" {
            continue;
        }

        let value = value.strip_prefix(b" ").unwrap_or(value);
        if value.contains(&0) {
            continue;
        }

        event_id = Some(String::from_utf8_lossy(value).into_owned());
    }

    event_id
}

struct SseLineIter<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> SseLineIter<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

impl<'a> Iterator for SseLineIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.bytes.len() {
            return None;
        }

        let start = self.pos;
        let mut i = start;

        while i < self.bytes.len() && self.bytes[i] != b'\n' && self.bytes[i] != b'\r' {
            i += 1;
        }

        let line = &self.bytes[start..i];
        if i >= self.bytes.len() {
            self.pos = i;
            return Some(line);
        }

        self.pos = if self.bytes[i] == b'\r' && self.bytes.get(i + 1) == Some(&b'\n') {
            i + 2
        } else {
            i + 1
        };

        Some(line)
    }
}

fn find_last_boundary_end(buffer: &BytesMut) -> Option<usize> {
    let bytes = buffer.as_ref();
    let mut end = bytes.len();

    while end > 0 {
        if end >= 4 && bytes[end - 4..end] == *b"\r\n\r\n" {
            return Some(end);
        }
        if end >= 3 && bytes[end - 3..end] == *b"\r\n\n" {
            return Some(end);
        }
        if end >= 3 && bytes[end - 3..end] == *b"\n\r\n" {
            return Some(end);
        }
        if end >= 2 && bytes[end - 2..end] == *b"\n\n" {
            return Some(end);
        }
        if end >= 2 && bytes[end - 2..end] == *b"\r\r" {
            return Some(end);
        }
        end -= 1;
    }

    None
}

fn find_boundary(buffer: &BytesMut, scan_pos: &mut usize) -> Option<(usize, usize)> {
    let bytes = buffer.as_ref();
    let mut i = (*scan_pos).min(bytes.len());

    while i < bytes.len() {
        let first_eol = if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
            2
        } else if matches!(bytes[i], b'\r' | b'\n') {
            1
        } else {
            i += 1;
            continue;
        };

        let second = i + first_eol;
        if second < bytes.len() {
            let second_eol = if bytes[second] == b'\r' && bytes.get(second + 1) == Some(&b'\n') {
                2
            } else if matches!(bytes[second], b'\r' | b'\n') {
                1
            } else {
                i += first_eol;
                continue;
            };

            *scan_pos = i;
            return Some((i, first_eol + second_eol));
        }

        break;
    }

    *scan_pos = i.saturating_sub(1);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> SseParser {
        SseParser::new()
    }

    #[test]
    fn parses_split_frame() {
        let mut p = parser();
        p.push(b"data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel")
            .unwrap();
        assert!(p.next_event().is_none());

        p.push(b"lo\"}}]}\n\n").unwrap();
        match p.next_event().unwrap().unwrap() {
            StreamEvent::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
            }
            StreamEvent::Done => panic!(),
        }
    }

    #[test]
    fn ignores_comments_and_parses_multiline_data() {
        let mut p = parser();
        p.push(b": heartbeat\nevent: message\ndata: {\"id\":\"x\",\ndata: \"choices\":[]}\n\n")
            .unwrap();

        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
    }

    #[test]
    fn tracks_and_clears_sse_event_id() {
        let mut p = parser();
        p.push(
            br#"id: one
data: {"id":"x","choices":[]}

"#,
        )
        .unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        assert_eq!(p.last_event_id(), Some("one"));

        p.push(b"data: {\"id\":\"x2\",\"choices\":[]}\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        assert_eq!(p.last_event_id(), Some("one"));

        p.push(b"id:\ndata: [DONE]\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
        assert_eq!(p.last_event_id(), Some(""));
    }

    #[test]
    fn preserves_id_only_event_cursor() {
        let mut p = parser();
        p.push(b"id: cursor-only\n\n").unwrap();
        assert!(p.next_event().is_none());
        assert_eq!(p.last_event_id(), Some("cursor-only"));

        p.push(b"data: [DONE]\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
        assert_eq!(p.last_event_id(), Some("cursor-only"));
    }

    #[test]
    fn parses_lone_cr_and_mixed_line_endings() {
        let mut p = parser();
        p.push(b"id: 7\rdata: {\"id\":\"x\",\"choices\":[]}\r\r")
            .unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        assert_eq!(p.last_event_id(), Some("7"));

        p.push(b"data: [DONE]\r\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
    }

    #[test]
    fn uses_last_id_field_and_ignores_null_id() {
        let mut p = parser();
        p.push(b"id: first\nid: second\ndata: {\"id\":\"x\",\"choices\":[]}\n\n")
            .unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        assert_eq!(p.last_event_id(), Some("second"));

        p.push(b"id: bad\0id\ndata: [DONE]\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
        assert_eq!(p.last_event_id(), Some("second"));
    }

    #[test]
    fn parses_crlf_and_done() {
        let mut p = parser();
        p.push(b"data: {\"id\":\"x\",\"choices\":[]}\r\n\r\n")
            .unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));

        p.push(b"data: [DONE]\r\n\r\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
    }

    #[test]
    fn skips_empty_and_comment_only_frames_without_recursion() {
        let mut p = parser();
        p.push(b": one\n\n: two\n\ndata: [DONE]\n\n").unwrap();
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
    }

    #[test]
    fn parses_usage_only_chunk() {
        let mut p = parser();
        p.push(
            br#"data: {"id":"x","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

"#,
        )
        .unwrap();

        match p.next_event().unwrap().unwrap() {
            StreamEvent::Chunk(chunk) => {
                assert_eq!(chunk.usage.unwrap().total_tokens, Some(15));
            }
            StreamEvent::Done => panic!(),
        }
    }

    #[test]
    fn allows_many_complete_events_beyond_event_limit() {
        let mut p = parser();
        let event = b"data: [DONE]\n\n";
        let count = MAX_EVENT_BYTES / event.len() + 2;

        for _ in 0..count {
            p.push(event).unwrap();
        }

        for _ in 0..count {
            assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
        }
    }

    #[test]
    fn rejects_oversized_incomplete_event() {
        let mut p = parser();
        let data = vec![b'x'; MAX_EVENT_BYTES];
        assert!(p.push(&data).is_ok());
        assert!(matches!(p.push(b"x"), Err(StreamingError::EventTooLarge)));
    }

    #[test]
    fn boundary_scan_survives_incremental_pushes() {
        let mut p = parser();
        for byte in b"data: [DONE]\n\n" {
            p.push(std::slice::from_ref(byte)).unwrap();
            if *byte == b'\n' {
                let _ = p.next_event();
            }
        }
        assert!(p.next_event().is_none());
    }

    #[tokio::test]
    async fn chat_stream_cancel_is_available_on_plain_stream() {
        let cancellation = tokio_util::sync::CancellationToken::new();
        let cancelled = cancellation.clone().cancelled_owned();

        let body = futures_util::stream::empty::<Result<Bytes, reqwest::Error>>();
        let mut stream = ChatStream {
            body,
            parser: SseParser::new(),
            stats: StreamStats::default(),
            cancellation,
            cancelled,
            finished: false,
            last_event_id: None,
        };

        stream.cancel();
        let result = stream.next().await.unwrap();
        assert!(matches!(result, Err(StreamingError::Cancelled)));
        assert!(stream.next().await.is_none());
    }

    proptest::proptest! {
        #[test]
        fn arbitrary_bytes_never_panic(input in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..8192)) {
            let mut p = parser();
            for chunk in input.chunks(37) {
                let _ = p.push(chunk);
                let _ = p.next_event();
            }
            let _ = p.next_event();
        }

        #[test]
        fn valid_event_survives_all_split_points(split in 0usize..=70) {
            const EVENT: &[u8] = br#"data: {"id":"x","choices":[{"index":0,"delta":{"content":"hello"}}]}

"#;
            let split = split.min(EVENT.len());
            let mut p = parser();
            p.push(&EVENT[..split]).unwrap();
            if split < EVENT.len() {
                assert!(p.next_event().is_none());
            }
            p.push(&EVENT[split..]).unwrap();
            assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        }
    }
}
