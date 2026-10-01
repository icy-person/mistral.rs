use bytes::{Buf, Bytes, BytesMut};
use futures_core::Stream;
use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::{
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

#[derive(Debug, Clone)]
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
    pub tool_calls: Option<Vec<serde_json::Value>>,
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
}

#[derive(Debug, Error)]
pub enum StreamingError {
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
    #[error("SSE data is not valid UTF-8")]
    InvalidUtf8,
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
        } else {
            builder = builder.timeout(None);
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

    pub async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream, StreamingError> {
        request.stream = true;
        let started = Instant::now();

        let mut request_builder = self
            .client
            .post(self.chat_url.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .header(reqwest::header::CACHE_CONTROL, "no-cache")
            .header(reqwest::header::ACCEPT_ENCODING, "identity");

        // Authorization is intentionally attached per request so the client itself
        // remains cloneable without exposing credentials through Debug.
        if let Some(token) = self.client.default_headers().get(reqwest::header::AUTHORIZATION) {
            request_builder = request_builder.header(reqwest::header::AUTHORIZATION, token);
        }

        let response = request_builder.json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(StreamingError::HttpStatus { status, body });
        }

        Ok(ChatStream {
            body: Box::pin(response.bytes_stream()),
            parser: SseParser::new(),
            stats: StreamStats {
                started: Some(started),
                ..Default::default()
            },
            saw_done: false,
        })
    }
}

pub struct ChatStream {
    body: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseParser,
    stats: StreamStats,
    saw_done: bool,
}

impl ChatStream {
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }
}

impl Stream for ChatStream {
    type Item = Result<StreamEvent, StreamingError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(result) = self.parser.next_event() {
                match result {
                    Ok(StreamEvent::Done) => {
                        self.saw_done = true;
                        self.stats.finished = Some(Instant::now());
                        return Poll::Ready(Some(Ok(StreamEvent::Done)));
                    }
                    Ok(StreamEvent::Chunk(chunk)) => {
                        self.stats.chunks += 1;

                        for choice in &chunk.choices {
                            if let Some(text) = &choice.delta.content {
                                if !text.is_empty() {
                                    self.stats.first_output.get_or_insert_with(Instant::now);
                                    self.stats.content_bytes += text.len() as u64;
                                    self.stats.output_events += 1;
                                }
                            }

                            if let Some(text) = &choice.delta.reasoning {
                                if !text.is_empty() {
                                    self.stats.first_output.get_or_insert_with(Instant::now);
                                    self.stats.reasoning_bytes += text.len() as u64;
                                }
                            }
                        }

                        return Poll::Ready(Some(Ok(StreamEvent::Chunk(chunk))));
                    }
                    Err(error) => return Poll::Ready(Some(Err(error))),
                }
            }

            match self.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    if let Err(error) = self.parser.push(&bytes) {
                        return Poll::Ready(Some(Err(error)));
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    return Poll::Ready(Some(Err(StreamingError::Http(error))));
                }
                Poll::Ready(None) => {
                    self.stats.finished.get_or_insert_with(Instant::now);
                    if self.saw_done {
                        return Poll::Ready(None);
                    }
                    return Poll::Ready(Some(Err(StreamingError::UnexpectedEof)));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

struct SseParser {
    buffer: BytesMut,
    scan_pos: usize,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(READ_BUFFER_CAPACITY),
            scan_pos: 0,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<(), StreamingError> {
        if self.buffer.len().saturating_add(bytes.len()) > MAX_EVENT_BYTES {
            return Err(StreamingError::EventTooLarge);
        }
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }

    fn next_event(&mut self) -> Option<Result<StreamEvent, StreamingError>> {
        loop {
            let (frame_len, separator_len) = find_boundary(&self.buffer, &mut self.scan_pos)?;
            let frame = self.buffer.split_to(frame_len);
            self.buffer.advance(separator_len);
            self.scan_pos = 0;

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
    let mut multiline = false;
    let mut data_len = 0usize;

    for raw_line in frame.split(|b| *b == b'\n') {
        let line = raw_line.strip_suffix(b'\r').unwrap_or(raw_line);
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
        } else {
            multiline = true;
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

    for raw_line in frame.split(|b| *b == b'\n') {
        let line = raw_line.strip_suffix(b'\r').unwrap_or(raw_line);
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

    let _ = multiline;
    if data == b"[DONE]" {
        return Some(Ok(StreamEvent::Done));
    }

    match serde_json::from_slice::<ChatChunk>(&data) {
        Ok(chunk) => Some(Ok(StreamEvent::Chunk(chunk))),
        Err(error) => Some(Err(StreamingError::Json(error))),
    }
}

fn find_boundary(buffer: &BytesMut, scan_pos: &mut usize) -> Option<(usize, usize)> {
    let bytes = buffer.as_ref();
    let mut i = (*scan_pos).min(bytes.len());

    while i + 1 < bytes.len() {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            *scan_pos = i;
            return Some((i, 2));
        }

        if i + 3 < bytes.len()
            && bytes[i] == b'\r'
            && bytes[i + 1] == b'\n'
            && bytes[i + 2] == b'\r'
            && bytes[i + 3] == b'\n'
        {
            *scan_pos = i;
            return Some((i, 4));
        }

        i += 1;
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
        p.push(
            b": heartbeat\nevent: message\ndata: {\"id\":\"x\",\ndata: \"choices\":[]}\n\n",
        )
        .unwrap();

        assert!(matches!(
            p.next_event(),
            Some(Ok(StreamEvent::Chunk(_)))
        ));
    }

    #[test]
    fn parses_crlf_and_done() {
        let mut p = parser();
        p.push(b"data: {\"id\":\"x\",\"choices\":[]}\r\n\r\n").unwrap();
        assert!(matches!(
            p.next_event(),
            Some(Ok(StreamEvent::Chunk(_)))
        ));

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
}
