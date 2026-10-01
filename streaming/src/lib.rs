use bytes::{Bytes, BytesMut};
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
}

impl StreamStats {
    pub fn ttft(&self) -> Option<Duration> {
        Some(self.first_output?.duration_since(self.started?))
    }

    pub fn elapsed(&self) -> Option<Duration> {
        Some(self.finished?.duration_since(self.started?))
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
    #[error("server error: {0}")]
    Server(String),
}

impl Default for StreamingClientBuilder {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            request_timeout: None,
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

    pub async fn stream(&self, request: ChatRequest) -> Result<ChatStream, StreamingError> {
        let started = Instant::now();
        let response = self
            .client
            .post(self.chat_url.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .header(reqwest::header::CACHE_CONTROL, "no-cache")
            .json(&request)
            .send()
            .await?;

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
                        let output = chunk.choices.iter().map(|choice| {
                            choice.delta.content.as_deref().unwrap_or("")
                        }).collect::<String>();
                        if !output.is_empty() && self.stats.first_output.is_none() {
                            self.stats.first_output = Some(Instant::now());
                        }
                        self.stats.content_bytes += output.len() as u64;
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
                    self.stats.finished = Some(Instant::now());
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
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(READ_BUFFER_CAPACITY),
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<(), StreamingError> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() > MAX_EVENT_BYTES {
            return Err(StreamingError::EventTooLarge);
        }
        Ok(())
    }

    fn next_event(&mut self) -> Option<Result<StreamEvent, StreamingError>> {
        let boundary = find_boundary(&self.buffer)?;
        let frame_len = boundary.0;
        let separator_len = boundary.1;
        let frame = self.buffer.split_to(frame_len);
        self.buffer.advance(separator_len);

        let mut data = String::new();
        for raw_line in frame.split(|b| *b == b'\n') {
            let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
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
            if !data.is_empty() {
                data.push('\n');
            }
            match std::str::from_utf8(value) {
                Ok(value) => data.push_str(value),
                Err(_) => {
                    return Some(Err(StreamingError::Server(
                        "SSE data is not UTF-8".into(),
                    )))
                }
            }
        }

        if data.is_empty() {
            return self.next_event();
        }

        if data == "[DONE]" {
            return Some(Ok(StreamEvent::Done));
        }

        match serde_json::from_str::<ChatChunk>(&data) {
            Ok(chunk) => Some(Ok(StreamEvent::Chunk(chunk))),
            Err(error) => Some(Err(error.into())),
        }
    }
}

fn find_boundary(buffer: &BytesMut) -> Option<(usize, usize)> {
    let bytes = buffer.as_ref();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < bytes.len()
            && bytes[i] == b'\r'
            && bytes[i + 1] == b'\n'
            && bytes[i + 2] == b'\r'
            && bytes[i + 3] == b'\n'
        {
            return Some((i, 4));
        }
    }
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
        p.push(b"data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel");
        assert!(p.next_event().is_none());
        p.push(b"lo\"}}]}\n\n");
        match p.next_event().unwrap().unwrap() {
            StreamEvent::Chunk(chunk) => assert_eq!(
                chunk.choices[0].delta.content.as_deref(),
                Some("hello")
            ),
            StreamEvent::Done => panic!(),
        }
    }

    #[test]
    fn parses_crlf_and_done() {
        let mut p = parser();
        p.push(b"data: {\"id\":\"x\",\"choices\":[]}\r\n\r\n");
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Chunk(_)))));
        p.push(b"data: [DONE]\r\n\r\n");
        assert!(matches!(p.next_event(), Some(Ok(StreamEvent::Done))));
    }
}
