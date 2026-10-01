# mistral.rs streaming

A standalone, low-overhead OpenAI-compatible SSE streaming client for mistral.rs.

## Architecture

The hot path is deliberately small:

1. `reqwest::Response::bytes_stream()` receives network chunks incrementally.
2. `BytesMut` buffers only incomplete SSE frames.
3. SSE framing supports both `LF` and `CRLF`, comments, and multiple `data:` lines.
4. Complete JSON payloads are deserialized into structured `ChatChunk` values.
5. The async `Stream` yields immediately; no polling thread or per-token task is created.

The client is independent from the main mistral.rs workspace, so it can be benchmarked or integrated without changing the existing server path.

## Performance-oriented behavior

- `TCP_NODELAY` enabled.
- HTTP/2 adaptive flow-control window.
- Persistent connection pooling.
- No request timeout by default, suitable for long generations.
- 8 MiB maximum incomplete SSE frame to bound memory use.
- Backpressure comes naturally from the consumer: the next network chunk is not polled until the caller asks for the next event.
- Dropping `ChatStream` drops the response body and cancels the HTTP request.
- TTFT starts at request dispatch and is recorded on the first non-empty generated text.
- No token-by-token `String` allocation in the stream hot path for statistics.

## API

`StreamEvent::Chunk` preserves the full OpenAI-compatible delta, including:

- `content`
- `reasoning_content` / `reasoning`
- `tool_calls`
- `finish_reason`
- `usage`
- `model`
- `id`

`StreamStats` exposes:

- TTFT
- total elapsed time
- chunk count
- content bytes
- reasoning bytes
- output bytes/second

## Run

Start mistral.rs with its OpenAI-compatible endpoint and run:

    cd streaming
    cargo run --release -- "سلام، خودت را معرفی کن"

Custom endpoint/model:

    MISTRALRS_BASE_URL=http://127.0.0.1:1234/v1 MODEL=Qwen3.5-2B cargo run --release -- "Hello"

The CLI flushes stdout after each received chunk, so generated text becomes visible immediately.

## Library usage

    use futures_util::StreamExt;
    use mistralrs_streaming::{ChatRequest, Message, StreamEvent, StreamingClient};

    let client = StreamingClient::builder()
        .base_url("http://127.0.0.1:1234/v1")
        .build()?;

    let request = ChatRequest {
        model: "Qwen3.5-2B".into(),
        messages: vec![Message {
            role: "user".into(),
            content: "Hello".into(),
        }],
        stream: true,
        max_tokens: Some(512),
        temperature: None,
        top_p: None,
    };

    let mut stream = client.stream(request).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Chunk(chunk) => {
                println!("{chunk:?}");
            }
            StreamEvent::Text { .. } | StreamEvent::Reasoning { .. } => {}
            StreamEvent::Done => break,
        }
    }

## Validation

    cargo fmt --check
    cargo test --release
    cargo clippy --all-targets -- -D warnings

For low-level latency work, run release mode and compare TTFT and end-to-end elapsed time over multiple identical prompts rather than relying on a single sample.
