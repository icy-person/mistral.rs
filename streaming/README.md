# mistral.rs streaming

A standalone, low-overhead OpenAI-compatible SSE streaming client for mistral.rs.

## Architecture

The hot path is deliberately small:

1. `reqwest::Response::bytes_stream()` receives network chunks incrementally.
2. `BytesMut` buffers only incomplete SSE frames.
3. A scan cursor prevents repeatedly rescanning the same incomplete bytes.
4. SSE framing supports both `LF` and `CRLF`, comments, and multiple `data:` lines.
5. Single-line `data:` frames go directly from bytes into `serde_json`, avoiding an intermediate `String`; multiline frames are assembled only when required.
6. Complete JSON payloads are deserialized into structured `ChatChunk` values.
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
- TTFT starts at request dispatch and is recorded on the first non-empty content or reasoning delta.
- No token-by-token `String` allocation in the parser hot path for ordinary single-line SSE frames.
- `stream` is enforced to `true` by the streaming API, preventing accidental non-stream requests.
- Optional Bearer authentication is supported by the builder.

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

## CLI flags

The CLI is streaming-first: it flushes each received delta immediately. Options are parsed by
Clap, and the endpoint/model can be configured by flags or environment variables.

```bash
# Basic streaming (reasoning is hidden by default)
cargo run --release -- --model Qwen3.5-2B "سلام، خودت را معرفی کن"

# Show reasoning and timing/throughput statistics
cargo run --release -- --show-reasoning --stats "Explain mixture-of-experts models"

# Tune generation
cargo run --release -- --max-tokens 1024 --temperature 0.2 --top-p 0.9 "Write a concise summary"

# JSON Lines events, suitable for piping into another program
cargo run --release -- --output jsonl "Hello"

# Reconnect only when the server supports SSE event IDs and replay
cargo run --release -- --reconnect --max-retries 5 "Continue this task"
```

Supported flags include `--base-url`, `--model`, `--api-key`, `--max-tokens`,
`--temperature`, `--top-p`, `--show-reasoning`, `--stats`, `--output text|jsonl`,
`--reconnect`, and `--max-retries`. The `MISTRALRS_BASE_URL`, `MODEL`, and
`OPENAI_API_KEY` environment variables are also supported. Use `--help` for the full list.

## Library usage

    use futures_util::StreamExt;
    use mistralrs_streaming::{ChatRequest, Message, StreamEvent, StreamingClient};

    let client = StreamingClient::builder()
        .base_url("http://127.0.0.1:1234/v1")
        // Optional for OpenAI-compatible servers that require authentication.
        // .bearer_token(std::env::var("OPENAI_API_KEY")?)
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
            StreamEvent::Done => break,
        }
    }

## Validation

    cargo fmt --check
    cargo test --release
    cargo clippy --all-targets -- -D warnings

For low-level latency work, run release mode and compare TTFT and end-to-end elapsed time over multiple identical prompts rather than relying on a single sample.


## Extended streaming support

The client now exposes the major Chat Completions streaming controls without changing the hot path:

- typed streamed tool-call deltas (tool_calls[].function.name/arguments)
- refusal deltas
- stream_options.include_usage and include_obfuscation
- request controls for penalties, stop, tools, tool choice, response format, parallel tool calls, and seed
- explicit cancellation through tokio_util::sync::CancellationToken
- ChatStream::cancel() convenience cancellation
- SSE id: tracking and last_event_id()
- stream_with_resume(...), which sends Last-Event-ID for servers that implement SSE resume semantics
- stream_with_reconnect(...), with bounded exponential backoff and automatic resume after transport/EOF failures
- automatic reconnect is enabled only after a server-provided SSE id has been observed, preventing unsafe replay when the server cannot provide a resume cursor
- generic ChatStream<S>; the normal reqwest body stream is no longer boxed, removing the previous dynamic-dispatch layer from the normal streaming hot path

Resume is deliberately cursor-based: the SSE server must emit event IDs and support replay after Last-Event-ID. The reconnect API does not restart a partially consumed generation from scratch, because doing so could duplicate model output. This follows the SSE reconnection contract, where Last-Event-ID identifies the last successfully dispatched event.

## Benchmarks and parser fuzz coverage

Release benchmarks are available with:

    cargo bench

The benchmark suite covers single-line SSE, CRLF framing, and deliberately fragmented byte-at-a-time input.

Property-based tests exercise arbitrary byte streams and all tested split points of valid SSE events:

    cargo test --release --all-targets

This combination is intended to catch parser panics, framing regressions, and incremental-boundary bugs while keeping the production parser free of fuzzing overhead.
