# mistral.rs streaming

Standalone low-overhead OpenAI-compatible SSE streaming client.

## Properties

- Incremental HTTP body consumption with no polling thread.
- Zero per-token task spawning.
- BytesMut frame buffering.
- TCP_NODELAY and HTTP/2 adaptive windows.
- Unlimited request timeout by default for long generations.
- 8 MiB SSE frame limit.
- UTF-8 validation at complete SSE data fields.
- Content, reasoning, tool calls, finish reasons and usage are exposed.
- Dropping ChatStream cancels the underlying request.
- TTFT and total elapsed time are measured locally.

## Run

Start mistral.rs on the OpenAI-compatible endpoint, then:

    cd streaming
    cargo run --release -- "سلام، خودت را معرفی کن"

Optional endpoint/model:

    MISTRALRS_BASE_URL=http://127.0.0.1:1234/v1 MODEL=Qwen3.5-2B cargo run --release -- "Hello"

## Validate

    cargo fmt --check
    cargo test --release
    cargo clippy --all-targets -- -D warnings
