use clap::{Parser, ValueEnum};
use futures_core::Stream;
use futures_util::StreamExt;
use mistralrs_streaming::{
    ChatRequest, Message, ReconnectPolicy, StreamEvent, StreamingClient, StreamingError,
};
use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Jsonl,
}

#[derive(Debug, Parser)]
#[command(
    name = "mistralrs-streaming",
    version,
    about = "Low-overhead streaming client for an OpenAI-compatible mistral.rs endpoint",
    after_help = "Examples:\n  mistralrs-streaming --model Qwen3.5-2B --stats 'سلام'\n  mistralrs-streaming --base-url http://127.0.0.1:1234/v1 --show-reasoning 'Explain MoE'\n  mistralrs-streaming --reconnect --max-retries 5 'Write a short poem'"
)]
struct Args {
    /// Prompt to send. Quote the prompt to preserve spaces.
    #[arg(required = true, num_args = 1.., value_name = "PROMPT")]
    prompt: Vec<String>,

    /// OpenAI-compatible API base URL.
    #[arg(long, env = "MISTRALRS_BASE_URL", default_value = "http://127.0.0.1:1234/v1")]
    base_url: String,

    /// Model identifier accepted by the server.
    #[arg(long, env = "MODEL", default_value = "default")]
    model: String,

    /// Bearer token (defaults to OPENAI_API_KEY if set).
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    /// Maximum number of generated tokens.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..))]
    max_tokens: u32,

    /// Sampling temperature.
    #[arg(long, value_parser = clap::value_parser!(f32).range(0.0..=2.0))]
    temperature: Option<f32>,

    /// Nucleus sampling probability.
    #[arg(long, value_parser = clap::value_parser!(f32).range(0.0..=1.0))]
    top_p: Option<f32>,

    /// Display reasoning deltas as they arrive (hidden by default).
    #[arg(long)]
    show_reasoning: bool,

    /// Print TTFT, elapsed time, chunk count and output throughput to stderr.
    #[arg(long)]
    stats: bool,

    /// Output each event as JSON Lines instead of rendering plain text.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,

    /// Resume from the last SSE event ID after retryable transport failures.
    /// Requires the server to emit event IDs and support Last-Event-ID replay.
    #[arg(long)]
    reconnect: bool,

    /// Maximum reconnect attempts (only used with --reconnect).
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32))]
    max_retries: u32,
}

fn build_request(args: &Args) -> ChatRequest {
    ChatRequest {
        model: args.model.clone(),
        messages: vec![Message {
            role: "user".into(),
            content: args.prompt.join(" "),
        }],
        stream: true,
        max_tokens: Some(args.max_tokens),
        temperature: args.temperature,
        top_p: args.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        response_format: None,
        seed: None,
        stream_options: None,
    }
}

fn print_jsonl(value: serde_json::Value) -> anyhow::Result<()> {
    serde_json::to_writer(io::stdout().lock(), &value)?;
    println!();
    io::stdout().flush()?;
    Ok(())
}

async fn consume_stream<S>(mut stream: S, args: &Args, started: Instant) -> anyhow::Result<()>
where
    S: Stream<Item = Result<StreamEvent, StreamingError>> + Unpin,
{
    let mut first_output: Option<Duration> = None;
    let mut chunks = 0u64;
    let mut content_bytes = 0u64;
    let mut reasoning_bytes = 0u64;
    let mut finish_reason: Option<String> = None;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Chunk(chunk) => {
                chunks += 1;
                let usage = chunk.usage;
                for choice in chunk.choices {
                    if let Some(reasoning) = choice.delta.reasoning {
                        if !reasoning.is_empty() && first_output.is_none() {
                            first_output = Some(started.elapsed());
                        }
                        reasoning_bytes += reasoning.len() as u64;
                        match args.output {
                            OutputFormat::Text if args.show_reasoning => {
                                print!("{reasoning}");
                                io::stdout().flush()?;
                            }
                            OutputFormat::Jsonl => {
                                print_jsonl(serde_json::json!({
                                    "type": "reasoning",
                                    "text": reasoning
                                }))?;
                            }
                            _ => {}
                        }
                    }

                    if let Some(content) = choice.delta.content {
                        if !content.is_empty() && first_output.is_none() {
                            first_output = Some(started.elapsed());
                        }
                        content_bytes += content.len() as u64;
                        match args.output {
                            OutputFormat::Text => {
                                print!("{content}");
                                io::stdout().flush()?;
                            }
                            OutputFormat::Jsonl => {
                                print_jsonl(serde_json::json!({
                                    "type": "content",
                                    "text": content
                                }))?;
                            }
                        }
                    }

                    if let Some(refusal) = choice.delta.refusal {
                        match args.output {
                            OutputFormat::Text => eprint!("\n[refusal] {refusal}\n"),
                            OutputFormat::Jsonl => print_jsonl(serde_json::json!({
                                "type": "refusal",
                                "text": refusal
                            }))?,
                        }
                    }

                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tool in tool_calls {
                            let function = tool.function;
                            let event = serde_json::json!({
                                "type": "tool_call_delta",
                                "index": tool.index,
                                "id": tool.id,
                                "name": function.as_ref().and_then(|f| f.name.clone()),
                                "arguments": function.and_then(|f| f.arguments),
                            });
                            if matches!(args.output, OutputFormat::Jsonl) {
                                print_jsonl(event)?;
                            } else {
                                eprintln!("\n[tool call delta] {event}");
                            }
                        }
                    }

                    if choice.finish_reason.is_some() {
                        finish_reason = choice.finish_reason;
                    }
                }
                if matches!(args.output, OutputFormat::Jsonl) {
                    if let Some(usage) = usage {
                        print_jsonl(serde_json::json!({
                            "type": "usage",
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }))?;
                    }
                }
            }
            StreamEvent::Done => break,
        }
    }

    let elapsed = started.elapsed();
    if matches!(args.output, OutputFormat::Text) {
        println!();
    }
    if args.stats {
        let seconds = elapsed.as_secs_f64();
        let bytes_per_second = if seconds > 0.0 {
            content_bytes as f64 / seconds
        } else {
            0.0
        };
        eprintln!("\nStreaming stats:");
        eprintln!(
            "  TTFT: {}",
            first_output
                .map(|d| format!("{:.3}s", d.as_secs_f64()))
                .unwrap_or_else(|| "n/a".into())
        );
        eprintln!("  Elapsed: {:.3}s", seconds);
        eprintln!("  Chunks: {chunks}");
        eprintln!("  Content: {content_bytes} bytes");
        eprintln!("  Reasoning: {reasoning_bytes} bytes");
        eprintln!("  Content throughput: {:.1} bytes/s", bytes_per_second);
        eprintln!(
            "  Finish reason: {}",
            finish_reason.as_deref().unwrap_or("unknown")
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut builder = StreamingClient::builder().base_url(args.base_url.clone());
    if let Some(token) = args.api_key.as_deref() {
        builder = builder.bearer_token(token);
    }
    let client = builder.build()?;
    let request = build_request(&args);
    let started = Instant::now();

    if args.reconnect {
        let policy = ReconnectPolicy {
            max_retries: args.max_retries,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(4),
        };
        let stream = client.stream_with_reconnect(request, policy).await?;
        consume_stream(Box::pin(stream), &args, started).await?;
    } else {
        let stream = client.stream(request).await?;
        consume_stream(Box::pin(stream), &args, started).await?;
    }
    Ok(())
}
