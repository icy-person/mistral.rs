//! OpenAI-compatible SSE streaming client command.

use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

use anyhow::Result;
use futures_util::StreamExt;
use mistralrs_streaming::{
    ChatRequest, Message, ReconnectPolicy, StreamEvent, StreamingClient, StreamingError,
};
use tokio_util::sync::CancellationToken;

use crate::args::GlobalOptions;

pub(crate) async fn run_stream(
    base_url: String,
    model: String,
    input: String,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    connect_timeout_ms: u64,
    request_timeout_ms: Option<u64>,
    reconnect: u32,
    reconnect_backoff_ms: u64,
    reconnect_max_backoff_ms: u64,
    resume_id: Option<String>,
    no_stats: bool,
    global: GlobalOptions,
) -> Result<()> {
    if reconnect > 0 && resume_id.is_some() {
        anyhow::bail!(
            "--resume-id cannot be combined with --reconnect; reconnect resumes automatically              from SSE event ids"
        );
    }

    let mut builder = StreamingClient::builder()
        .base_url(base_url)
        .connect_timeout(Duration::from_millis(connect_timeout_ms));

    if let Some(timeout_ms) = request_timeout_ms {
        builder = builder.request_timeout(Some(Duration::from_millis(timeout_ms)));
    }

    let client = builder.build()?;
    let request = ChatRequest {
        model,
        messages: vec![Message {
            role: "user".to_owned(),
            content: input,
        }],
        stream: true,
        max_tokens: Some(max_tokens),
        temperature,
        top_p,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        response_format: None,
        seed: global.seed,
        stream_options: Some(mistralrs_streaming::StreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        }),
    };

    let cancellation = CancellationToken::new();
    install_ctrl_c_cancellation(cancellation.clone());

    if reconnect > 0 {
        let policy = ReconnectPolicy {
            max_retries: reconnect,
            initial_backoff: Duration::from_millis(reconnect_backoff_ms),
            max_backoff: Duration::from_millis(reconnect_max_backoff_ms.max(reconnect_backoff_ms)),
        };
        let stream = client
            .stream_with_reconnect_and_cancellation(request, policy, cancellation)
            .await?;
        consume_reconnect_stream(stream, no_stats).await?;
    } else {
        let mut stream = match resume_id {
            Some(id) => client
                .stream_with_resume_and_cancellation(request, id, cancellation)
                .await?,
            None => client.stream_with_cancellation(request, cancellation).await?,
        };

        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Chunk(chunk) => print_chunk(&chunk),
                StreamEvent::Done => break,
            }
        }

        io::stdout().flush()?;
        if !no_stats {
            eprintln!();
            eprintln!("TTFT: {:?}", stream.stats().ttft());
            eprintln!("Elapsed: {:?}", stream.stats().elapsed());
            eprintln!("Content bytes: {}", stream.stats().content_bytes);
            eprintln!("Reasoning bytes: {}", stream.stats().reasoning_bytes);
            eprintln!("Chunks: {}", stream.stats().chunks);
            eprintln!("Last-Event-ID: {}", stream.last_event_id().unwrap_or("<none>"));
        }
    }

    Ok(())
}

fn install_ctrl_c_cancellation(cancellation: CancellationToken) {
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        cancellation.cancel();
    });
}

fn print_chunk(chunk: &mistralrs_streaming::ChatChunk) {
    for choice in &chunk.choices {
        if let Some(reasoning) = choice.delta.reasoning.as_deref() {
            print!("{reasoning}");
        }
        if let Some(content) = choice.delta.content.as_deref() {
            print!("{content}");
        }
        if let Some(refusal) = choice.delta.refusal.as_deref() {
            print!("{refusal}");
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for call in tool_calls {
                if let Some(function) = &call.function {
                    if let Some(name) = &function.name {
                        eprint!("[tool:{name}]");
                    }
                    if let Some(arguments) = &function.arguments {
                        eprint!("{arguments}");
                    }
                }
            }
        }
    }
    io::stdout().flush().expect("stdout flush failed");
}

async fn consume_reconnect_stream(
    stream: impl futures_util::Stream<Item = Result<StreamEvent, StreamingError>>,
    no_stats: bool,
) -> Result<()> {
    let started = Instant::now();
    let mut first_output = None;
    let mut chunks = 0u64;
    let mut content_bytes = 0u64;
    let mut reasoning_bytes = 0u64;
    let mut stream = Box::pin(stream);

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Chunk(chunk) => {
                chunks += 1;
                for choice in &chunk.choices {
                    if let Some(reasoning) = choice.delta.reasoning.as_deref() {
                        if !reasoning.is_empty() {
                            first_output.get_or_insert_with(Instant::now);
                            reasoning_bytes += reasoning.len() as u64;
                        }
                    }
                    if let Some(content) = choice.delta.content.as_deref() {
                        if !content.is_empty() {
                            first_output.get_or_insert_with(Instant::now);
                            content_bytes += content.len() as u64;
                        }
                    }
                }
                print_chunk(&chunk);
            }
            StreamEvent::Done => break,
        }
    }

    io::stdout().flush()?;
    if !no_stats {
        let elapsed = started.elapsed();
        let ttft = first_output.map(|at| at.duration_since(started));
        eprintln!();
        eprintln!("TTFT: {ttft:?}");
        eprintln!("Elapsed: {elapsed:?}");
        eprintln!("Content bytes: {content_bytes}");
        eprintln!("Reasoning bytes: {reasoning_bytes}");
        eprintln!("Chunks: {chunks}");
    }

    Ok(())
}
