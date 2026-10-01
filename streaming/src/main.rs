use futures_util::StreamExt;
use mistralrs_streaming::{ChatRequest, Message, StreamEvent, StreamingClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:1234/v1".into());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "default".into());
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Hello from mistral.rs streaming.".to_owned()
    } else {
        prompt
    };

    let client = StreamingClient::builder().base_url(base_url).build()?;
    let request = ChatRequest {
        model,
        messages: vec![Message {
            role: "user".into(),
            content: prompt,
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
                for choice in chunk.choices {
                    if let Some(reasoning) = choice.delta.reasoning {
                        print!("{reasoning}");
                    }
                    if let Some(content) = choice.delta.content {
                        print!("{content}");
                    }
                }
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            StreamEvent::Done => break,
        }
    }

    eprintln!();
    eprintln!("TTFT: {:?}", stream.stats().ttft());
    eprintln!("Elapsed: {:?}", stream.stats().elapsed());
    Ok(())
}
