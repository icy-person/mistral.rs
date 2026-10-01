use std::hint::black_box;

use mistralrs_streaming::ChatRequest;

fn main() {
    let request = ChatRequest {
        model: "benchmark".into(),
        messages: Vec::new(),
        stream: true,
        max_tokens: Some(128),
        temperature: None,
        top_p: None,
    };
    black_box(serde_json::to_vec(&request).expect("serialize"));
}
