//! Unified reasoning/thinking parser framework.
//!
//! Supports multiple reasoning formats:
//! - **Think tags**: `<think>...</think>` (DeepSeek R1, QwQ, SmolLM3)
//! - **Channel tags**: `<|channel>thought\n...<channel|>` (Gemma 4)
 //! - **OpenAI Harmony**: GPT-OSS multi-channel output

pub mod harmony;
pub mod tag_based;

pub use harmony::{HarmonyContext, HarmonyToolCall};
pub use tag_based::TagReasoningContext;

/// Trait for reasoning content parsers.
///
/// All reasoning parsers extract two streams from model output:
/// - **content**: The user-visible response
/// - **reasoning**: Internal chain-of-thought (hidden from user)
pub trait ReasoningParser: Send + Sync {
    /// Process incremental bytes (for text-based parsers like tag_based).
    fn process_bytes(&mut self, bytes: &[u8]);

    /// Process one model token. Token-native parsers such as OpenAI Harmony can
    /// use the original token id instead of re-tokenizing decoded bytes.
    fn process_token(&mut self, token_id: u32, bytes: &[u8]) {
        self.process_bytes(bytes);
    }
    /// Finalize at end of stream (flush buffers, handle unclosed blocks).
    fn finalize(&mut self);
    /// Get new content since last call (for streaming).
    fn get_content_delta(&mut self) -> Option<String>;
    /// Get new reasoning since last call (for streaming).
    fn get_reasoning_delta(&mut self) -> Option<String>;
    /// Get all accumulated content.
    fn content(&self) -> Option<String>;
    /// Get all accumulated reasoning.
    fn reasoning_content(&self) -> Option<String>;
}

/// The active reasoning format for a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningMode {
    /// Tag-based reasoning (think tags or Gemma 4 channel tags)
    TagBased,
    /// OpenAI Harmony multi-channel reasoning used by GPT-OSS.
    Harmony,
}

/// Check if a template uses any reasoning format (think tags or channel tags).
pub fn is_reasoning_template(template: &str) -> bool {
    harmony::is_harmony_template(template)
        || tag_based::is_think_tag_template(template)
        || tag_based::is_channel_tag_template(template)
}
