// SPDX-License-Identifier: Apache-2.0
//! ComposableKV D-SEG: locate PI chunk token spans in a tokenized prompt.
//!
//! The request names the `read` tool results that are PI chunks
//! (`nvext.ckv.files`, O-META). For each one the tool message content is
//! tokenized standalone (no special tokens — the same encoding the builder
//! used), hashed with the builder's `chunk_hash`, and searched for as a
//! contiguous subsequence of the prompt tokens. A chunk whose in-context
//! tokens differ from its standalone tokens is simply not found and is left
//! out (D-SEG-3).

use std::sync::OnceLock;

use dynamo_kv_router::composition::ChunkSpan;
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
};
use sha2::{Digest, Sha256};

use crate::protocols::common::extensions::CkvFile;

/// `ckv.chunk.chunk_hash`: sha256(model_id ‖ 0 ‖ len as u64 LE ‖ ids as u32 LE)[:32 hex].
pub fn chunk_hash(model_id: &str, token_ids: &[u32]) -> String {
    let mut h = Sha256::new();
    h.update(model_id.as_bytes());
    h.update([0u8]);
    h.update((token_ids.len() as u64).to_le_bytes());
    for t in token_ids {
        h.update(t.to_le_bytes());
    }
    let digest = h.finalize();
    let mut hex = String::with_capacity(32);
    for b in &digest[..16] {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

/// Model id used for chunk hashes: `CKV_MODEL_ID` (must match the workers'
/// `CKV_MODEL_ID`), else the request's model name.
pub fn model_id_for(request_model: &str) -> String {
    static ENV: OnceLock<Option<String>> = OnceLock::new();
    ENV.get_or_init(|| std::env::var("CKV_MODEL_ID").ok().filter(|v| !v.is_empty()))
        .clone()
        .unwrap_or_else(|| request_model.to_string())
}

/// First position `>= from` where `needle` occurs in `haystack`.
pub fn find_subsequence(haystack: &[u32], needle: &[u32], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut i = from;
    while i <= last_start {
        if haystack[i] == first && haystack[i..i + needle.len()] == *needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Text of the tool message answering `tool_call_id`, if it is plain text.
fn tool_message_text<'a>(
    messages: &'a [ChatCompletionRequestMessage],
    tool_call_id: &str,
) -> Option<&'a str> {
    messages.iter().find_map(|m| match m {
        ChatCompletionRequestMessage::Tool(t) if t.tool_call_id == tool_call_id => match &t.content
        {
            ChatCompletionRequestToolMessageContent::Text(s) => Some(s.as_str()),
            ChatCompletionRequestToolMessageContent::Array(_) => None,
        },
        _ => None,
    })
}

/// Locate every listed file's chunk in `token_ids`. Files are searched in the
/// given order, each from the end of the previous match (conversation order),
/// falling back to a search from the start. Returns spans sorted by start.
pub fn locate_chunks<E>(
    model_id: &str,
    files: &[CkvFile],
    messages: &[ChatCompletionRequestMessage],
    token_ids: &[u32],
    encode: E,
) -> Vec<ChunkSpan>
where
    E: Fn(&str) -> anyhow::Result<Vec<u32>>,
{
    let mut spans: Vec<ChunkSpan> = Vec::new();
    let mut cursor = 0usize;
    for file in files {
        let Some(text) = tool_message_text(messages, &file.tool_call_id) else {
            tracing::debug!(tool_call_id = %file.tool_call_id, path = %file.path, "ckv: no text tool message for file");
            continue;
        };
        let ids = match encode(text) {
            Ok(ids) => ids,
            Err(error) => {
                tracing::warn!(%error, path = %file.path, "ckv: failed to tokenize chunk text");
                continue;
            }
        };
        if ids.is_empty() {
            continue;
        }
        let found = find_subsequence(token_ids, &ids, cursor)
            .or_else(|| find_subsequence(token_ids, &ids, 0));
        match found {
            Some(start) => {
                let hash = chunk_hash(model_id, &ids);
                tracing::debug!(path = %file.path, chunk_hash = %hash, start, len = ids.len(), "ckv: chunk located");
                cursor = start + ids.len();
                spans.push(ChunkSpan {
                    chunk_hash: hash,
                    start: start as u32,
                    len: ids.len() as u32,
                });
            }
            None => {
                // D-SEG-3: in-context tokenization differs from standalone.
                tracing::info!(path = %file.path, len = ids.len(), "ckv: chunk tokens not found in prompt; excluded");
            }
        }
    }
    spans.sort_by_key(|s| s.start);
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionRequestToolMessage;

    /// Fake tokenizer: one token per byte, with a twist — a chunk starting
    /// with '!' tokenizes to a different sequence inside context (simulated by
    /// the test prompt), to exercise D-SEG-3.
    fn enc(s: &str) -> anyhow::Result<Vec<u32>> {
        Ok(s.bytes().map(|b| b as u32).collect())
    }

    fn tool(id: &str, text: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
            content: ChatCompletionRequestToolMessageContent::Text(text.into()),
            tool_call_id: id.into(),
        })
    }

    fn file(id: &str, path: &str) -> CkvFile {
        CkvFile {
            tool_call_id: id.into(),
            path: path.into(),
        }
    }

    #[test]
    fn chunk_hash_matches_python_vectors() {
        // ckv.chunk.chunk_hash("m", [1,2,3]) / ("Qwen/Qwen3-0.6B", range(40))
        assert_eq!(
            chunk_hash("m", &[1, 2, 3]),
            "a65cfcdb96932d56f7ccfcebcc9eb388"
        );
        let ids: Vec<u32> = (0..40).collect();
        assert_eq!(
            chunk_hash("Qwen/Qwen3-0.6B", &ids),
            "13966c65f60ff5e914bcc4c361930ff1"
        );
    }

    #[test]
    fn t6_1_spans_are_found_in_conversation_order() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let b = "<path>/r/b.py</path>\nBBBBBB";
        let prompt = format!("<sys>hello</sys><tool>{a}</tool><user>x</user><tool>{b}</tool>tail");
        let ids = enc(&prompt).unwrap();
        let msgs = vec![tool("c1", a), tool("c2", b)];
        let spans = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/b.py")],
            &msgs,
            &ids,
            enc,
        );
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start as usize, prompt.find(a).unwrap());
        assert_eq!(spans[0].len as usize, a.len());
        assert_eq!(spans[0].chunk_hash, chunk_hash("m", &enc(a).unwrap()));
        assert_eq!(spans[1].start as usize, prompt.find(b).unwrap());
        // same hash as a standalone build of the same text
        assert_eq!(spans[1].chunk_hash, chunk_hash("m", &enc(b).unwrap()));
    }

    #[test]
    fn t6_2_no_files_means_no_spans() {
        let ids = enc("anything").unwrap();
        assert!(locate_chunks("m", &[], &[], &ids, enc).is_empty());
        // listed but no matching tool message
        assert!(
            locate_chunks("m", &[file("zz", "/r/a.py")], &[tool("c1", "x")], &ids, enc).is_empty()
        );
    }

    #[test]
    fn t6_3_chunk_whose_context_tokens_differ_is_excluded() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let b = "<path>/r/b.py</path>\nBBBB";
        // the prompt contains a mangled copy of b (merged boundary token)
        let prompt = format!("<tool>{a}</tool><tool>{}</tool>", b.replace("\nB", "~B"));
        let ids = enc(&prompt).unwrap();
        let msgs = vec![tool("c1", a), tool("c2", b)];
        let spans = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/b.py")],
            &msgs,
            &ids,
            enc,
        );
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].chunk_hash, chunk_hash("m", &enc(a).unwrap()));
    }

    #[test]
    fn same_file_read_twice_yields_two_spans() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let prompt = format!("<tool>{a}</tool>mid<tool>{a}</tool>");
        let ids = enc(&prompt).unwrap();
        let msgs = vec![tool("c1", a), tool("c2", a)];
        let spans = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/a.py")],
            &msgs,
            &ids,
            enc,
        );
        assert_eq!(spans.len(), 2);
        assert!(spans[0].start < spans[1].start);
        assert_eq!(spans[0].chunk_hash, spans[1].chunk_hash);
    }
}
