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
use serde::{Deserialize, Serialize};
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

/// Where a chunk's standalone tokens sit in the prompt, allowing for BPE
/// merges at both boundaries: the first and last standalone tokens may fuse
/// with the surrounding template text (e.g. Qwen3 merges the closing `>` with
/// the newline before `</tool_response>`). The interior must match exactly.
///
/// Returns `(start, usable_len)`: `start` is the position of the chunk's first
/// token (possibly a merged token; it lies inside the link recompute, so its
/// KV is never taken from the chunk), and `usable_len` excludes the last
/// token when it did not match, so no PI block covers it.
pub fn locate_span(token_ids: &[u32], chunk: &[u32], from: usize) -> Option<(usize, usize)> {
    if chunk.len() < 3 {
        return find_subsequence(token_ids, chunk, from).map(|p| (p, chunk.len()));
    }
    if let Some(p) = find_subsequence(token_ids, chunk, from) {
        return Some((p, chunk.len()));
    }
    let interior = &chunk[1..chunk.len() - 1];
    let p = find_subsequence(token_ids, interior, from.saturating_add(1))?;
    let start = p - 1;
    let last_pos = p + interior.len();
    let last_matches = token_ids.get(last_pos) == Some(&chunk[chunk.len() - 1]);
    Some((
        start,
        if last_matches {
            chunk.len()
        } else {
            chunk.len() - 1
        },
    ))
}

/// Why a listed file did or did not become a [`ChunkSpan`] (D-OBS-2).
/// Serialized as `located | no_text | tokenize_error | empty | not_found`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CkvFileStatus {
    Located,
    /// No plain-text tool message answers `tool_call_id`.
    NoText,
    TokenizeError,
    /// The tool message text tokenizes to nothing.
    Empty,
    /// D-SEG-3: standalone tokens do not occur in the prompt.
    NotFound,
}

/// Per-file result of [`locate_chunks`], one per `nvext.ckv.files` entry in
/// request order. Logged in the router's `route` record so a session's
/// `tool_use.callID` joins to the chunk the router planned with. `len` is
/// the usable length (see [`locate_span`]), `full_len` the standalone token
/// count the hash is over. Absent fields serialize as `null`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CkvFileOutcome {
    pub tool_call_id: String,
    pub path: String,
    pub status: CkvFileStatus,
    #[serde(default)]
    pub chunk_hash: Option<String>,
    #[serde(default)]
    pub start: Option<u32>,
    #[serde(default)]
    pub len: Option<u32>,
    #[serde(default)]
    pub full_len: Option<u32>,
}

/// Output of [`locate_chunks`]: `spans` sorted by start (the planner input),
/// `files` in request order (one per listed file, for the trace).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CkvLocated {
    pub spans: Vec<ChunkSpan>,
    pub files: Vec<CkvFileOutcome>,
}

/// Locate every listed file's chunk in `token_ids`. Files are searched in the
/// given order, each from the end of the previous match (conversation order),
/// falling back to a search from the start. Returns spans sorted by start.
/// `ChunkSpan::len` is the usable length (see [`locate_span`]); the hash is
/// always over the full standalone token sequence, as the builder computes it.
pub fn locate_chunks<E>(
    model_id: &str,
    files: &[CkvFile],
    messages: &[ChatCompletionRequestMessage],
    token_ids: &[u32],
    encode: E,
) -> CkvLocated
where
    E: Fn(&str) -> anyhow::Result<Vec<u32>>,
{
    let mut spans: Vec<ChunkSpan> = Vec::new();
    let mut outcomes: Vec<CkvFileOutcome> = Vec::with_capacity(files.len());
    let mut cursor = 0usize;
    for file in files {
        let mut outcome = CkvFileOutcome {
            tool_call_id: file.tool_call_id.clone(),
            path: file.path.clone(),
            status: CkvFileStatus::NotFound,
            chunk_hash: None,
            start: None,
            len: None,
            full_len: None,
        };
        let Some(text) = tool_message_text(messages, &file.tool_call_id) else {
            tracing::debug!(tool_call_id = %file.tool_call_id, path = %file.path, "ckv: no text tool message for file");
            outcome.status = CkvFileStatus::NoText;
            outcomes.push(outcome);
            continue;
        };
        let ids = match encode(text) {
            Ok(ids) => ids,
            Err(error) => {
                tracing::warn!(%error, path = %file.path, "ckv: failed to tokenize chunk text");
                outcome.status = CkvFileStatus::TokenizeError;
                outcomes.push(outcome);
                continue;
            }
        };
        if ids.is_empty() {
            outcome.status = CkvFileStatus::Empty;
            outcome.full_len = Some(0);
            outcomes.push(outcome);
            continue;
        }
        // The hash is known whether or not the chunk is found, so a not_found
        // file can still be matched against the worker's chunk store.
        let hash = chunk_hash(model_id, &ids);
        outcome.chunk_hash = Some(hash.clone());
        outcome.full_len = Some(ids.len() as u32);
        let found =
            locate_span(token_ids, &ids, cursor).or_else(|| locate_span(token_ids, &ids, 0));
        match found {
            Some((start, usable)) => {
                tracing::debug!(path = %file.path, chunk_hash = %hash, start, len = ids.len(), usable,
                    "ckv: chunk located");
                cursor = start + usable;
                outcome.status = CkvFileStatus::Located;
                outcome.start = Some(start as u32);
                outcome.len = Some(usable as u32);
                spans.push(ChunkSpan {
                    chunk_hash: hash,
                    start: start as u32,
                    len: usable as u32,
                    path: file.path.clone(),
                    tool_call_id: file.tool_call_id.clone(),
                });
            }
            None => {
                // D-SEG-3: in-context tokenization differs from standalone.
                tracing::info!(path = %file.path, len = ids.len(), "ckv: chunk tokens not found in prompt; excluded");
            }
        }
        outcomes.push(outcome);
    }
    spans.sort_by_key(|s| s.start);
    CkvLocated {
        spans,
        files: outcomes,
    }
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
        let located = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/b.py")],
            &msgs,
            &ids,
            enc,
        );
        let spans = &located.spans;
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start as usize, prompt.find(a).unwrap());
        assert_eq!(spans[0].len as usize, a.len());
        assert_eq!(spans[0].chunk_hash, chunk_hash("m", &enc(a).unwrap()));
        assert_eq!(spans[1].start as usize, prompt.find(b).unwrap());
        // same hash as a standalone build of the same text
        assert_eq!(spans[1].chunk_hash, chunk_hash("m", &enc(b).unwrap()));
        // spans carry the file identity for the trace join
        assert_eq!(
            (spans[0].path.as_str(), spans[0].tool_call_id.as_str()),
            ("/r/a.py", "c1")
        );
        assert_eq!(
            (spans[1].path.as_str(), spans[1].tool_call_id.as_str()),
            ("/r/b.py", "c2")
        );
        // one outcome per listed file, in request order
        assert_eq!(
            located.files,
            vec![
                CkvFileOutcome {
                    tool_call_id: "c1".into(),
                    path: "/r/a.py".into(),
                    status: CkvFileStatus::Located,
                    chunk_hash: Some(spans[0].chunk_hash.clone()),
                    start: Some(spans[0].start),
                    len: Some(a.len() as u32),
                    full_len: Some(a.len() as u32),
                },
                CkvFileOutcome {
                    tool_call_id: "c2".into(),
                    path: "/r/b.py".into(),
                    status: CkvFileStatus::Located,
                    chunk_hash: Some(spans[1].chunk_hash.clone()),
                    start: Some(spans[1].start),
                    len: Some(b.len() as u32),
                    full_len: Some(b.len() as u32),
                },
            ]
        );
    }

    #[test]
    fn t6_2_no_files_means_no_spans() {
        let ids = enc("anything").unwrap();
        let located = locate_chunks("m", &[], &[], &ids, enc);
        assert!(located.spans.is_empty());
        assert!(located.files.is_empty());
        // listed but no matching tool message: no span, outcome no_text
        let located = locate_chunks("m", &[file("zz", "/r/a.py")], &[tool("c1", "x")], &ids, enc);
        assert!(located.spans.is_empty());
        assert_eq!(
            located.files,
            vec![CkvFileOutcome {
                tool_call_id: "zz".into(),
                path: "/r/a.py".into(),
                status: CkvFileStatus::NoText,
                chunk_hash: None,
                start: None,
                len: None,
                full_len: None,
            }]
        );
        // a tool message with array content is no_text too
        let arr = ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
            content: ChatCompletionRequestToolMessageContent::Array(vec![]),
            tool_call_id: "c1".into(),
        });
        let located = locate_chunks("m", &[file("c1", "/r/a.py")], &[arr], &ids, enc);
        assert_eq!(located.files[0].status, CkvFileStatus::NoText);
    }

    #[test]
    fn t6_3_chunk_whose_context_tokens_differ_is_excluded() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let b = "<path>/r/b.py</path>\nBBBB";
        // the prompt contains a mangled copy of b (an interior token differs)
        let prompt = format!(
            "<tool>{a}</tool><tool>{}</tool>",
            b.replace("</path>", "</pth>")
        );
        let ids = enc(&prompt).unwrap();
        let msgs = vec![tool("c1", a), tool("c2", b)];
        let located = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/b.py")],
            &msgs,
            &ids,
            enc,
        );
        let spans = &located.spans;
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].chunk_hash, chunk_hash("m", &enc(a).unwrap()));
        // the excluded file is reported as not_found with its standalone hash
        assert_eq!(located.files.len(), 2);
        assert_eq!(located.files[0].status, CkvFileStatus::Located);
        assert_eq!(
            located.files[1],
            CkvFileOutcome {
                tool_call_id: "c2".into(),
                path: "/r/b.py".into(),
                status: CkvFileStatus::NotFound,
                chunk_hash: Some(chunk_hash("m", &enc(b).unwrap())),
                start: None,
                len: None,
                full_len: Some(b.len() as u32),
            }
        );
    }

    /// Qwen3 merges the chunk's closing `>` with the template newline: the
    /// interior still matches, the span excludes the fused last token, and the
    /// hash stays the standalone one (what the builder stored).
    #[test]
    fn t6_3_boundary_merge_keeps_chunk_without_its_last_token() {
        let a = "<path>/r/a.py</path>\nAAAA</content>";
        let mut ids = enc(a).unwrap();
        let standalone = ids.clone();
        let prompt_prefix = enc("<tool>\n").unwrap();
        let last = ids.pop().unwrap();
        let mut prompt = prompt_prefix.clone();
        prompt.extend(&ids);
        prompt.push(last + 1000); // fused ">\n" token: different id
        prompt.extend(enc("</tool>").unwrap());
        let msgs = vec![tool("c1", a)];
        let located = locate_chunks("m", &[file("c1", "/r/a.py")], &msgs, &prompt, enc);
        let spans = &located.spans;
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start as usize, prompt_prefix.len());
        assert_eq!(spans[0].len as usize, standalone.len() - 1);
        assert_eq!(spans[0].chunk_hash, chunk_hash("m", &standalone));
        // outcome: usable len excludes the fused token, full_len does not
        assert_eq!(located.files[0].len, Some(standalone.len() as u32 - 1));
        assert_eq!(located.files[0].full_len, Some(standalone.len() as u32));
        // a fused *first* token is tolerated too (it lies inside the link recompute)
        let mut prompt2 = prompt_prefix.clone();
        prompt2.push(standalone[0] + 1000);
        prompt2.extend(&standalone[1..]);
        let spans = locate_chunks("m", &[file("c1", "/r/a.py")], &msgs, &prompt2, enc).spans;
        assert_eq!(
            (spans[0].start as usize, spans[0].len as usize),
            (prompt_prefix.len(), standalone.len())
        );
    }

    #[test]
    fn same_file_read_twice_yields_two_spans() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let prompt = format!("<tool>{a}</tool>mid<tool>{a}</tool>");
        let ids = enc(&prompt).unwrap();
        let msgs = vec![tool("c1", a), tool("c2", a)];
        let located = locate_chunks(
            "m",
            &[file("c1", "/r/a.py"), file("c2", "/r/a.py")],
            &msgs,
            &ids,
            enc,
        );
        let spans = &located.spans;
        assert_eq!(spans.len(), 2);
        assert!(spans[0].start < spans[1].start);
        assert_eq!(spans[0].chunk_hash, spans[1].chunk_hash);
        assert_eq!(spans[0].tool_call_id, "c1");
        assert_eq!(spans[1].tool_call_id, "c2");
        // D-OBS-2: one outcome per listed file, in request order, even for the same path.
        let files = &located.files;
        assert_eq!(files.len(), 2);
        assert_eq!(
            files
                .iter()
                .map(|f| (f.tool_call_id.as_str(), f.status))
                .collect::<Vec<_>>(),
            vec![
                ("c1", CkvFileStatus::Located),
                ("c2", CkvFileStatus::Located)
            ]
        );
        assert_eq!(files[0].chunk_hash, files[1].chunk_hash);
        assert!(files[0].start < files[1].start);
    }

    /// D-OBS-2: every listed file gets exactly one outcome, in request order,
    /// whatever happened to it; the serialized status names are the spec's.
    #[test]
    fn t6_1_outcomes_cover_every_status() {
        let a = "<path>/r/a.py</path>\nAAAA";
        let prompt = format!("<tool>{a}</tool>");
        let ids = enc(&prompt).unwrap();
        let msgs = vec![
            tool("ok", a),
            tool("empty", ""),
            tool("bad", "!boom"),
            tool("missing", "ZZZZ not in prompt"),
        ];
        let files = [
            file("missing", "/r/missing.py"),
            file("ok", "/r/a.py"),
            file("notext", "/r/notext.py"),
            file("empty", "/r/empty.py"),
            file("bad", "/r/bad.py"),
        ];
        // tokenizer that fails on text starting with '!'
        let enc2 = |s: &str| -> anyhow::Result<Vec<u32>> {
            if s.starts_with('!') {
                anyhow::bail!("unsupported text");
            }
            enc(s)
        };
        let located = locate_chunks("m", &files, &msgs, &ids, enc2);
        assert_eq!(located.spans.len(), 1);
        let statuses: Vec<(&str, CkvFileStatus)> = located
            .files
            .iter()
            .map(|f| (f.tool_call_id.as_str(), f.status))
            .collect();
        assert_eq!(
            statuses,
            vec![
                ("missing", CkvFileStatus::NotFound),
                ("ok", CkvFileStatus::Located),
                ("notext", CkvFileStatus::NoText),
                ("empty", CkvFileStatus::Empty),
                ("bad", CkvFileStatus::TokenizeError),
            ]
        );
        assert_eq!(located.files[3].full_len, Some(0));
        assert!(located.files[4].chunk_hash.is_none());
        // serialized shape: snake_case status, absent fields as null
        let v = serde_json::to_value(&located.files).unwrap();
        let names: Vec<&str> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["status"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec!["not_found", "located", "no_text", "empty", "tokenize_error"]
        );
        assert!(v[0]["start"].is_null());
        assert_eq!(
            v[0]["chunk_hash"],
            chunk_hash("m", &enc("ZZZZ not in prompt").unwrap())
        );
        assert_eq!(v[1]["path"], "/r/a.py");
        assert_eq!(v[1]["len"], a.len() as u64);
        assert_eq!(v[1]["full_len"], a.len() as u64);
        assert_eq!(v[2]["chunk_hash"], serde_json::Value::Null);
        let back: Vec<CkvFileOutcome> = serde_json::from_value(v).unwrap();
        assert_eq!(back, located.files);
    }
}
