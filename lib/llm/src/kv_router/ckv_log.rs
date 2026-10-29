// SPDX-License-Identifier: Apache-2.0
//! ComposableKV router-side request log (SPEC D-OBS, F7).
//!
//! When `CKV_ROUTER_LOG` names a file, the router appends one JSON object per
//! line: a `route` record when a worker is selected (with the composition
//! plan summary, if any) and a `finish` record when the request completes
//! (frontend-side TTFT and end-to-end time). `ckv-aggregate` joins them with
//! the workers' records by request id. Off by default; never fails a request.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use std::collections::HashMap;

use dynamo_kv_router::composition::{CompositionPlan, CreditWeights, PiCredit};
use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::ckv_segment::CkvFileOutcome;

static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();

fn sink() -> Option<&'static Mutex<File>> {
    SINK.get_or_init(|| {
        let path = std::env::var("CKV_ROUTER_LOG").ok().filter(|p| !p.is_empty())?;
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => {
                tracing::info!(path = %path, "ComposableKV router log enabled");
                Some(Mutex::new(f))
            }
            Err(error) => {
                tracing::warn!(path = %path, %error, "CKV_ROUTER_LOG: cannot open; router log disabled");
                None
            }
        }
    })
    .as_ref()
}

pub fn enabled() -> bool {
    sink().is_some()
}

/// What a `route` record is built from. Plain data so the record shape
/// ([`route_record`]) is testable without a router.
pub struct RouteInfo<'a> {
    pub rid: &'a str,
    pub worker: WorkerWithDpRank,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub overlap_blocks: f64,
    /// Chunk spans the preprocessor found (planner input), not files listed.
    pub chunks_found: u64,
    pub planner: Option<&'a str>,
    /// Link recompute length `k` this plan was made with (D-OBS-2).
    pub link_tokens: u32,
    /// Per-file location outcomes, in `nvext.ckv.files` order.
    pub files: Option<&'a [CkvFileOutcome]>,
    pub plan: Option<&'a CompositionPlan>,
    /// F10 (D-ROUTE-5): the planner's per-candidate composable match, the
    /// weights it was folded in with, how the worker was chosen, and the
    /// session's previous worker (from the request host's affinity target).
    pub credits: Option<&'a HashMap<WorkerWithDpRank, PiCredit>>,
    pub credit_weights: Option<CreditWeights>,
    pub chosen_by: &'a str,
    pub prev_worker: Option<WorkerWithDpRank>,
    pub session_id: Option<&'a str>,
}

/// The `route` record (D-OBS, D-OBS-2): `plan` is the predicted summary,
/// `segments` the full plan (`null` without one), `files` the outcome array
/// (`[]` when the request listed none). `type`/`ts` are added by [`write`].
pub fn route_record(info: &RouteInfo<'_>) -> serde_json::Map<String, serde_json::Value> {
    use serde_json::Value;
    let mut rec = serde_json::Map::new();
    rec.insert("rid".into(), info.rid.into());
    rec.insert("worker_id".into(), info.worker.worker_id.into());
    rec.insert("dp_rank".into(), info.worker.dp_rank.into());
    rec.insert("prompt_tokens".into(), info.prompt_tokens.into());
    rec.insert("cached_tokens".into(), info.cached_tokens.into());
    rec.insert("overlap_blocks".into(), info.overlap_blocks.into());
    rec.insert("chunks_found".into(), info.chunks_found.into());
    rec.insert("planner".into(), info.planner.into());
    rec.insert("link_tokens".into(), info.link_tokens.into());
    rec.insert(
        "plan".into(),
        info.plan
            .and_then(|p| serde_json::to_value(&p.predicted).ok())
            .unwrap_or(Value::Null),
    );
    rec.insert(
        "segments".into(),
        info.plan
            .and_then(|p| serde_json::to_value(&p.segments).ok())
            .unwrap_or(Value::Null),
    );
    rec.insert(
        "files".into(),
        info.files
            .and_then(|f| serde_json::to_value(f).ok())
            .unwrap_or_else(|| Value::Array(Vec::new())),
    );
    // D-ROUTE-5: candidates sorted by rank; `credit_blocks` is what the
    // selector added (`w_gpu * pi_gpu + w_dram * pi_dram`).
    let weights = info.credit_weights.unwrap_or(CreditWeights {
        gpu: 0.0,
        gpu_rotate: 0.0,
        dram: 0.0,
    });
    let candidates: Vec<Value> = info
        .credits
        .map(|c| {
            let mut v: Vec<(&WorkerWithDpRank, &PiCredit)> = c.iter().collect();
            v.sort_by_key(|(w, _)| (w.worker_id, w.dp_rank));
            v.into_iter()
                .map(|(w, c)| {
                    serde_json::json!({
                        "worker_id": w.worker_id,
                        "dp_rank": w.dp_rank,
                        "prefix_blocks": c.prefix_blocks,
                        "pi_gpu_blocks": c.pi_gpu_blocks,
                        "pi_gpu_rotate_blocks": c.pi_gpu_rotate_blocks,
                        "pi_dram_blocks": c.pi_dram_blocks,
                        "credit_blocks": c.weighted(weights),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    rec.insert("candidates".into(), Value::Array(candidates));
    rec.insert(
        "credit_weights".into(),
        info.credit_weights
            .map(|w| serde_json::json!({"gpu": w.gpu, "gpu_rotate": w.gpu_rotate, "dram": w.dram}))
            .unwrap_or(Value::Null),
    );
    rec.insert("chosen_by".into(), info.chosen_by.into());
    rec.insert(
        "prev_worker".into(),
        info.prev_worker
            .map(|w| w.worker_id.into())
            .unwrap_or(Value::Null),
    );
    rec.insert(
        "moved".into(),
        info.prev_worker
            .map(|p| Value::Bool(p.worker_id != info.worker.worker_id))
            .unwrap_or(Value::Null),
    );
    rec.insert("session_id".into(), info.session_id.into());
    rec
}

/// Append one record (a JSON object; `type` and `ts` are added here).
pub fn write(kind: &str, mut record: serde_json::Map<String, serde_json::Value>) {
    let Some(file) = sink() else { return };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    record.insert("type".into(), kind.into());
    record.insert("ts".into(), ts.into());
    let mut line = serde_json::Value::Object(record).to_string();
    line.push('\n');
    if let Ok(mut f) = file.lock() {
        let _ = f.write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckv_segment::CkvFileStatus;
    use dynamo_kv_router::composition::{ChunkSpan, LINK_TOKENS, make_plan};

    fn outcome(id: &str, status: CkvFileStatus, hash: Option<&str>) -> CkvFileOutcome {
        // start/len only exist for a located file; the hash and full_len also
        // for not_found (the text tokenized, it just was not in the prompt).
        let located = status == CkvFileStatus::Located;
        CkvFileOutcome {
            tool_call_id: id.into(),
            path: format!("/r/{id}.py"),
            status,
            chunk_hash: hash.map(String::from),
            start: located.then_some(1024),
            len: located.then_some(4096),
            full_len: hash.map(|_| 4096),
        }
    }

    /// D-OBS-2: the route record carries k, the per-file outcomes and the
    /// plan's full segment list next to the existing summary fields.
    #[test]
    fn route_record_has_files_segments_and_link_tokens() {
        let chunk = ChunkSpan {
            chunk_hash: "h".into(),
            start: 1024,
            len: 4096,
            path: "/r/c1.py".into(),
            tool_call_id: "c1".into(),
        };
        let plan = make_plan(5920, 1024, &[chunk], 16, LINK_TOKENS).unwrap();
        let files = vec![
            outcome("c1", CkvFileStatus::Located, Some("h")),
            outcome("c2", CkvFileStatus::NotFound, Some("h2")),
            outcome("c3", CkvFileStatus::NoText, None),
        ];
        let rec = route_record(&RouteInfo {
            rid: "r1",
            worker: WorkerWithDpRank {
                worker_id: 7,
                dp_rank: 1,
            },
            prompt_tokens: 5920,
            cached_tokens: 1024,
            overlap_blocks: 64.0,
            chunks_found: 1,
            planner: Some("ckv-stub"),
            link_tokens: 32,
            files: Some(&files),
            plan: Some(&plan),
            credits: None,
            credit_weights: None,
            chosen_by: "argmin",
            prev_worker: None,
            session_id: None,
        });
        let v = serde_json::Value::Object(rec);
        assert_eq!(v["rid"], "r1");
        assert_eq!(v["worker_id"], 7);
        assert_eq!(v["dp_rank"], 1);
        assert_eq!(v["prompt_tokens"], 5920);
        assert_eq!(v["cached_tokens"], 1024);
        assert_eq!(v["overlap_blocks"], 64.0);
        assert_eq!(v["chunks_found"], 1);
        assert_eq!(v["planner"], "ckv-stub");
        assert_eq!(v["link_tokens"], 32);
        // existing summary is unchanged
        assert_eq!(v["plan"]["pi"], 4064);
        assert_eq!(v["plan"]["pi_segments"], 1);
        // full segments: type,start,len[,chunk_hash,chunk_offset]
        let segs = v["segments"].as_array().unwrap();
        assert_eq!(segs.len(), 4);
        assert_eq!(segs[0]["type"], "prefix");
        assert_eq!(segs[2]["type"], "pi");
        assert_eq!(segs[2]["start"], 1056);
        assert_eq!(segs[2]["len"], 4064);
        assert_eq!(segs[2]["chunk_hash"], "h");
        assert_eq!(segs[2]["chunk_offset"], 32);
        assert!(segs[0].get("chunk_hash").is_none());
        // files: one per listed file, status names as in the spec
        let files = v["files"].as_array().unwrap();
        assert_eq!(files.len(), 3);
        assert_eq!(files[0]["tool_call_id"], "c1");
        assert_eq!(files[0]["path"], "/r/c1.py");
        assert_eq!(files[0]["status"], "located");
        assert_eq!(files[0]["chunk_hash"], "h");
        assert_eq!(files[0]["start"], 1024);
        assert_eq!(files[0]["len"], 4096);
        assert_eq!(files[0]["full_len"], 4096);
        assert_eq!(files[1]["status"], "not_found");
        assert_eq!(files[1]["chunk_hash"], "h2");
        assert!(files[1]["start"].is_null());
        assert_eq!(files[2]["status"], "no_text");
        assert!(files[2]["chunk_hash"].is_null());
        // the pi hash in segments is the located file's hash (trace derivation)
        assert_eq!(segs[2]["chunk_hash"], files[0]["chunk_hash"]);
    }

    /// No plan (baseline / no holder): `segments` is null, `plan` is null,
    /// and no listed files gives `files: []`, never a missing key.
    #[test]
    fn route_record_without_plan_or_files() {
        let rec = route_record(&RouteInfo {
            rid: "r2",
            worker: WorkerWithDpRank {
                worker_id: 1,
                dp_rank: 0,
            },
            prompt_tokens: 100,
            cached_tokens: 0,
            overlap_blocks: 0.0,
            chunks_found: 0,
            planner: None,
            link_tokens: 64,
            files: None,
            plan: None,
            credits: None,
            credit_weights: None,
            chosen_by: "argmin",
            prev_worker: None,
            session_id: None,
        });
        let v = serde_json::Value::Object(rec);
        assert_eq!(v["segments"], serde_json::Value::Null);
        assert_eq!(v["plan"], serde_json::Value::Null);
        assert_eq!(v["planner"], serde_json::Value::Null);
        assert_eq!(v["files"], serde_json::json!([]));
        assert_eq!(v["link_tokens"], 64);
        assert_eq!(v["chunks_found"], 0);
        // files listed but none located still yields the outcomes
        let files = vec![outcome("c9", CkvFileStatus::TokenizeError, None)];
        let rec = route_record(&RouteInfo {
            rid: "r3",
            worker: WorkerWithDpRank {
                worker_id: 1,
                dp_rank: 0,
            },
            prompt_tokens: 100,
            cached_tokens: 0,
            overlap_blocks: 0.0,
            chunks_found: 0,
            planner: Some("ckv-stub"),
            link_tokens: 32,
            files: Some(&files),
            plan: None,
            credits: None,
            credit_weights: None,
            chosen_by: "argmin",
            prev_worker: None,
            session_id: None,
        });
        let v = serde_json::Value::Object(rec);
        assert_eq!(v["files"][0]["status"], "tokenize_error");
        assert_eq!(v["segments"], serde_json::Value::Null);
    }

    #[test]
    fn writes_json_lines_when_enabled() {
        // The sink is process-global; this test only checks the record shape via a
        // fresh file when the env var is set for the test binary.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("router.jsonl");
        // SAFETY: tests in this module run in one process; set before first use.
        unsafe { std::env::set_var("CKV_ROUTER_LOG", &path) };
        let mut rec = serde_json::Map::new();
        rec.insert("rid".into(), "r1".into());
        rec.insert("worker_id".into(), 7u64.into());
        write("route", rec);
        if !enabled() {
            return; // another test initialised the sink first; nothing to check here
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(v["type"], "route");
        assert_eq!(v["rid"], "r1");
        assert_eq!(v["worker_id"], 7);
        assert!(v["ts"].as_f64().unwrap() > 0.0);
    }

    /// D-ROUTE-5: candidates (sorted by rank, weighted credit), how the worker
    /// was chosen, the session's previous worker and whether it moved.
    #[test]
    fn t10_3_route_record_has_candidates_and_move() {
        let w = |id: u64| WorkerWithDpRank {
            worker_id: id,
            dp_rank: 0,
        };
        let mut credits = HashMap::new();
        credits.insert(
            w(2),
            PiCredit {
                prefix_blocks: 0,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 28,
            },
        );
        credits.insert(
            w(1),
            PiCredit {
                prefix_blocks: 4,
                pi_gpu_blocks: 29,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 28,
            },
        );
        let rec = route_record(&RouteInfo {
            rid: "r2",
            worker: w(1),
            prompt_tokens: 1600,
            cached_tokens: 64,
            overlap_blocks: 4.0,
            chunks_found: 2,
            planner: Some("ckv-aware"),
            link_tokens: 32,
            files: None,
            plan: None,
            credits: Some(&credits),
            credit_weights: Some(CreditWeights {
                gpu: 1.0,
                gpu_rotate: 0.9,
                dram: 0.5,
            }),
            chosen_by: "argmin",
            prev_worker: Some(w(2)),
            session_id: Some("s1"),
        });
        let v = serde_json::Value::Object(rec);
        let c = v["candidates"].as_array().unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["worker_id"], 1);
        assert_eq!(c[0]["pi_gpu_blocks"], 29);
        assert_eq!(c[0]["credit_blocks"], 29.0 + 14.0);
        assert_eq!(c[1]["worker_id"], 2);
        assert_eq!(c[1]["credit_blocks"], 14.0);
        assert_eq!(v["credit_weights"]["dram"], 0.5);
        assert_eq!(v["chosen_by"], "argmin");
        assert_eq!(v["prev_worker"], 2);
        assert_eq!(v["moved"], true);
        assert_eq!(v["session_id"], "s1");
        // without a planner: empty candidates, nulls
        let rec = route_record(&RouteInfo {
            rid: "r3",
            worker: w(1),
            prompt_tokens: 16,
            cached_tokens: 0,
            overlap_blocks: 0.0,
            chunks_found: 0,
            planner: None,
            link_tokens: 32,
            files: None,
            plan: None,
            credits: None,
            credit_weights: None,
            chosen_by: "argmin",
            prev_worker: None,
            session_id: None,
        });
        let v = serde_json::Value::Object(rec);
        assert_eq!(v["candidates"].as_array().unwrap().len(), 0);
        assert!(
            v["credit_weights"].is_null() && v["prev_worker"].is_null() && v["moved"].is_null()
        );
    }
}
