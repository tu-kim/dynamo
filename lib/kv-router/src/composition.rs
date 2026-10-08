// SPDX-License-Identifier: Apache-2.0
//! ComposableKV composition planning (SPEC §6.3 D-PLAN, §7).
//!
//! A [`CompositionPlanner`] turns a request's PI chunk spans (found by the
//! frontend preprocessor, D-SEG) plus the router's prefix-overlap and PI
//! holdings into a `composition_plan` for the selected worker. Worker
//! selection itself stays with the existing selector; a planner that wants a
//! particular worker returns it from [`CompositionPlanner::pin_worker`], which
//! the router turns into a scheduler pin before the selector runs.
//!
//! The plan rules mirror `ckv/plan.py` (`make_plan`/`validate`) exactly: the
//! worker re-validates and re-cuts the plan against its real prefix hit.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::indexer::pi_index::PiIndex;
use crate::protocols::{ChunkMedium, WorkerId, WorkerWithDpRank};
use crate::scheduling::OverlapSignals;

/// EPIC link recompute length (SPEC Q5: k = 32).
pub const LINK_TOKENS: u32 = 32;
pub const PLAN_VERSION: u32 = 1;

/// A PI chunk located in the prompt: `start` is the position of the chunk's
/// first token, `len` the chunk's standalone token count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkSpan {
    pub chunk_hash: String,
    pub start: u32,
    pub len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SegmentKind {
    Prefix,
    Recompute,
    Pi,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSegment {
    #[serde(rename = "type")]
    pub kind: SegmentKind,
    pub start: u32,
    pub len: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_offset: Option<u32>,
}

impl PlanSegment {
    pub fn end(&self) -> u32 {
        self.start + self.len
    }
}

/// Router-side prediction of what the worker will reuse (D-OBS); the worker
/// logs it next to the actual outcome.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanCounts {
    pub prefix: u32,
    pub recompute: u32,
    pub pi: u32,
    pub pi_segments: u32,
    /// Chunks found in the prompt but planned as recompute (no holder).
    pub chunks_skipped: u32,
}

/// SPEC §7 plan: `{"version": 1, "segments": [...]}` plus `predicted`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionPlan {
    pub version: u32,
    pub segments: Vec<PlanSegment>,
    #[serde(default)]
    pub predicted: PlanCounts,
}

impl CompositionPlan {
    pub fn counts(&self) -> PlanCounts {
        let mut c = PlanCounts {
            chunks_skipped: self.predicted.chunks_skipped,
            ..Default::default()
        };
        for s in &self.segments {
            match s.kind {
                SegmentKind::Prefix => c.prefix += s.len,
                SegmentKind::Recompute => c.recompute += s.len,
                SegmentKind::Pi => {
                    c.pi += s.len;
                    c.pi_segments += 1;
                }
            }
        }
        c
    }

    pub fn pi_segments(&self) -> impl Iterator<Item = &PlanSegment> {
        self.segments.iter().filter(|s| s.kind == SegmentKind::Pi)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid composition plan: {0}")]
pub struct PlanError(pub String);

/// SPEC §7 rules: segments cover the prompt in order without gaps or overlap,
/// `prefix` only first, a `pi` segment follows its link recompute (or the
/// prefix) and is block-aligned.
pub fn validate(plan: &CompositionPlan, prompt_len: u32, block_size: u32) -> Result<(), PlanError> {
    let segs = &plan.segments;
    if segs.is_empty() {
        return Err(PlanError("empty plan".into()));
    }
    let mut pos = 0u32;
    for (i, s) in segs.iter().enumerate() {
        if s.start != pos {
            return Err(PlanError(format!(
                "segment {i}: starts at {}, expected {pos} (gap or overlap)",
                s.start
            )));
        }
        if s.len == 0 {
            return Err(PlanError(format!("segment {i}: empty")));
        }
        match s.kind {
            SegmentKind::Prefix if i != 0 => {
                return Err(PlanError(format!("segment {i}: prefix must be first")));
            }
            SegmentKind::Pi => {
                if i == 0 || segs[i - 1].kind == SegmentKind::Pi {
                    return Err(PlanError(format!(
                        "segment {i}: pi must follow a recompute (link) or prefix segment"
                    )));
                }
                if s.start % block_size != 0 || s.len % block_size != 0 {
                    return Err(PlanError(format!(
                        "segment {i}: pi segment [{}, {}) is not block-aligned",
                        s.start,
                        s.end()
                    )));
                }
                if s.chunk_hash.as_deref().is_none_or(str::is_empty) || s.chunk_offset.is_none() {
                    return Err(PlanError(format!(
                        "segment {i}: pi needs chunk_hash and chunk_offset"
                    )));
                }
            }
            _ => {}
        }
        pos = s.end();
    }
    if pos != prompt_len {
        return Err(PlanError(format!(
            "plan covers {pos} tokens, prompt has {prompt_len}"
        )));
    }
    Ok(())
}

/// The PI cut of a chunk at prompt position `start`: `(link_len k, pi_start, pi_end)`.
/// `k` is `link_tokens` rounded up so that `start + k` is block-aligned.
pub fn pi_cut(span: &ChunkSpan, block_size: u32, link_tokens: u32) -> (u32, u32, u32) {
    let k = link_tokens + (block_size - (span.start + link_tokens) % block_size) % block_size;
    let pi_start = span.start + k;
    let pi_end = (span.start + span.len) / block_size * block_size;
    (k, pi_start, pi_end)
}

/// Build a plan for a prompt with `prefix_len` cached tokens and PI chunks
/// at `chunks` (same rules as `ckv.plan.make_plan`). Chunks inside the
/// prefix or too short for one block become recompute.
pub fn make_plan(
    prompt_len: u32,
    prefix_len: u32,
    chunks: &[ChunkSpan],
    block_size: u32,
    link_tokens: u32,
) -> Result<CompositionPlan, PlanError> {
    if !prefix_len.is_multiple_of(block_size) && prefix_len < prompt_len {
        return Err(PlanError(format!(
            "prefix_len {prefix_len} must be block-aligned"
        )));
    }
    let mut segs: Vec<PlanSegment> = Vec::new();
    let mut pos = 0u32;
    if prefix_len > 0 {
        segs.push(PlanSegment {
            kind: SegmentKind::Prefix,
            start: 0,
            len: prefix_len,
            chunk_hash: None,
            chunk_offset: None,
        });
        pos = prefix_len;
    }
    fn recompute_to(segs: &mut Vec<PlanSegment>, pos: &mut u32, end: u32) {
        if end > *pos {
            match segs.last_mut() {
                Some(last) if last.kind == SegmentKind::Recompute => last.len += end - *pos,
                _ => segs.push(PlanSegment {
                    kind: SegmentKind::Recompute,
                    start: *pos,
                    len: end - *pos,
                    chunk_hash: None,
                    chunk_offset: None,
                }),
            }
            *pos = end;
        }
    }
    let mut sorted: Vec<&ChunkSpan> = chunks.iter().collect();
    sorted.sort_by_key(|c| c.start);
    let mut skipped = 0u32;
    for c in sorted {
        let (k, pi_start, pi_end) = pi_cut(c, block_size, link_tokens);
        if pi_start < pos || pi_end < pi_start + block_size {
            skipped += 1;
            continue;
        }
        recompute_to(&mut segs, &mut pos, pi_start);
        segs.push(PlanSegment {
            kind: SegmentKind::Pi,
            start: pi_start,
            len: pi_end - pi_start,
            chunk_hash: Some(c.chunk_hash.clone()),
            chunk_offset: Some(k),
        });
        pos = pi_end;
    }
    recompute_to(&mut segs, &mut pos, prompt_len);
    let mut plan = CompositionPlan {
        version: PLAN_VERSION,
        segments: segs,
        predicted: PlanCounts::default(),
    };
    validate(&plan, prompt_len, block_size)?;
    plan.predicted = plan.counts();
    plan.predicted.chunks_skipped = skipped;
    Ok(plan)
}

/// Everything a planner may look at (D-PLAN-1).
pub struct PlannerInput<'a> {
    pub prompt_len: u32,
    pub block_size: u32,
    pub link_tokens: u32,
    pub chunks: &'a [ChunkSpan],
    /// Per-worker prefix overlap (effective cached tokens / overlap blocks).
    pub overlap: &'a OverlapSignals,
    /// Per-worker PI holdings.
    pub pi: &'a PiIndex,
    /// Workers the scheduler may pick (None = all).
    pub candidate_workers: Option<&'a [WorkerId]>,
}

pub trait CompositionPlanner: Send + Sync {
    fn name(&self) -> &str;

    /// Called before scheduling. `Some(worker)` pins the request to that rank
    /// (an explicit pin in the request wins). The stub returns `None` so the
    /// existing cost function selects (D-PLAN-2).
    fn pin_worker(&self, _input: &PlannerInput<'_>) -> Option<WorkerWithDpRank> {
        None
    }

    /// Called after scheduling with the selected rank and its prefix hit in
    /// tokens. `None` means "send no plan" (baseline behaviour).
    fn plan(
        &self,
        input: &PlannerInput<'_>,
        worker: WorkerWithDpRank,
        cached_tokens: u32,
    ) -> Option<CompositionPlan>;
}

/// Which chunks the selected rank can splice, under the single-node
/// assumption (SPEC Q8): GPU entries must be on that rank with the exact
/// `(hash, link offset)` and PI length (the worker's PiPool key); DRAM
/// entries reported by any rank count for every rank and only need to cover
/// `k + pi_len` tokens (the worker slices the original).
pub fn usable_chunks(
    input: &PlannerInput<'_>,
    worker: WorkerWithDpRank,
) -> Vec<(ChunkSpan, ChunkMedium)> {
    let mut out = Vec::new();
    for c in input.chunks {
        let (k, pi_start, pi_end) = pi_cut(c, input.block_size, input.link_tokens);
        if pi_end < pi_start + input.block_size {
            continue;
        }
        let pi_len = pi_end - pi_start;
        let gpu = input.pi.holders(&c.chunk_hash, k).into_iter().any(|h| {
            h.medium == ChunkMedium::Gpu
                && h.worker_id == worker.worker_id
                && h.dp_rank == worker.dp_rank
                && h.num_tokens == pi_len
        });
        if gpu {
            out.push((c.clone(), ChunkMedium::Gpu));
            continue;
        }
        let dram = input
            .pi
            .holders(&c.chunk_hash, 0)
            .into_iter()
            .any(|h| h.medium == ChunkMedium::Dram && h.num_tokens >= k + pi_len);
        if dram {
            out.push((c.clone(), ChunkMedium::Dram));
        }
    }
    out
}

/// Block-aligned prefix hit for a rank, from the overlap signals.
pub fn prefix_len_for(
    input: &PlannerInput<'_>,
    worker: WorkerWithDpRank,
    cached_tokens: u32,
) -> u32 {
    let from_overlap = input
        .overlap
        .effective_cached_tokens
        .get(&worker)
        .map(|n| *n as u32)
        .unwrap_or(cached_tokens);
    let aligned = from_overlap / input.block_size * input.block_size;
    aligned.min(input.prompt_len / input.block_size * input.block_size)
}

/// D-PLAN-2 stub: worker selection untouched; plan = every PI chunk the
/// selected rank holds after its prefix hit.
#[derive(Debug, Default)]
pub struct StubPlanner;

pub const STUB_PLANNER: &str = "ckv-stub";
pub const PIN_WORKER_PLANNER: &str = "ckv-pin-worker";

impl CompositionPlanner for StubPlanner {
    fn name(&self) -> &str {
        STUB_PLANNER
    }

    fn plan(
        &self,
        input: &PlannerInput<'_>,
        worker: WorkerWithDpRank,
        cached_tokens: u32,
    ) -> Option<CompositionPlan> {
        let usable = usable_chunks(input, worker);
        if usable.is_empty() {
            return None;
        }
        let chunks: Vec<ChunkSpan> = usable.into_iter().map(|(c, _)| c).collect();
        let prefix_len = prefix_len_for(input, worker, cached_tokens);
        match make_plan(
            input.prompt_len,
            prefix_len,
            &chunks,
            input.block_size,
            input.link_tokens,
        ) {
            Ok(mut plan) => {
                plan.predicted.chunks_skipped += input.chunks.len() as u32 - chunks.len() as u32;
                if plan.predicted.pi_segments == 0 {
                    return None;
                }
                Some(plan)
            }
            Err(error) => {
                tracing::warn!(%error, "ckv planner produced an invalid plan; sending none");
                None
            }
        }
    }
}

/// Test planner (T6-6): always pins one worker, then plans like the stub.
#[derive(Debug)]
pub struct PinWorkerPlanner {
    pub worker_id: WorkerId,
    pub dp_rank: u32,
}

impl CompositionPlanner for PinWorkerPlanner {
    fn name(&self) -> &str {
        PIN_WORKER_PLANNER
    }

    fn pin_worker(&self, _input: &PlannerInput<'_>) -> Option<WorkerWithDpRank> {
        Some(WorkerWithDpRank {
            worker_id: self.worker_id,
            dp_rank: self.dp_rank,
        })
    }

    fn plan(
        &self,
        input: &PlannerInput<'_>,
        worker: WorkerWithDpRank,
        cached_tokens: u32,
    ) -> Option<CompositionPlan> {
        StubPlanner.plan(input, worker, cached_tokens)
    }
}

/// Planner factory: `parameters` is the text after the first `:` in the
/// configured name (`ckv-pin-worker:1234` → `"1234"`).
pub type PlannerFactory =
    Arc<dyn Fn(&str) -> Result<Arc<dyn CompositionPlanner>, String> + Send + Sync>;

#[derive(Default)]
pub struct CompositionPlannerRegistry {
    factories: HashMap<String, PlannerFactory>,
}

impl CompositionPlannerRegistry {
    pub fn register(&mut self, name: &str, factory: PlannerFactory) -> Result<(), String> {
        if self.factories.contains_key(name) {
            return Err(format!(
                "composition planner {name:?} is already registered"
            ));
        }
        self.factories.insert(name.to_string(), factory);
        Ok(())
    }

    /// `spec` is `name` or `name:parameters`.
    pub fn resolve(&self, spec: &str) -> Result<Arc<dyn CompositionPlanner>, String> {
        let (name, params) = spec.split_once(':').unwrap_or((spec, ""));
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| format!("unknown composition planner {name:?}"))?;
        factory(params)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.factories.keys().cloned().collect();
        v.sort();
        v
    }
}

static REGISTRY: OnceLock<Mutex<CompositionPlannerRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<CompositionPlannerRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(CompositionPlannerRegistry::default()))
}

/// Process-wide registration (plugins call this at startup).
pub fn register_planner(name: &str, factory: PlannerFactory) -> Result<(), String> {
    registry().lock().unwrap().register(name, factory)
}

/// Resolve a configured planner spec against the process registry.
pub fn resolve_planner(spec: &str) -> Result<Arc<dyn CompositionPlanner>, String> {
    registry().lock().unwrap().resolve(spec)
}

/// Register the planners this crate ships (idempotent).
pub fn register_builtin_planners() {
    let mut r = registry().lock().unwrap();
    let _ = r.register(
        STUB_PLANNER,
        Arc::new(|_| Ok(Arc::new(StubPlanner) as Arc<dyn CompositionPlanner>)),
    );
    let _ = r.register(
        PIN_WORKER_PLANNER,
        Arc::new(|params: &str| {
            let (w, d) = params.split_once('.').unwrap_or((params, "0"));
            let worker_id = w.trim().parse::<WorkerId>().map_err(|_| {
                format!("{PIN_WORKER_PLANNER}: expected <worker_id>[.<dp_rank>], got {params:?}")
            })?;
            let dp_rank = d
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("{PIN_WORKER_PLANNER}: bad dp_rank {d:?}"))?;
            Ok(Arc::new(PinWorkerPlanner { worker_id, dp_rank }) as Arc<dyn CompositionPlanner>)
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{ChunkEventData, ChunkEventKind};

    const B: u32 = 16;

    fn span(h: &str, start: u32, len: u32) -> ChunkSpan {
        ChunkSpan {
            chunk_hash: h.into(),
            start,
            len,
        }
    }

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank {
            worker_id: id,
            dp_rank: 0,
        }
    }

    fn stored(
        pi: &PiIndex,
        worker: WorkerWithDpRank,
        hash: &str,
        offset: u32,
        n: u32,
        medium: ChunkMedium,
    ) {
        pi.apply(
            worker,
            &ChunkEventData {
                kind: ChunkEventKind::Stored,
                chunk_hash: hash.into(),
                offset,
                num_tokens: n,
                medium,
            },
        );
    }

    /// Same input as ckv/tests (make_plan): prompt 5920, prefix 1024, chunk at 1024 len 4096.
    #[test]
    fn t6_5_make_plan_matches_python_rules() {
        let plan = make_plan(5920, 1024, &[span("h", 1024, 4096)], B, LINK_TOKENS).unwrap();
        let kinds: Vec<(SegmentKind, u32, u32)> = plan
            .segments
            .iter()
            .map(|s| (s.kind, s.start, s.len))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (SegmentKind::Prefix, 0, 1024),
                (SegmentKind::Recompute, 1024, 32),
                (SegmentKind::Pi, 1056, 4064),
                (SegmentKind::Recompute, 5120, 800),
            ]
        );
        assert_eq!(plan.segments[2].chunk_offset, Some(32));
        assert_eq!(plan.predicted.pi, 4064);
        // unaligned chunk start: k rounds up to the next block boundary
        let plan = make_plan(600, 0, &[span("h", 10, 500)], B, LINK_TOKENS).unwrap();
        let pi = plan.pi_segments().next().unwrap();
        assert_eq!((pi.start, pi.len, pi.chunk_offset), (48, 448, Some(38)));
        validate(&plan, 600, B).unwrap();
    }

    #[test]
    fn t6_5_validate_rejects_bad_plans() {
        let mut plan = make_plan(5920, 1024, &[span("h", 1024, 4096)], B, LINK_TOKENS).unwrap();
        plan.segments[1].len += 1;
        assert!(validate(&plan, 5920, B).is_err());
        let mut plan = make_plan(5920, 1024, &[span("h", 1024, 4096)], B, LINK_TOKENS).unwrap();
        plan.segments.swap(0, 1);
        assert!(validate(&plan, 5920, B).is_err());
        assert!(
            validate(
                &CompositionPlan {
                    version: 1,
                    segments: vec![],
                    predicted: Default::default()
                },
                0,
                B
            )
            .is_err()
        );
    }

    #[test]
    fn t6_4_stub_uses_only_chunks_the_selected_worker_holds() {
        let pi = PiIndex::new();
        // W2 holds chunk a on GPU (exact cut); nobody holds b; c is in DRAM via W3.
        stored(&pi, w(2), "a", 32, 464, ChunkMedium::Gpu); // pi cut of [0,500) is [32,496)
        stored(&pi, w(3), "c", 0, 500, ChunkMedium::Dram);
        let chunks = vec![span("a", 0, 500), span("b", 500, 500), span("c", 1000, 500)];
        let mut overlap = OverlapSignals::default();
        overlap.effective_cached_tokens.insert(w(1), 1008); // W1: big prefix hit (63 blocks)
        overlap.effective_cached_tokens.insert(w(2), 0);
        let input = PlannerInput {
            prompt_len: 1600,
            block_size: B,
            link_tokens: LINK_TOKENS,
            chunks: &chunks,
            overlap: &overlap,
            pi: &pi,
            candidate_workers: None,
        };
        let stub = StubPlanner;
        assert!(stub.pin_worker(&input).is_none()); // selection stays with the cost function

        // Cost function picked W2: plan has a (GPU) and c (DRAM, node-shared), not b.
        let plan = stub.plan(&input, w(2), 0).unwrap();
        let pis: Vec<(&str, u32)> = plan
            .pi_segments()
            .map(|s| (s.chunk_hash.as_deref().unwrap(), s.start))
            .collect();
        assert_eq!(pis, vec![("a", 32), ("c", 1040)]);
        assert_eq!(plan.predicted.chunks_skipped, 1);
        validate(&plan, 1600, B).unwrap();

        // Cost function picked W1: its prefix hit covers a and b; only c (DRAM) is spliced.
        let plan = stub.plan(&input, w(1), 1008).unwrap();
        assert_eq!(
            plan.segments[0],
            PlanSegment {
                kind: SegmentKind::Prefix,
                start: 0,
                len: 1008,
                chunk_hash: None,
                chunk_offset: None
            }
        );
        let pis: Vec<&str> = plan
            .pi_segments()
            .map(|s| s.chunk_hash.as_deref().unwrap())
            .collect();
        assert_eq!(pis, vec!["c"]);

        // W9 holds nothing on GPU; c is DRAM so it still gets one PI segment (single node).
        let plan = stub.plan(&input, w(9), 0).unwrap();
        assert_eq!(plan.pi_segments().count(), 1);
    }

    #[test]
    fn t6_4_gpu_entry_with_other_cut_is_not_used() {
        let pi = PiIndex::new();
        stored(&pi, w(2), "a", 32, 432, ChunkMedium::Gpu); // different PI length
        let chunks = vec![span("a", 0, 500)];
        let overlap = OverlapSignals::default();
        let input = PlannerInput {
            prompt_len: 500,
            block_size: B,
            link_tokens: LINK_TOKENS,
            chunks: &chunks,
            overlap: &overlap,
            pi: &pi,
            candidate_workers: None,
        };
        assert!(StubPlanner.plan(&input, w(2), 0).is_none());
    }

    #[test]
    fn t6_6_registry_resolves_custom_planner() {
        register_builtin_planners();
        let p = resolve_planner("ckv-pin-worker:42.1").unwrap();
        let pi = PiIndex::new();
        let overlap = OverlapSignals::default();
        let input = PlannerInput {
            prompt_len: 100,
            block_size: B,
            link_tokens: LINK_TOKENS,
            chunks: &[],
            overlap: &overlap,
            pi: &pi,
            candidate_workers: None,
        };
        assert_eq!(
            p.pin_worker(&input),
            Some(WorkerWithDpRank {
                worker_id: 42,
                dp_rank: 1
            })
        );
        assert_eq!(resolve_planner("ckv-stub").unwrap().name(), STUB_PLANNER);
        assert!(resolve_planner("nope").is_err());
        assert!(resolve_planner("ckv-pin-worker:x").is_err());
        register_planner(
            "test-always-7",
            Arc::new(|_| {
                Ok(Arc::new(PinWorkerPlanner {
                    worker_id: 7,
                    dp_rank: 0,
                }) as Arc<dyn CompositionPlanner>)
            }),
        )
        .unwrap();
        assert_eq!(
            resolve_planner("test-always-7")
                .unwrap()
                .pin_worker(&input)
                .unwrap()
                .worker_id,
            7
        );
    }

    #[test]
    fn plan_json_matches_spec_shape() {
        let plan = make_plan(5920, 1024, &[span("h", 1024, 4096)], B, LINK_TOKENS).unwrap();
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["version"], 1);
        assert_eq!(v["segments"][2]["type"], "pi");
        assert_eq!(v["segments"][2]["chunk_offset"], 32);
        assert!(v["segments"][0].get("chunk_hash").is_none());
        assert_eq!(v["predicted"]["pi"], 4064);
    }
}
