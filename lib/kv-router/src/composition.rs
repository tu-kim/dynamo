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

/// Link recompute length `k` for this process: `CKV_LINK_TOKENS` (frontend env,
/// e.g. 64 or 128 to trade PI tokens for accuracy) or [`LINK_TOKENS`]. Read once.
pub fn link_tokens() -> u32 {
    static K: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("CKV_LINK_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|k| *k > 0)
            .unwrap_or(LINK_TOKENS)
    })
}
pub const PLAN_VERSION: u32 = 1;

/// D-ROUTE-1 default credit weights: PI already in the rank's GPU pool counts
/// like its device prefix hit; PI it must load from DRAM a little less (the
/// load is cheaper than recompute but not free).
pub const DEFAULT_PI_GPU_WEIGHT: f64 = 1.0;
pub const DEFAULT_PI_DRAM_WEIGHT: f64 = 0.8;
/// F12 (D-ROUTE-6): a whole-chunk GPU entry at position 0 (prefetch) the worker
/// re-rotates on use: cheaper than a DRAM load, dearer than an exact hit.
pub const DEFAULT_PI_GPU_ROTATE_WEIGHT: f64 = 0.9;

/// Credit weights per holding class (D-ROUTE-1/6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CreditWeights {
    pub gpu: f64,
    pub gpu_rotate: f64,
    pub dram: f64,
}

impl Default for CreditWeights {
    fn default() -> Self {
        Self {
            gpu: DEFAULT_PI_GPU_WEIGHT,
            gpu_rotate: DEFAULT_PI_GPU_ROTATE_WEIGHT,
            dram: DEFAULT_PI_DRAM_WEIGHT,
        }
    }
}

/// Where a candidate rank would take a chunk's PI from (router view).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Holding {
    /// Exact PiPool key on that rank: `(hash, span start)` with the PI length.
    Gpu,
    /// Whole chunk on that rank at position 0 (F12 prefetch): re-rotated on use.
    GpuRotate,
    /// Node DRAM store: loaded (and rotated) on use.
    Dram,
}

/// A PI chunk located in the prompt: `start` is the position of the chunk's
/// first token, `len` the chunk's standalone token count. `path` and
/// `tool_call_id` name the `read` result it came from (D-OBS-2 trace join).
/// Frontend-only: the worker gets [`PlanSegment`]s, never spans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkSpan {
    pub chunk_hash: String,
    pub start: u32,
    pub len: u32,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub tool_call_id: String,
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
    /// Every rank the router knows (all workers x their dp ranks), so a cold
    /// rank with neither an overlap signal nor a PI entry is still scored and
    /// recorded (D-ROUTE-4). None (unit tests): overlap keys ∪ PI index ranks.
    pub ranks: Option<&'a [WorkerWithDpRank]>,
}

/// Per-candidate composable match (F10, D-ROUTE): the blocks a rank would
/// not prefill. `prefix_blocks` from the overlap signals, `pi_gpu_blocks`
/// from PI entries already in that rank's GPU pool (exact cut), and
/// `pi_dram_blocks` from entries it would load from the node DRAM store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiCredit {
    pub prefix_blocks: u32,
    pub pi_gpu_blocks: u32,
    #[serde(default)]
    pub pi_gpu_rotate_blocks: u32,
    pub pi_dram_blocks: u32,
}

impl PiCredit {
    pub fn weighted(&self, w: CreditWeights) -> f64 {
        w.gpu * self.pi_gpu_blocks as f64
            + w.gpu_rotate * self.pi_gpu_rotate_blocks as f64
            + w.dram * self.pi_dram_blocks as f64
    }
}

pub trait CompositionPlanner: Send + Sync {
    fn name(&self) -> &str;

    /// Called before scheduling: the composable match of every candidate
    /// rank, folded into the selector's cost as credit blocks (D-ROUTE-1).
    /// Empty (the default and the stub) means no PI credit: selection by
    /// prefix overlap and load only.
    fn worker_credits(&self, _input: &PlannerInput<'_>) -> HashMap<WorkerWithDpRank, PiCredit> {
        HashMap::new()
    }

    /// Weights applied to `worker_credits` before scheduling.
    fn credit_weights(&self) -> CreditWeights {
        CreditWeights::default()
    }

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
/// PiPool key and PI length. The worker keys a GPU entry by the **absolute
/// prompt position of the chunk's token 0** (`PiKey(hash, pos - chunk_offset)`
/// = the span's `start`; the KV is rotated to that position at load), so a
/// GPU entry is reusable only when the chunk sits at the same position again
/// (the next turns of the same session). DRAM entries (offset 0, the
/// unrotated original) reported by any rank count for every rank and only
/// need to cover `k + pi_len` tokens (the worker slices and rotates).
pub fn usable_chunks(
    input: &PlannerInput<'_>,
    worker: WorkerWithDpRank,
) -> Vec<(ChunkSpan, Holding)> {
    let mut out = Vec::new();
    for c in input.chunks {
        let (k, pi_start, pi_end) = pi_cut(c, input.block_size, input.link_tokens);
        if pi_end < pi_start + input.block_size {
            continue;
        }
        let pi_len = pi_end - pi_start;
        let on_rank = |h: &crate::indexer::pi_index::PiHolding| {
            h.worker_id == worker.worker_id && h.dp_rank == worker.dp_rank
        };
        let gpu = input
            .pi
            .holders(&c.chunk_hash, c.start)
            .iter()
            .any(|h| h.medium == ChunkMedium::Gpu && on_rank(h) && h.num_tokens == pi_len);
        if gpu {
            out.push((c.clone(), Holding::Gpu));
            continue;
        }
        // position-0 holders: a GPU whole-chunk (prefetch, F12) on this rank, or the DRAM original
        let at_zero = input.pi.holders(&c.chunk_hash, 0);
        if at_zero
            .iter()
            .any(|h| h.medium == ChunkMedium::Gpu && on_rank(h) && h.num_tokens >= k + pi_len)
        {
            out.push((c.clone(), Holding::GpuRotate));
            continue;
        }
        if at_zero
            .iter()
            .any(|h| h.medium == ChunkMedium::Dram && h.num_tokens >= k + pi_len)
        {
            out.push((c.clone(), Holding::Dram));
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

/// The candidate ranks a planner scores: `candidate_workers` if given, else
/// every rank with an overlap signal or a PI entry.
pub fn candidate_ranks(input: &PlannerInput<'_>) -> Vec<WorkerWithDpRank> {
    let all: Vec<WorkerWithDpRank> = match input.ranks {
        Some(ranks) => ranks.to_vec(),
        None => input
            .overlap
            .effective_cached_tokens
            .keys()
            .copied()
            .chain(input.pi.workers())
            .collect(),
    };
    let mut v: Vec<WorkerWithDpRank> = match input.candidate_workers {
        Some(ids) => {
            let ids: std::collections::HashSet<WorkerId> = ids.iter().copied().collect();
            all.into_iter()
                .filter(|w| ids.contains(&w.worker_id))
                .collect()
        }
        None => all,
    };
    v.sort_by_key(|w| (w.worker_id, w.dp_rank));
    v.dedup();
    v
}

/// The composable match of one rank (D-ROUTE-2): the plan the stub would send
/// if this rank were selected, split by where its PI segments come from. The
/// prefix hit is the overlap prediction (`cached_tokens` is unknown before
/// scheduling), so `pi_gpu + pi_dram` equals the sent plan's `predicted.pi`
/// whenever the scheduler reports the same hit.
pub fn credit_for(input: &PlannerInput<'_>, worker: WorkerWithDpRank) -> PiCredit {
    let b = input.block_size.max(1);
    let prefix_len = prefix_len_for(input, worker, 0);
    let usable = usable_chunks(input, worker);
    let holding_of: HashMap<&str, Holding> = usable
        .iter()
        .map(|(c, m)| (c.chunk_hash.as_str(), *m))
        .collect();
    let chunks: Vec<ChunkSpan> = usable.iter().map(|(c, _)| c.clone()).collect();
    let (mut gpu, mut rot, mut dram) = (0u32, 0u32, 0u32);
    if let Ok(plan) = make_plan(input.prompt_len, prefix_len, &chunks, b, input.link_tokens) {
        for s in plan.pi_segments() {
            let m = s
                .chunk_hash
                .as_deref()
                .and_then(|h| holding_of.get(h))
                .copied()
                .unwrap_or(Holding::Dram);
            match m {
                Holding::Gpu => gpu += s.len,
                Holding::GpuRotate => rot += s.len,
                Holding::Dram => dram += s.len,
            }
        }
    }
    PiCredit {
        prefix_blocks: prefix_len / b,
        pi_gpu_blocks: gpu / b,
        pi_gpu_rotate_blocks: rot / b,
        pi_dram_blocks: dram / b,
    }
}

/// The per-rank credit the selector adds to its overlap credit
/// (`TierOverlapBlocks::ckv_pi_credit_blocks`): `w_gpu * pi_gpu + w_dram * pi_dram`,
/// ranks with no PI left out.
pub fn weighted_credit_blocks(
    credits: &HashMap<WorkerWithDpRank, PiCredit>,
    weights: CreditWeights,
) -> rustc_hash::FxHashMap<WorkerWithDpRank, f64> {
    credits
        .iter()
        .filter(|(_, c)| c.pi_gpu_blocks > 0 || c.pi_gpu_rotate_blocks > 0 || c.pi_dram_blocks > 0)
        .map(|(w, c)| (*w, c.weighted(weights)))
        .collect()
}

/// D-PLAN-2 stub: worker selection untouched; plan = every PI chunk the
/// selected rank holds after its prefix hit.
#[derive(Debug, Default)]
pub struct StubPlanner;

/// F10 `ckv-aware`: selection credits every candidate's composable match
/// (D-ROUTE-1); the plan is the stub's. Parameters `gpu=<f64>,dram=<f64>`
/// override the router's credit weights for this planner.
#[derive(Debug, Clone, Default)]
pub struct AwarePlanner {
    pub gpu_weight: Option<f64>,
    pub gpu_rotate_weight: Option<f64>,
    pub dram_weight: Option<f64>,
}

pub const AWARE_PLANNER: &str = "ckv-aware";

impl AwarePlanner {
    pub fn parse(params: &str) -> Result<Self, String> {
        let mut p = Self::default();
        for kv in params.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("{AWARE_PLANNER}: expected key=value, got {kv:?}"))?;
            let v: f64 = v
                .trim()
                .parse()
                .map_err(|_| format!("{AWARE_PLANNER}: bad number for {k:?}: {v:?}"))?;
            if !(0.0..=10.0).contains(&v) {
                return Err(format!("{AWARE_PLANNER}: {k}={v} out of range [0, 10]"));
            }
            match k.trim() {
                "gpu" => p.gpu_weight = Some(v),
                "gpu_rotate" => p.gpu_rotate_weight = Some(v),
                "dram" => p.dram_weight = Some(v),
                other => return Err(format!("{AWARE_PLANNER}: unknown parameter {other:?}")),
            }
        }
        Ok(p)
    }
}

impl CompositionPlanner for AwarePlanner {
    fn name(&self) -> &str {
        AWARE_PLANNER
    }

    fn worker_credits(&self, input: &PlannerInput<'_>) -> HashMap<WorkerWithDpRank, PiCredit> {
        candidate_ranks(input)
            .into_iter()
            .map(|w| (w, credit_for(input, w)))
            .collect()
    }

    fn credit_weights(&self) -> CreditWeights {
        CreditWeights {
            gpu: self.gpu_weight.unwrap_or(DEFAULT_PI_GPU_WEIGHT),
            gpu_rotate: self
                .gpu_rotate_weight
                .unwrap_or(DEFAULT_PI_GPU_ROTATE_WEIGHT),
            dram: self.dram_weight.unwrap_or(DEFAULT_PI_DRAM_WEIGHT),
        }
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
        AWARE_PLANNER,
        Arc::new(|params: &str| {
            Ok(Arc::new(AwarePlanner::parse(params)?) as Arc<dyn CompositionPlanner>)
        }),
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
            path: String::new(),
            tool_call_id: String::new(),
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
        stored(&pi, w(2), "a", 0, 464, ChunkMedium::Gpu); // chunk at prompt position 0; pi cut of [0,500) is [32,496)
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
            ranks: None,
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
        stored(&pi, w(2), "a", 16, 464, ChunkMedium::Gpu); // right length, chunk at another position
        stored(&pi, w(2), "a", 0, 432, ChunkMedium::Gpu); // same position, different PI length
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
            ranks: None,
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
            ranks: None,
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

    /// T10-1: per-rank composable match. W1 holds chunk a on GPU at the exact
    /// cut, W2 holds it at another cut (-> DRAM), W3 nothing; c is DRAM for all;
    /// b has no holder; W1's prefix hit covers chunk a entirely (no PI credit
    /// for it there, only prefix blocks).
    #[test]
    fn t10_1_worker_credits_per_rank() {
        let pi = PiIndex::new();
        stored(&pi, w(1), "a", 0, 464, ChunkMedium::Gpu); // at the span's position 0, pi cut [32,496)
        stored(&pi, w(2), "a", 1024, 464, ChunkMedium::Gpu); // same chunk at another prompt position: not reusable on GPU
        stored(&pi, w(9), "a", 0, 500, ChunkMedium::Dram); // DRAM original, node-shared
        stored(&pi, w(9), "c", 0, 500, ChunkMedium::Dram);
        let chunks = vec![span("a", 0, 500), span("b", 500, 500), span("c", 1000, 500)];
        let mut overlap = OverlapSignals::default();
        overlap.effective_cached_tokens.insert(w(1), 0);
        overlap.effective_cached_tokens.insert(w(2), 0);
        overlap.effective_cached_tokens.insert(w(3), 1008); // 63 blocks: covers a and most of b
        let input = PlannerInput {
            prompt_len: 1600,
            block_size: B,
            link_tokens: LINK_TOKENS,
            chunks: &chunks,
            overlap: &overlap,
            pi: &pi,
            candidate_workers: None,
            ranks: None,
        };
        let aware = AwarePlanner::default();
        let credits = aware.worker_credits(&input);
        let mut ranks: Vec<u64> = credits.keys().map(|w| w.worker_id).collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 2, 3, 9]); // overlap keys ∪ PI index ranks
        // a: pi [32,496) = 464 tokens = 29 blocks; c: pi [1040,1488) = 448 = 28 blocks
        assert_eq!(
            credits[&w(1)],
            PiCredit {
                prefix_blocks: 0,
                pi_gpu_blocks: 29,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 28
            }
        );
        assert_eq!(
            credits[&w(2)],
            PiCredit {
                prefix_blocks: 0,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 57
            }
        );
        assert_eq!(
            credits[&w(3)],
            PiCredit {
                prefix_blocks: 63,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 28
            }
        );
        assert_eq!(
            credits[&w(9)],
            PiCredit {
                prefix_blocks: 0,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 57
            }
        );
        // D-ROUTE-2: the credited PI equals the plan the stub sends to that rank.
        for r in [1u64, 2, 3, 9] {
            let plan = aware.plan(&input, w(r), 0).unwrap();
            let c = &credits[&w(r)];
            assert_eq!(
                plan.predicted.pi / B,
                c.pi_gpu_blocks + c.pi_gpu_rotate_blocks + c.pi_dram_blocks,
                "rank {r}"
            );
        }
        // the stub credits nothing
        assert!(StubPlanner.worker_credits(&input).is_empty());
        // explicit ranks (the router's worker list): a cold rank is scored too (D-ROUTE-4)
        let ranks = [w(1), w(4)];
        let with_ranks = PlannerInput {
            ranks: Some(&ranks),
            ..input
        };
        let credits = aware.worker_credits(&with_ranks);
        let mut got: Vec<u64> = credits.keys().map(|w| w.worker_id).collect();
        got.sort();
        assert_eq!(got, vec![1, 4]);
        assert_eq!(
            credits[&w(4)],
            PiCredit {
                prefix_blocks: 0,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 57
            }
        );
        // candidate_workers narrows the set
        let narrowed = PlannerInput {
            candidate_workers: Some(&[2, 3]),
            ..input
        };
        let mut ranks: Vec<u64> = aware
            .worker_credits(&narrowed)
            .keys()
            .map(|w| w.worker_id)
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![2, 3]);
    }

    #[test]
    fn t10_1_aware_planner_params_and_registry() {
        register_builtin_planners();
        let p = resolve_planner("ckv-aware").unwrap();
        assert_eq!(p.name(), AWARE_PLANNER);
        let p = AwarePlanner::parse("gpu=1.0, dram=0.5").unwrap();
        assert_eq!((p.gpu_weight, p.dram_weight), (Some(1.0), Some(0.5)));
        assert!(AwarePlanner::parse("x=1").is_err());
        assert!(AwarePlanner::parse("gpu=-1").is_err());
        assert!(resolve_planner("ckv-aware:dram=0.7").is_ok());
        assert_eq!(
            resolve_planner("ckv-aware:dram=0.7,gpu_rotate=0.5")
                .unwrap()
                .credit_weights(),
            CreditWeights {
                gpu: 1.0,
                gpu_rotate: 0.5,
                dram: 0.7
            }
        );
        assert_eq!(StubPlanner.credit_weights(), CreditWeights::default());
        // weighted blocks: gpu 1.0, dram 0.5; ranks without PI are left out
        let mut credits = HashMap::new();
        credits.insert(
            w(1),
            PiCredit {
                prefix_blocks: 3,
                pi_gpu_blocks: 29,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 28,
            },
        );
        credits.insert(
            w(2),
            PiCredit {
                prefix_blocks: 63,
                pi_gpu_blocks: 0,
                pi_gpu_rotate_blocks: 0,
                pi_dram_blocks: 0,
            },
        );
        let wb = weighted_credit_blocks(
            &credits,
            CreditWeights {
                gpu: 1.0,
                gpu_rotate: 0.9,
                dram: 0.5,
            },
        );
        assert_eq!(wb.len(), 1);
        assert_eq!(wb[&w(1)], 29.0 + 14.0);
    }

    /// T12-4 (D-ROUTE-6): a whole-chunk GPU entry at position 0 on the rank counts
    /// as GpuRotate (any cut), an exact-position entry as Gpu, otherwise DRAM; a
    /// short position-0 GPU entry does not qualify.
    #[test]
    fn t12_4_prefetched_whole_chunk_credits_gpu_rotate() {
        let pi = PiIndex::new();
        stored(&pi, w(1), "a", 0, 500, ChunkMedium::Gpu); // prefetched whole chunk on rank 1
        stored(&pi, w(2), "a", 0, 300, ChunkMedium::Gpu); // too short for k + pi_len (32 + 464)
        stored(&pi, w(9), "a", 0, 500, ChunkMedium::Dram);
        stored(&pi, w(3), "a", 1024, 464, ChunkMedium::Gpu); // exact key for a span starting at 1024
        let overlap = OverlapSignals::default();
        let chunks = vec![span("a", 1024, 500)];
        let input = PlannerInput {
            prompt_len: 1600,
            block_size: B,
            link_tokens: LINK_TOKENS,
            chunks: &chunks,
            overlap: &overlap,
            pi: &pi,
            candidate_workers: None,
            ranks: None,
        };
        let h = |r: u64| {
            usable_chunks(&input, w(r))
                .into_iter()
                .map(|(_, m)| m)
                .collect::<Vec<_>>()
        };
        assert_eq!(h(1), vec![Holding::GpuRotate]);
        assert_eq!(h(2), vec![Holding::Dram]);
        assert_eq!(h(3), vec![Holding::Gpu]);
        assert_eq!(h(9), vec![Holding::Dram]);
        let credits = AwarePlanner::default().worker_credits(&input);
        assert_eq!(credits[&w(1)].pi_gpu_rotate_blocks, 29);
        assert_eq!(credits[&w(3)].pi_gpu_blocks, 29);
        assert_eq!(credits[&w(2)].pi_dram_blocks, 29);
        let wb = weighted_credit_blocks(&credits, CreditWeights::default());
        assert!(wb[&w(3)] > wb[&w(1)] && wb[&w(1)] > wb[&w(2)]);
        assert_eq!(wb[&w(1)], 0.9 * 29.0);
    }
}
