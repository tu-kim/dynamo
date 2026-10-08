// SPDX-License-Identifier: Apache-2.0
//! ComposableKV PI index (D-IDX): which position-independent KV chunks each
//! worker rank holds, on the GPU PI pool or in the node DRAM store. Updated
//! from `KvCacheEventData::Chunk` events and reset together with the prefix
//! index when a rank is cleared or leaves.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::protocols::{
    ChunkEventData, ChunkEventKind, ChunkMedium, DpRank, WorkerId, WorkerWithDpRank,
};

/// Identity of a PI entry: the chunk plus the absolute position it was rotated to
/// (0 for the DRAM original).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PiKey {
    pub chunk_hash: String,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiHolding {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub medium: ChunkMedium,
    pub num_tokens: u32,
}

#[derive(Debug, Default)]
pub struct PiIndex {
    /// (worker, medium) -> entries. GPU and DRAM are separate sets since a
    /// `ChunksCleared` only clears one medium.
    inner: Mutex<HashMap<(WorkerWithDpRank, ChunkMedium), HashMap<PiKey, u32>>>,
}

impl PiIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&self, worker: WorkerWithDpRank, data: &ChunkEventData) {
        let mut inner = self.inner.lock().unwrap();
        let slot = (worker, data.medium);
        match data.kind {
            ChunkEventKind::Stored => {
                inner.entry(slot).or_default().insert(
                    PiKey {
                        chunk_hash: data.chunk_hash.clone(),
                        offset: data.offset,
                    },
                    data.num_tokens,
                );
            }
            ChunkEventKind::Removed => {
                if let Some(entries) = inner.get_mut(&slot) {
                    entries.remove(&PiKey {
                        chunk_hash: data.chunk_hash.clone(),
                        offset: data.offset,
                    });
                    if entries.is_empty() {
                        inner.remove(&slot);
                    }
                }
            }
            ChunkEventKind::Cleared => {
                inner.remove(&slot);
            }
        }
    }

    pub fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        let mut inner = self.inner.lock().unwrap();
        inner.retain(|(w, _), _| !(w.worker_id == worker_id && w.dp_rank == dp_rank));
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        let mut inner = self.inner.lock().unwrap();
        inner.retain(|(w, _), _| w.worker_id != worker_id);
    }

    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    /// Every rank/medium holding `(chunk_hash, offset)`.
    pub fn holders(&self, chunk_hash: &str, offset: u32) -> Vec<PiHolding> {
        let key = PiKey {
            chunk_hash: chunk_hash.to_string(),
            offset,
        };
        let inner = self.inner.lock().unwrap();
        let mut out: Vec<PiHolding> = inner
            .iter()
            .filter_map(|((w, medium), entries)| {
                entries.get(&key).map(|n| PiHolding {
                    worker_id: w.worker_id,
                    dp_rank: w.dp_rank,
                    medium: *medium,
                    num_tokens: *n,
                })
            })
            .collect();
        out.sort_by_key(|h| (h.worker_id, h.dp_rank, h.medium as u8));
        out
    }

    /// Chunks held by one rank on one medium (any offset), for planning.
    pub fn entries(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        medium: ChunkMedium,
    ) -> Vec<(PiKey, u32)> {
        let inner = self.inner.lock().unwrap();
        inner
            .get(&(WorkerWithDpRank { worker_id, dp_rank }, medium))
            .map(|e| {
                let mut v: Vec<(PiKey, u32)> = e.iter().map(|(k, n)| (k.clone(), *n)).collect();
                v.sort_by(|a, b| (&a.0.chunk_hash, a.0.offset).cmp(&(&b.0.chunk_hash, b.0.offset)));
                v
            })
            .unwrap_or_default()
    }

    /// The whole index as `Stored` events (one per entry), e.g. for a
    /// recovery dump or a debug endpoint.
    pub fn dump(&self) -> Vec<(WorkerWithDpRank, ChunkEventData)> {
        let inner = self.inner.lock().unwrap();
        let mut out: Vec<(WorkerWithDpRank, ChunkEventData)> = inner
            .iter()
            .flat_map(|((w, medium), entries)| {
                let (w, medium) = (*w, *medium);
                entries.iter().map(move |(k, n)| {
                    (
                        w,
                        ChunkEventData {
                            kind: ChunkEventKind::Stored,
                            chunk_hash: k.chunk_hash.clone(),
                            offset: k.offset,
                            num_tokens: *n,
                            medium,
                        },
                    )
                })
            })
            .collect();
        out.sort_by(|a, b| {
            (
                a.0.worker_id,
                a.0.dp_rank,
                a.1.medium as u8,
                &a.1.chunk_hash,
                a.1.offset,
            )
                .cmp(&(
                    b.0.worker_id,
                    b.0.dp_rank,
                    b.1.medium as u8,
                    &b.1.chunk_hash,
                    b.1.offset,
                ))
        });
        out
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().values().map(|e| e.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(id: u64, dp: u32) -> WorkerWithDpRank {
        WorkerWithDpRank {
            worker_id: id,
            dp_rank: dp,
        }
    }

    fn ev(kind: ChunkEventKind, h: &str, offset: u32, medium: ChunkMedium) -> ChunkEventData {
        ChunkEventData {
            kind,
            chunk_hash: h.to_string(),
            offset,
            num_tokens: 1024,
            medium,
        }
    }

    #[test]
    fn t5_3_stored_then_removed_follows_event_order() {
        let idx = PiIndex::new();
        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Stored, "a", 32, ChunkMedium::Gpu),
        );
        idx.apply(
            w(2, 0),
            &ev(ChunkEventKind::Stored, "a", 32, ChunkMedium::Gpu),
        );
        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Stored, "a", 0, ChunkMedium::Dram),
        );
        let holders = idx.holders("a", 32);
        assert_eq!(holders.len(), 2);
        assert_eq!(holders[0].worker_id, 1);
        assert_eq!(holders[0].medium, ChunkMedium::Gpu);
        assert_eq!(idx.holders("a", 0)[0].medium, ChunkMedium::Dram);

        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Removed, "a", 32, ChunkMedium::Gpu),
        );
        let holders = idx.holders("a", 32);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].worker_id, 2);
        // Removing the GPU copy leaves the DRAM original.
        assert_eq!(idx.holders("a", 0).len(), 1);
        assert_eq!(idx.len(), 2);
        // Removing something unknown is a no-op.
        idx.apply(
            w(9, 0),
            &ev(ChunkEventKind::Removed, "zzz", 0, ChunkMedium::Gpu),
        );
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn t5_4_cleared_and_worker_removal_only_touch_that_rank() {
        let idx = PiIndex::new();
        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Stored, "a", 32, ChunkMedium::Gpu),
        );
        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Stored, "a", 0, ChunkMedium::Dram),
        );
        idx.apply(
            w(1, 1),
            &ev(ChunkEventKind::Stored, "b", 32, ChunkMedium::Gpu),
        );
        idx.apply(
            w(2, 0),
            &ev(ChunkEventKind::Stored, "a", 32, ChunkMedium::Gpu),
        );

        // ChunksCleared(GPU) from rank (1,0): its GPU entries go, DRAM and other ranks stay.
        idx.apply(
            w(1, 0),
            &ev(ChunkEventKind::Cleared, "", 0, ChunkMedium::Gpu),
        );
        assert_eq!(
            idx.holders("a", 32)
                .iter()
                .map(|h| h.worker_id)
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(idx.holders("a", 0).len(), 1);
        assert_eq!(idx.holders("b", 32).len(), 1);

        // Rank (1,1) leaves.
        idx.remove_worker_dp_rank(1, 1);
        assert!(idx.holders("b", 32).is_empty());
        assert_eq!(idx.holders("a", 0).len(), 1);

        // Worker 1 leaves entirely.
        idx.remove_worker(1);
        assert!(idx.holders("a", 0).is_empty());
        assert_eq!(idx.holders("a", 32).len(), 1);
        assert_eq!(idx.dump().len(), 1);
        assert_eq!(idx.dump()[0].0, w(2, 0));
    }
}
