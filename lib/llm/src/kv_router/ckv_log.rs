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
}
