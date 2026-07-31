use std::fmt;

use hypercube::Snapshot;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh64::Xxh64;

/// Versioned algorithm name used by [`snapshot_digest`].
pub const SNAPSHOT_DIGEST_ALGORITHM: &str = "xxh64-hypercube-semantic-v1";

/// Stable digest of the semantic fields in one Hypercube snapshot.
///
/// Operational fields such as execution mode and per-node compute duration are
/// deliberately excluded. This lets a live snapshot and its replay compare
/// equally while retaining strict, bitwise comparison of calculated `f64`
/// values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotDigest {
    /// Name and version of the canonical hashing algorithm.
    pub algorithm: String,
    /// Unsigned xxHash64 value over the canonical snapshot representation.
    pub hash: u64,
    /// Number of calculated cell values represented by the digest.
    pub values: usize,
    /// Sum of per-node missing-value counts.
    pub missing: usize,
}

impl fmt::Display for SnapshotDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{:016x} values={} missing={}",
            self.algorithm, self.hash, self.values, self.missing
        )
    }
}

/// Calculate a stable semantic digest for `snapshot`.
///
/// The digest includes generation, observation time, entity count, ordered
/// cell coordinates and bitwise values, plus value/missing status counts. It
/// excludes [`hypercube::ExecutionMode`] and `NodeStatus::compute_micros`.
pub fn snapshot_digest(snapshot: &Snapshot) -> SnapshotDigest {
    let mut canonical = Xxh64::new(0);
    write_bytes(&mut canonical, SNAPSHOT_DIGEST_ALGORITHM.as_bytes());
    write_u64(&mut canonical, snapshot.generation);
    write_i64(&mut canonical, snapshot.observed_at_ms);
    write_usize(&mut canonical, snapshot.entity_count);
    write_usize(&mut canonical, snapshot.values.len());
    for value in &snapshot.values {
        write_str(&mut canonical, &value.node);
        write_str(&mut canonical, &value.key);
        write_u64(&mut canonical, value.value.to_bits());
        write_i64(&mut canonical, value.observed_at_ms);
    }
    write_usize(&mut canonical, snapshot.statuses.len());
    for status in &snapshot.statuses {
        write_str(&mut canonical, &status.node);
        write_usize(&mut canonical, status.values);
        write_usize(&mut canonical, status.missing);
    }

    SnapshotDigest {
        algorithm: SNAPSHOT_DIGEST_ALGORITHM.to_owned(),
        hash: canonical.digest(),
        values: snapshot.values.len(),
        missing: snapshot.statuses.iter().fold(0_usize, |total, status| {
            total.saturating_add(status.missing)
        }),
    }
}

fn write_str(output: &mut Xxh64, value: &str) {
    write_bytes(output, value.as_bytes());
}

fn write_bytes(output: &mut Xxh64, value: &[u8]) {
    write_u64(output, value.len() as u64);
    output.update(value);
}

fn write_usize(output: &mut Xxh64, value: usize) {
    write_u64(output, value as u64);
}

fn write_u64(output: &mut Xxh64, value: u64) {
    output.update(&value.to_le_bytes());
}

fn write_i64(output: &mut Xxh64, value: i64) {
    output.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use hypercube::{CellValue, ExecutionMode, NodeStatus, Snapshot};

    use super::*;

    fn snapshot(mode: ExecutionMode, compute_micros: u64, value: f64) -> Snapshot {
        Snapshot {
            generation: 7,
            observed_at_ms: 1_234,
            mode,
            entity_count: 1,
            values: vec![CellValue {
                node: "score".to_owned(),
                key: "A".to_owned(),
                value,
                observed_at_ms: 1_200,
            }],
            statuses: vec![NodeStatus {
                node: "score".to_owned(),
                values: 1,
                missing: 0,
                compute_micros,
            }],
        }
    }

    #[test]
    fn digest_ignores_mode_and_compute_duration() {
        let live = snapshot(ExecutionMode::Live, 10, 1.5);
        let replay = snapshot(ExecutionMode::Replay, 999, 1.5);

        assert_eq!(snapshot_digest(&live), snapshot_digest(&replay));
    }

    #[test]
    fn digest_detects_bitwise_value_change() {
        let first = snapshot(ExecutionMode::Live, 10, 1.5);
        let changed = snapshot(ExecutionMode::Live, 10, 1.500_000_000_1);

        assert_eq!(snapshot_digest(&first).hash, 10_865_084_671_589_957_843);
        assert_ne!(snapshot_digest(&first), snapshot_digest(&changed));
    }
}
