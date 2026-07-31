use std::io::BufRead;
use std::sync::Arc;

use hypercube::{CubeError, ExecutionMode, HypercubeEngine};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CircuitConfig, CircuitError, DisruptorCircuit, RecordingError, RecordingReader,
    TriggerConfigError, TriggerProcessor,
};

/// Kind of semantic discrepancy found during replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceKind {
    /// Replayed Disruptor sequence differed from the recording.
    CircuitSequence,
    /// Calculated Hypercube snapshot digest differed.
    SnapshotDigest,
    /// Stateful trigger transitions differed.
    TriggerTransitions,
    /// Complete stateful trigger cross-section differed.
    TriggerStates,
}

/// First concrete expected-versus-actual discrepancy in a replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    /// Hypercube generation containing the discrepancy.
    pub generation: u64,
    /// Semantic category that differed.
    pub kind: DivergenceKind,
    /// Compact expected representation.
    pub expected: String,
    /// Compact replayed representation.
    pub actual: String,
}

/// Aggregate exactness report for one replay run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReport {
    /// Original run identifier from the recording manifest.
    pub source_run_id: String,
    /// Build identifier from the recording manifest.
    pub source_build_id: String,
    /// Number of complete generations replayed.
    pub generations: usize,
    /// Number of calculated cell values checked.
    pub values: usize,
    /// Number of transitions recorded in the original run.
    pub expected_transitions: usize,
    /// Number of transitions emitted during replay.
    pub actual_transitions: usize,
    /// Number of trigger-state cells recorded in the original run.
    pub expected_states: usize,
    /// Number of trigger-state cells reconstructed during replay.
    pub actual_states: usize,
    /// Number of generations containing at least one discrepancy.
    pub divergent_generations: usize,
    /// Earliest concrete divergence, if any.
    pub first_divergence: Option<Divergence>,
}

impl ReplayReport {
    /// Return true when every generation, digest, and transition matched.
    pub fn is_exact(&self) -> bool {
        self.divergent_generations == 0
    }
}

/// Failure to construct or execute a replay.
#[derive(Debug, Error)]
pub enum ReplayError {
    /// Recording input was malformed or inconsistent.
    #[error(transparent)]
    Recording(#[from] RecordingError),
    /// Stateful trigger configuration was invalid.
    #[error(transparent)]
    Trigger(#[from] TriggerConfigError),
    /// Hypercube rejected a recorded update.
    #[error(transparent)]
    Engine(#[from] CubeError),
    /// Disruptor submission or completion failed.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    /// Expected or actual transition details could not be rendered.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Deterministic lockstep runner over a versioned generation recording.
#[derive(Debug, Clone, Copy)]
pub struct ReplayRunner {
    circuit_config: CircuitConfig,
}

impl ReplayRunner {
    /// Construct a runner with the supplied Disruptor settings.
    pub fn new(circuit_config: CircuitConfig) -> Self {
        Self { circuit_config }
    }

    /// Replay all generations through a fresh engine and fresh trigger state.
    pub fn run<R: BufRead>(
        &self,
        mut recording: RecordingReader<R>,
    ) -> Result<ReplayReport, ReplayError> {
        let manifest = recording.manifest().clone();
        let processor = TriggerProcessor::new(manifest.triggers)?;
        let mut circuit = DisruptorCircuit::with_processor(processor, self.circuit_config)?;
        let mut engine = HypercubeEngine::new();
        let mut report = ReplayReport {
            source_run_id: manifest.run_id,
            source_build_id: manifest.build_id,
            generations: 0,
            values: 0,
            expected_transitions: 0,
            actual_transitions: 0,
            expected_states: 0,
            actual_states: 0,
            divergent_generations: 0,
            first_divergence: None,
        };

        while let Some(record) = recording.read_generation()? {
            let mut update = record.update;
            update.mode = ExecutionMode::Replay;
            let snapshot = engine.update(update)?;
            let actual = circuit.process(Arc::new(snapshot))?;
            let expected_sequence = record.circuit_sequence;
            let actual_sequence = u64::try_from(actual.circuit_sequence).ok();
            let actual_output = actual.output;

            report.generations += 1;
            report.values = report.values.saturating_add(actual_output.snapshot.values);
            report.expected_transitions = report
                .expected_transitions
                .saturating_add(record.expected.transitions.len());
            report.actual_transitions = report
                .actual_transitions
                .saturating_add(actual_output.transitions.len());
            report.expected_states = report
                .expected_states
                .saturating_add(record.expected.states.len());
            report.actual_states = report
                .actual_states
                .saturating_add(actual_output.states.len());

            let mut generation_diverged = false;
            if actual_sequence != Some(expected_sequence) {
                generation_diverged = true;
                set_first(
                    &mut report,
                    Divergence {
                        generation: actual_output.generation,
                        kind: DivergenceKind::CircuitSequence,
                        expected: expected_sequence.to_string(),
                        actual: actual_sequence
                            .map(|sequence| sequence.to_string())
                            .unwrap_or_else(|| actual.circuit_sequence.to_string()),
                    },
                );
            }
            if actual_output.snapshot != record.expected.snapshot {
                generation_diverged = true;
                set_first(
                    &mut report,
                    Divergence {
                        generation: actual_output.generation,
                        kind: DivergenceKind::SnapshotDigest,
                        expected: record.expected.snapshot.to_string(),
                        actual: actual_output.snapshot.to_string(),
                    },
                );
            }
            if actual_output.transitions != record.expected.transitions {
                generation_diverged = true;
                let (expected, actual) =
                    sequence_difference(&record.expected.transitions, &actual_output.transitions)?;
                set_first(
                    &mut report,
                    Divergence {
                        generation: actual_output.generation,
                        kind: DivergenceKind::TriggerTransitions,
                        expected,
                        actual,
                    },
                );
            }
            if actual_output.states != record.expected.states {
                generation_diverged = true;
                let (expected, actual) =
                    sequence_difference(&record.expected.states, &actual_output.states)?;
                set_first(
                    &mut report,
                    Divergence {
                        generation: actual_output.generation,
                        kind: DivergenceKind::TriggerStates,
                        expected,
                        actual,
                    },
                );
            }
            if generation_diverged {
                report.divergent_generations += 1;
            }
        }
        Ok(report)
    }
}

impl Default for ReplayRunner {
    fn default() -> Self {
        Self::new(CircuitConfig::default())
    }
}

fn set_first(report: &mut ReplayReport, divergence: Divergence) {
    if report.first_divergence.is_none() {
        report.first_divergence = Some(divergence);
    }
}

fn sequence_difference<T: PartialEq + Serialize>(
    expected: &[T],
    actual: &[T],
) -> Result<(String, String), serde_json::Error> {
    let index = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    Ok((
        render_sequence_item(expected, index)?,
        render_sequence_item(actual, index)?,
    ))
}

fn render_sequence_item<T: Serialize>(
    values: &[T],
    index: usize,
) -> Result<String, serde_json::Error> {
    let item = values
        .get(index)
        .map(serde_json::to_string)
        .transpose()?
        .unwrap_or_else(|| "<missing>".to_owned());
    Ok(format!(
        "count={} first_difference[{index}]={item}",
        values.len()
    ))
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use hypercube::synthetic::{market_demo_nodes, OuMarketInjector};

    use crate::{
        capture::residual_score_trigger, CaptureSession, RecordingManifest, RecordingReader,
        RecordingWriter,
    };

    use super::*;

    fn recorded_bytes(generations: usize) -> Vec<u8> {
        let manifest = RecordingManifest::new(
            "synthetic-run",
            "test-build",
            "test-config",
            "test-layout",
            vec![residual_score_trigger().unwrap()],
        )
        .unwrap();
        let mut injector = OuMarketInjector::new(12, 42);
        let mut session =
            CaptureSession::new(Vec::new(), manifest, CircuitConfig::default()).unwrap();
        for index in 0..generations {
            let update = injector
                .next_frame(1_000 + index as i64 * 100)
                .into_update(market_demo_nodes());
            session.process(update).unwrap();
        }
        session.finish().unwrap()
    }

    #[test]
    fn exact_replay_reproduces_snapshots_state_and_transitions() {
        let bytes = recorded_bytes(20);
        let reader = RecordingReader::new(BufReader::new(Cursor::new(bytes))).unwrap();
        let report = ReplayRunner::default().run(reader).unwrap();

        assert!(report.is_exact(), "{report:#?}");
        assert_eq!(report.generations, 20);
        assert!(report.values > 0);
        assert!(report.actual_states > 0);
        assert_eq!(report.expected_states, report.actual_states);
        assert_eq!(report.first_divergence, None);
    }

    #[test]
    fn replay_reports_changed_input() {
        let bytes = recorded_bytes(4);
        let reader = RecordingReader::new(BufReader::new(Cursor::new(bytes))).unwrap();
        let manifest = reader.manifest().clone();
        let mut records = reader.collect::<Result<Vec<_>, _>>().unwrap();
        records[1].update.rows[0]
            .fields
            .insert("model_residual".to_owned(), 99.0);

        let mut changed = Vec::new();
        {
            let mut writer = RecordingWriter::new(&mut changed, manifest).unwrap();
            for record in &records {
                writer.write_generation(record).unwrap();
            }
            writer.flush().unwrap();
        }
        let replay = RecordingReader::new(BufReader::new(Cursor::new(changed))).unwrap();
        let report = ReplayRunner::default().run(replay).unwrap();

        assert!(!report.is_exact());
        assert_eq!(
            report.first_divergence.as_ref().map(|item| item.kind),
            Some(DivergenceKind::SnapshotDigest)
        );
    }

    #[test]
    fn serialized_update_recalculates_bitwise_values() {
        let mut injector = OuMarketInjector::new(12, 42);
        let update = injector.next_frame(1_000).into_update(market_demo_nodes());
        let encoded = serde_json::to_vec(&update).unwrap();
        let mut decoded = serde_json::from_slice::<hypercube::Update>(&encoded).unwrap();
        decoded.mode = ExecutionMode::Replay;

        let live = HypercubeEngine::new().update(update).unwrap();
        let replay = HypercubeEngine::new().update(decoded).unwrap();
        for (left, right) in live.values.iter().zip(&replay.values) {
            assert_eq!(
                left.value.to_bits(),
                right.value.to_bits(),
                "{} {}: {} != {}",
                left.node,
                left.key,
                left.value,
                right.value
            );
        }
    }
}
