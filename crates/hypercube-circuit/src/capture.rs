use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;

use hypercube::{CubeError, HypercubeEngine, Update};
use thiserror::Error;

use crate::{
    recording::validate_recordable_update, CircuitConfig, CircuitError, DisruptorCircuit,
    GenerationRecord, RecordingError, RecordingManifest, RecordingWriter, ThresholdTriggerSpec,
    TriggerConfigError, TriggerFrame, TriggerProcessor,
};

/// Failure while calculating, processing, or recording a live generation.
#[derive(Debug, Error)]
pub enum CaptureError {
    /// Stateful trigger configuration was invalid.
    #[error(transparent)]
    Trigger(#[from] TriggerConfigError),
    /// Hypercube rejected or failed the update.
    #[error(transparent)]
    Engine(#[from] CubeError),
    /// Disruptor submission or completion failed.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    /// Recording could not be written.
    #[error(transparent)]
    Recording(#[from] RecordingError),
    /// A prior post-compute failure made continued recording ambiguous.
    #[error("capture session is unusable after a post-compute failure")]
    Poisoned,
    /// Disruptor returned a negative sequence that cannot enter the file format.
    #[error("negative circuit sequence {0} cannot be recorded")]
    NegativeSequence(i64),
}

/// Live calculation session that records reproducible generations and their
/// expected stateful outputs.
///
/// This adapter emits versioned JSON Lines. Pass a buffered file, in-memory
/// vector, or another byte sink implementing [`Write`]. Message transports can
/// map the public manifest and generation structs directly instead.
pub struct CaptureSession<W: Write> {
    engine: HypercubeEngine,
    circuit: DisruptorCircuit<TriggerFrame>,
    recording: RecordingWriter<W>,
    poisoned: bool,
}

impl<W: Write> CaptureSession<W> {
    /// Start a fresh engine, trigger processor, and versioned recording.
    pub fn new(
        writer: W,
        manifest: RecordingManifest,
        config: CircuitConfig,
    ) -> Result<Self, CaptureError> {
        let processor = TriggerProcessor::new(manifest.triggers.clone())?;
        let circuit = DisruptorCircuit::with_processor(processor, config)?;
        let recording = RecordingWriter::new(writer, manifest)?;
        Ok(Self {
            engine: HypercubeEngine::new(),
            circuit,
            recording,
            poisoned: false,
        })
    }

    /// Evaluate, process, and append one complete input generation.
    ///
    /// Only successfully calculated and processed generations are appended.
    /// A failure after the engine advances poisons the session because
    /// continuing would create a gap between engine and recording state.
    pub fn process(&mut self, update: Update) -> Result<TriggerFrame, CaptureError> {
        self.process_with_positions(update, BTreeMap::new())
    }

    /// Evaluate and record one generation with upstream Aeron or transport
    /// frontiers.
    ///
    /// Position names are application-defined stable stream identities. Their
    /// values should denote the last input included in this generation.
    pub fn process_with_positions(
        &mut self,
        update: Update,
        source_positions: BTreeMap<String, i64>,
    ) -> Result<TriggerFrame, CaptureError> {
        if self.poisoned {
            return Err(CaptureError::Poisoned);
        }
        validate_recordable_update(&update)?;
        let snapshot = self.engine.update_ref(&update)?;
        let processed = match self.circuit.process(Arc::new(snapshot)) {
            Ok(processed) => processed,
            Err(error) => {
                self.poisoned = true;
                return Err(error.into());
            }
        };
        let circuit_sequence = match u64::try_from(processed.circuit_sequence) {
            Ok(sequence) => sequence,
            Err(_) => {
                self.poisoned = true;
                return Err(CaptureError::NegativeSequence(processed.circuit_sequence));
            }
        };
        let record = GenerationRecord {
            circuit_sequence,
            update,
            source_positions,
            expected: processed.output,
        };
        if let Err(error) = self.recording.write_generation(&record) {
            self.poisoned = true;
            return Err(error.into());
        }
        Ok(record.expected)
    }

    /// Flush the recording and return its underlying writer.
    pub fn finish(self) -> Result<W, CaptureError> {
        if self.poisoned {
            return Err(CaptureError::Poisoned);
        }
        Ok(self.recording.into_inner()?)
    }

    /// Return whether a post-compute failure prevents safe continuation.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

/// Build the standard residual-score trigger used by examples and smoke tests.
///
/// Applications should normally construct their own specifications. This
/// helper keeps the runnable replay example concise.
pub fn residual_score_trigger() -> Result<ThresholdTriggerSpec, TriggerConfigError> {
    ThresholdTriggerSpec::new(
        "persistent_liquid_residual",
        "liquid_residual_score",
        1.0,
        0.5,
        2,
    )
}
