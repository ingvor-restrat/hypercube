//! Stateful generation processing and deterministic record/replay for Hypercube.
//!
//! The crate keeps transport, temporal state, and effects outside the pure
//! [`hypercube`] calculation engine. A [`DisruptorCircuit`] passes one coherent
//! snapshot at a time to a stateful [`FrameProcessor`]. The recording layer
//! persists accepted engine updates together with semantic outputs so a fresh
//! engine and processor can verify them later.

#![warn(missing_docs)]

mod capture;
mod circuit;
mod digest;
mod recording;
mod replay;
mod trigger;

pub use capture::{residual_score_trigger, CaptureError, CaptureSession};
pub use circuit::{
    CircuitConfig, CircuitError, CircuitWaitStrategy, DisruptorCircuit, FrameContext,
    FrameProcessor, ProcessedFrame,
};
pub use digest::{snapshot_digest, SnapshotDigest, SNAPSHOT_DIGEST_ALGORITHM};
pub use recording::{
    GenerationRecord, RecordingError, RecordingManifest, RecordingReader, RecordingWriter,
    RECORDING_FORMAT_VERSION,
};
pub use replay::{Divergence, DivergenceKind, ReplayError, ReplayReport, ReplayRunner};
pub use trigger::{
    ThresholdTriggerSpec, TriggerConfigError, TriggerFrame, TriggerProcessor, TriggerState,
    TriggerTransition, TriggerTransitionKind,
};
