//! A small execution engine for live, replayed, and batch multidimensional data.
//!
//! Hypercube separates two concerns:
//!
//! - [`hypercube_slice`] publishes typed, entity-aligned vectors for cheap
//!   cross-process reads.
//! - [`HypercubeEngine`] evaluates deterministic field and composite nodes over
//!   one coherent input generation.
//!
//! The engine is domain-neutral. Rows can represent instruments, sensors,
//! services, experiments, or any other stable entity set.

#![warn(missing_docs)]

mod engine;
mod publisher;
mod rolling;
pub mod synthetic;

pub use engine::{
    CellValue, CubeError, CubeResult, ExecutionMode, HypercubeEngine, InputRow, NodeKind, NodeSpec,
    NodeStatus, Snapshot, Transform, Update, WeightedInput,
};
pub use publisher::{PublishDurability, SlicePublisher};
pub use rolling::{RollingError, RollingMoments};

/// The memory-mapped live-state API used by Hypercube.
pub use hypercube_slice as slice;
