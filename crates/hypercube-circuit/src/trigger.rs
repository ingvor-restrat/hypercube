use std::collections::{BTreeMap, BTreeSet};

use hypercube::Snapshot;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{snapshot_digest, FrameContext, FrameProcessor, SnapshotDigest};

/// Configuration for a persistent upper-threshold trigger with hysteresis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdTriggerSpec {
    /// Stable identifier included in emitted transitions.
    pub id: String,
    /// Hypercube node whose entity values are observed.
    pub node: String,
    /// Inactive entities begin qualifying at or above this value.
    pub enter_at_or_above: f64,
    /// Active entities exit at or below this value.
    pub exit_at_or_below: f64,
    /// Number of consecutive qualifying generations required to enter.
    pub min_consecutive: u32,
}

impl ThresholdTriggerSpec {
    /// Construct and validate one trigger specification.
    pub fn new(
        id: impl Into<String>,
        node: impl Into<String>,
        enter_at_or_above: f64,
        exit_at_or_below: f64,
        min_consecutive: u32,
    ) -> Result<Self, TriggerConfigError> {
        let spec = Self {
            id: id.into(),
            node: node.into(),
            enter_at_or_above,
            exit_at_or_below,
            min_consecutive,
        };
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<(), TriggerConfigError> {
        if self.id.trim().is_empty() {
            return Err(TriggerConfigError::EmptyIdentifier("trigger id"));
        }
        if self.node.trim().is_empty() {
            return Err(TriggerConfigError::EmptyIdentifier("trigger node"));
        }
        if !self.enter_at_or_above.is_finite() || !self.exit_at_or_below.is_finite() {
            return Err(TriggerConfigError::NonFiniteThreshold {
                trigger: self.id.clone(),
            });
        }
        if self.enter_at_or_above <= self.exit_at_or_below {
            return Err(TriggerConfigError::InvalidHysteresis {
                trigger: self.id.clone(),
                enter: self.enter_at_or_above,
                exit: self.exit_at_or_below,
            });
        }
        if self.min_consecutive == 0 {
            return Err(TriggerConfigError::ZeroPersistence {
                trigger: self.id.clone(),
            });
        }
        Ok(())
    }
}

/// Invalid threshold-trigger configuration.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TriggerConfigError {
    /// A required identifier was empty.
    #[error("{0} cannot be empty")]
    EmptyIdentifier(&'static str),
    /// A threshold was NaN or infinite.
    #[error("trigger {trigger} has a non-finite threshold")]
    NonFiniteThreshold {
        /// Invalid trigger identifier.
        trigger: String,
    },
    /// Entry did not sit strictly above exit.
    #[error("trigger {trigger} must enter above its exit threshold: enter={enter}, exit={exit}")]
    InvalidHysteresis {
        /// Invalid trigger identifier.
        trigger: String,
        /// Configured entry threshold.
        enter: f64,
        /// Configured exit threshold.
        exit: f64,
    },
    /// Persistence count was zero.
    #[error("trigger {trigger} must require at least one generation")]
    ZeroPersistence {
        /// Invalid trigger identifier.
        trigger: String,
    },
    /// Two specifications shared one identifier.
    #[error("duplicate trigger id: {0}")]
    DuplicateTrigger(String),
}

/// State change emitted by a threshold processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerTransitionKind {
    /// Entity satisfied entry persistence and became active.
    Entered,
    /// Active entity crossed the configured exit threshold.
    Exited,
    /// Active entity disappeared from the observed node.
    Invalidated,
}

/// One deterministic entity-level trigger transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriggerTransition {
    /// Trigger specification that produced the transition.
    pub trigger: String,
    /// Hypercube node observed by the trigger.
    pub node: String,
    /// Entity key whose state changed.
    pub key: String,
    /// Hypercube generation at which the transition occurred.
    pub generation: u64,
    /// Original observation time of the frame.
    pub observed_at_ms: i64,
    /// Enter, exit, or invalidation transition.
    pub kind: TriggerTransitionKind,
    /// Observed value, or `None` when missing data invalidated the state.
    pub value: Option<f64>,
}

/// One entity's complete state for a threshold trigger after a generation.
///
/// This is the dense callback output that an adapter can project into Slice
/// vectors or publish to new subscribers. Transitions remain the sparse
/// edge-event view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerState {
    /// Trigger specification that owns the state.
    pub trigger: String,
    /// Hypercube node observed by the trigger.
    pub node: String,
    /// Entity key represented by this state.
    pub key: String,
    /// Whether the trigger is currently active for the entity.
    pub active: bool,
    /// Current consecutive entry-qualification count.
    pub qualifying_generations: u32,
}

/// Deterministic output of threshold processing for one generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriggerFrame {
    /// Hypercube generation processed.
    pub generation: u64,
    /// Semantic digest of the complete input snapshot.
    pub snapshot: SnapshotDigest,
    /// Ordered state transitions emitted for this generation.
    pub transitions: Vec<TriggerTransition>,
    /// Complete ordered trigger-state cross-section after processing.
    pub states: Vec<TriggerState>,
}

#[derive(Debug, Clone, Default)]
struct EntityTriggerState {
    qualifying_generations: u32,
    active: bool,
}

/// Stateful processor implementing persistent entry, hysteretic exit, and
/// missing-value invalidation over Hypercube node values.
#[derive(Debug, Clone)]
pub struct TriggerProcessor {
    specs: Vec<ThresholdTriggerSpec>,
    nodes_by_trigger: BTreeMap<String, String>,
    states: BTreeMap<(String, String), EntityTriggerState>,
}

impl TriggerProcessor {
    /// Build a processor and reject duplicate or invalid specifications.
    pub fn new(specs: Vec<ThresholdTriggerSpec>) -> Result<Self, TriggerConfigError> {
        let mut ids = BTreeSet::new();
        let mut nodes_by_trigger = BTreeMap::new();
        for spec in &specs {
            spec.validate()?;
            if !ids.insert(spec.id.as_str()) {
                return Err(TriggerConfigError::DuplicateTrigger(spec.id.clone()));
            }
            nodes_by_trigger.insert(spec.id.clone(), spec.node.clone());
        }
        Ok(Self {
            specs,
            nodes_by_trigger,
            states: BTreeMap::new(),
        })
    }

    /// Return the immutable trigger configuration.
    pub fn specs(&self) -> &[ThresholdTriggerSpec] {
        &self.specs
    }

    fn process_snapshot(&mut self, context: FrameContext, snapshot: &Snapshot) -> TriggerFrame {
        let mut transitions = Vec::new();
        for spec in &self.specs {
            let mut seen = BTreeSet::new();
            for value in snapshot
                .values
                .iter()
                .filter(|value| value.node == spec.node)
            {
                seen.insert(value.key.as_str());
                let state = self
                    .states
                    .entry((spec.id.clone(), value.key.clone()))
                    .or_default();
                if state.active {
                    if value.value <= spec.exit_at_or_below {
                        state.active = false;
                        state.qualifying_generations = 0;
                        transitions.push(transition(
                            spec,
                            context,
                            &value.key,
                            TriggerTransitionKind::Exited,
                            Some(value.value),
                        ));
                    }
                } else if value.value >= spec.enter_at_or_above {
                    state.qualifying_generations = state.qualifying_generations.saturating_add(1);
                    if state.qualifying_generations >= spec.min_consecutive {
                        state.active = true;
                        state.qualifying_generations = 0;
                        transitions.push(transition(
                            spec,
                            context,
                            &value.key,
                            TriggerTransitionKind::Entered,
                            Some(value.value),
                        ));
                    }
                } else {
                    state.qualifying_generations = 0;
                }
            }

            let missing = self
                .states
                .iter()
                .filter_map(|((trigger, key), state)| {
                    (trigger == &spec.id && !seen.contains(key.as_str()))
                        .then_some((key.clone(), state.active))
                })
                .collect::<Vec<_>>();
            for (key, was_active) in missing {
                self.states.remove(&(spec.id.clone(), key.clone()));
                if was_active {
                    transitions.push(transition(
                        spec,
                        context,
                        &key,
                        TriggerTransitionKind::Invalidated,
                        None,
                    ));
                }
            }
        }
        transitions.sort_by(|left, right| {
            (
                left.trigger.as_str(),
                left.key.as_str(),
                left.kind,
                left.node.as_str(),
            )
                .cmp(&(
                    right.trigger.as_str(),
                    right.key.as_str(),
                    right.kind,
                    right.node.as_str(),
                ))
        });
        let states = self
            .states
            .iter()
            .map(|((trigger, key), state)| {
                let node = self
                    .nodes_by_trigger
                    .get(trigger)
                    .expect("state keys are created only from configured triggers")
                    .clone();
                TriggerState {
                    trigger: trigger.clone(),
                    node,
                    key: key.clone(),
                    active: state.active,
                    qualifying_generations: state.qualifying_generations,
                }
            })
            .collect();

        TriggerFrame {
            generation: context.generation,
            snapshot: snapshot_digest(snapshot),
            transitions,
            states,
        }
    }
}

impl FrameProcessor for TriggerProcessor {
    type Output = TriggerFrame;

    fn process(&mut self, context: FrameContext, snapshot: &Snapshot) -> Self::Output {
        self.process_snapshot(context, snapshot)
    }
}

fn transition(
    spec: &ThresholdTriggerSpec,
    context: FrameContext,
    key: &str,
    kind: TriggerTransitionKind,
    value: Option<f64>,
) -> TriggerTransition {
    TriggerTransition {
        trigger: spec.id.clone(),
        node: spec.node.clone(),
        key: key.to_owned(),
        generation: context.generation,
        observed_at_ms: context.observed_at_ms,
        kind,
        value,
    }
}

#[cfg(test)]
mod tests {
    use hypercube::{CellValue, ExecutionMode, NodeStatus};

    use super::*;

    fn snapshot(generation: u64, values: &[(&str, f64)]) -> Snapshot {
        Snapshot {
            generation,
            observed_at_ms: generation as i64 * 100,
            mode: ExecutionMode::Live,
            entity_count: values.len(),
            values: values
                .iter()
                .map(|(key, value)| CellValue {
                    node: "score".to_owned(),
                    key: (*key).to_owned(),
                    value: *value,
                    observed_at_ms: generation as i64 * 100,
                })
                .collect(),
            statuses: vec![NodeStatus {
                node: "score".to_owned(),
                values: values.len(),
                missing: 0,
                compute_micros: 1,
            }],
        }
    }

    fn context(generation: u64) -> FrameContext {
        FrameContext {
            circuit_sequence: generation as i64 - 1,
            generation,
            observed_at_ms: generation as i64 * 100,
            mode: ExecutionMode::Live,
        }
    }

    #[test]
    fn requires_persistence_and_uses_hysteresis() {
        let spec = ThresholdTriggerSpec::new("high", "score", 1.0, 0.5, 2).unwrap();
        let mut processor = TriggerProcessor::new(vec![spec]).unwrap();

        assert!(processor
            .process_snapshot(context(1), &snapshot(1, &[("A", 1.1)]))
            .transitions
            .is_empty());
        let entered = processor.process_snapshot(context(2), &snapshot(2, &[("A", 1.2)]));
        assert_eq!(entered.transitions.len(), 1);
        assert_eq!(entered.transitions[0].kind, TriggerTransitionKind::Entered);
        assert!(entered.states[0].active);
        assert_eq!(entered.states[0].qualifying_generations, 0);

        assert!(processor
            .process_snapshot(context(3), &snapshot(3, &[("A", 0.7)]))
            .transitions
            .is_empty());
        let exited = processor.process_snapshot(context(4), &snapshot(4, &[("A", 0.4)]));
        assert_eq!(exited.transitions[0].kind, TriggerTransitionKind::Exited);
    }

    #[test]
    fn missing_active_entity_is_invalidated() {
        let spec = ThresholdTriggerSpec::new("high", "score", 1.0, 0.5, 1).unwrap();
        let mut processor = TriggerProcessor::new(vec![spec]).unwrap();
        processor.process_snapshot(context(1), &snapshot(1, &[("A", 1.1)]));

        let invalidated = processor.process_snapshot(context(2), &snapshot(2, &[]));
        assert_eq!(
            invalidated.transitions[0].kind,
            TriggerTransitionKind::Invalidated
        );
        assert_eq!(invalidated.transitions[0].value, None);
    }

    #[test]
    fn missing_candidate_breaks_entry_persistence() {
        let spec = ThresholdTriggerSpec::new("high", "score", 1.0, 0.5, 2).unwrap();
        let mut processor = TriggerProcessor::new(vec![spec]).unwrap();

        assert!(processor
            .process_snapshot(context(1), &snapshot(1, &[("A", 1.1)]))
            .transitions
            .is_empty());
        assert!(processor
            .process_snapshot(context(2), &snapshot(2, &[]))
            .transitions
            .is_empty());
        assert!(processor
            .process_snapshot(context(3), &snapshot(3, &[("A", 1.2)]))
            .transitions
            .is_empty());
        let entered = processor.process_snapshot(context(4), &snapshot(4, &[("A", 1.3)]));
        assert_eq!(entered.transitions.len(), 1);
        assert_eq!(entered.transitions[0].kind, TriggerTransitionKind::Entered);
    }
}
