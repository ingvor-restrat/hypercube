//! Pure, generation-oriented calculation over entity cross-sections.
//!
//! This module owns validation, dependency resolution, transforms, and
//! snapshot construction. It performs no file or network I/O.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Selects the operational context recorded with a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// A continuing stream of strictly newer generations.
    Live,
    /// Deterministic re-evaluation of previously observed inputs.
    Replay,
    /// An offline bounded calculation.
    Batch,
}

/// Cross-sectional transformation applied after a node produces raw values.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    /// Preserve finite values unchanged.
    #[default]
    Identity,
    /// Center and scale using the population standard deviation.
    ZScore,
    /// Assign ascending, one-based ranks with average ranks for ties.
    Rank,
    /// Map ranks to the closed interval `[0, 1]`.
    Percentile,
    /// Rank values first, then z-score the resulting ranks.
    RankZScore,
}

/// Primitive values observed for one stable entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputRow {
    /// Stable entity key within this update.
    pub key: String,
    /// Unix-epoch milliseconds at which this row was observed.
    pub observed_at_ms: i64,
    /// Named primitive inputs available for calculation.
    pub fields: BTreeMap<String, f64>,
}

impl InputRow {
    /// Create an empty row for `key` at `observed_at_ms`.
    pub fn new(key: impl Into<String>, observed_at_ms: i64) -> Self {
        Self {
            key: key.into(),
            observed_at_ms,
            fields: BTreeMap::new(),
        }
    }

    /// Add or replace one named field and return the row for chaining.
    pub fn with_field(mut self, name: impl Into<String>, value: f64) -> Self {
        self.fields.insert(name.into(), value);
        self
    }
}

/// One weighted edge into a [`NodeKind::Linear`] node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedInput {
    /// Identifier of the upstream node.
    pub node: String,
    /// Signed coefficient applied to the upstream value.
    pub weight: f64,
    /// Whether a missing value suppresses this entity's output.
    #[serde(default = "required_by_default")]
    pub required: bool,
}

fn required_by_default() -> bool {
    true
}

impl WeightedInput {
    /// Declare a dependency that must have a value for each emitted entity.
    pub fn required(node: impl Into<String>, weight: f64) -> Self {
        Self {
            node: node.into(),
            weight,
            required: true,
        }
    }

    /// Declare a dependency that may be absent for an entity.
    pub fn optional(node: impl Into<String>, weight: f64) -> Self {
        Self {
            node: node.into(),
            weight,
            required: false,
        }
    }
}

/// Calculation performed by a node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NodeKind {
    /// Read one primitive field from every input row.
    Field {
        /// Name in [`InputRow::fields`].
        field: String,
    },
    /// Form a weighted combination of previously resolved nodes.
    Linear {
        /// Incoming dependency edges.
        inputs: Vec<WeightedInput>,
        /// Divide by the sum of absolute available weights when true.
        #[serde(default = "normalize_by_default")]
        normalize_weights: bool,
    },
}

fn normalize_by_default() -> bool {
    true
}

/// Version-independent declaration of one calculated cross-section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec {
    /// Unique node identifier within the update.
    pub id: String,
    /// Cross-sectional transformation applied to this node's raw output.
    #[serde(default)]
    pub transform: Transform,
    /// Primitive or derived calculation for this node.
    #[serde(flatten)]
    pub kind: NodeKind,
}

impl NodeSpec {
    /// Declare a node that reads one primitive field.
    pub fn field(id: impl Into<String>, field: impl Into<String>, transform: Transform) -> Self {
        Self {
            id: id.into(),
            transform,
            kind: NodeKind::Field {
                field: field.into(),
            },
        }
    }

    /// Declare a weighted derived node.
    pub fn linear(
        id: impl Into<String>,
        inputs: Vec<WeightedInput>,
        normalize_weights: bool,
        transform: Transform,
    ) -> Self {
        Self {
            id: id.into(),
            transform,
            kind: NodeKind::Linear {
                inputs,
                normalize_weights,
            },
        }
    }

    fn dependencies(&self) -> impl Iterator<Item = &WeightedInput> {
        match &self.kind {
            NodeKind::Field { .. } => [].iter(),
            NodeKind::Linear { inputs, .. } => inputs.iter(),
        }
    }
}

/// One coherent input generation presented to the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Update {
    /// Strictly increasing generation number.
    pub generation: u64,
    /// Publication time for the generation, in Unix-epoch milliseconds.
    pub observed_at_ms: i64,
    /// Operational context recorded in the output.
    pub mode: ExecutionMode,
    /// Primitive entity rows for this generation.
    pub rows: Vec<InputRow>,
    /// Complete graph to evaluate.
    pub nodes: Vec<NodeSpec>,
}

/// One calculated value at the `(node, entity)` coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellValue {
    /// Producing node identifier.
    pub node: String,
    /// Entity key.
    pub key: String,
    /// Finite calculated value.
    pub value: f64,
    /// Oldest observation time that contributed to this value.
    pub observed_at_ms: i64,
}

/// Per-node diagnostics for one completed generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node identifier.
    pub node: String,
    /// Number of emitted entity values.
    pub values: usize,
    /// Number of input entities without an emitted value.
    pub missing: usize,
    /// Local calculation duration in microseconds.
    pub compute_micros: u64,
}

/// Coherent result of evaluating every declared node for one generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Input generation represented by the snapshot.
    pub generation: u64,
    /// Publication time copied from the update.
    pub observed_at_ms: i64,
    /// Operational context copied from the update.
    pub mode: ExecutionMode,
    /// Number of distinct input entities.
    pub entity_count: usize,
    /// Values ordered first by node declaration, then by entity key.
    pub values: Vec<CellValue>,
    /// Status rows ordered by node declaration.
    pub statuses: Vec<NodeStatus>,
}

impl Snapshot {
    /// Select all values produced by `node`, in entity-key order.
    pub fn slice(&self, node: &str) -> Vec<&CellValue> {
        self.values
            .iter()
            .filter(|value| value.node == node)
            .collect()
    }

    /// Look up one calculated value by node and entity key.
    pub fn value(&self, node: &str, key: &str) -> Option<f64> {
        self.values
            .iter()
            .find(|value| value.node == node && value.key == key)
            .map(|value| value.value)
    }
}

/// Validation or dependency-resolution failure.
#[derive(Debug, Error, PartialEq)]
pub enum CubeError {
    /// The update did not advance beyond the last successful generation.
    #[error("generation {generation} is not newer than {previous}")]
    StaleGeneration {
        /// Rejected generation.
        generation: u64,
        /// Last successfully completed generation.
        previous: u64,
    },
    /// A required row, node, or field identifier was empty.
    #[error("identifier cannot be empty: {kind}")]
    EmptyIdentifier {
        /// Kind of identifier that was empty.
        kind: &'static str,
    },
    /// More than one input row used the same key.
    #[error("duplicate row key: {0}")]
    DuplicateRow(String),
    /// More than one node used the same identifier.
    #[error("duplicate node id: {0}")]
    DuplicateNode(String),
    /// A linear node declared no incoming edges.
    #[error("node {node} has no inputs")]
    EmptyLinearNode {
        /// Invalid node identifier.
        node: String,
    },
    /// A linear node repeated an upstream identifier.
    #[error("node {node} has duplicate input {input}")]
    DuplicateInput {
        /// Invalid node identifier.
        node: String,
        /// Repeated upstream identifier.
        input: String,
    },
    /// A coefficient was NaN or infinite.
    #[error("node {node} has a non-finite weight for {input}")]
    InvalidWeight {
        /// Invalid node identifier.
        node: String,
        /// Upstream edge carrying the bad coefficient.
        input: String,
    },
    /// A required edge referred to no declared node.
    #[error("node {node} requires unknown dependency {dependency}")]
    UnknownDependency {
        /// Invalid node identifier.
        node: String,
        /// Required upstream identifier that was not declared.
        dependency: String,
    },
    /// The remaining unresolved nodes form a dependency cycle.
    #[error("dependency cycle among nodes: {0}")]
    DependencyCycle(String),
}

/// Result type returned by the calculation engine.
pub type CubeResult<T> = Result<T, CubeError>;

/// Stateful generation guard around the otherwise pure node evaluator.
#[derive(Debug, Default)]
pub struct HypercubeEngine {
    last_generation: Option<u64>,
    cached_plan: Option<ExecutionPlan>,
    graph_compilations: u64,
}

impl HypercubeEngine {
    /// Construct an engine with no previously accepted generation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the last successfully completed generation.
    pub fn last_generation(&self) -> Option<u64> {
        self.last_generation
    }

    /// Return the number of graph plans compiled by this engine.
    ///
    /// A stable [`Update::nodes`] declaration is compiled once and reused by
    /// later generations. Changing any node declaration compiles a new plan.
    pub fn graph_compilations(&self) -> u64 {
        self.graph_compilations
    }

    /// Validate and evaluate one complete update.
    ///
    /// A failed update leaves [`Self::last_generation`] unchanged.
    pub fn update(&mut self, update: Update) -> CubeResult<Snapshot> {
        self.update_ref(&update)
    }

    /// Validate and evaluate one complete update without taking ownership.
    ///
    /// This is useful when the caller also needs to persist the accepted input
    /// for deterministic replay. A failed update leaves
    /// [`Self::last_generation`] unchanged.
    pub fn update_ref(&mut self, update: &Update) -> CubeResult<Snapshot> {
        if let Some(previous) = self.last_generation {
            if update.generation <= previous {
                return Err(CubeError::StaleGeneration {
                    generation: update.generation,
                    previous,
                });
            }
        }
        validate_rows(&update.rows)?;
        if !self
            .cached_plan
            .as_ref()
            .is_some_and(|plan| plan.nodes == update.nodes)
        {
            let plan = ExecutionPlan::compile(&update.nodes)?;
            self.cached_plan = Some(plan);
            self.graph_compilations = self.graph_compilations.saturating_add(1);
        }
        let plan = self
            .cached_plan
            .as_ref()
            .expect("a valid update always has a compiled plan");

        let mut entity_times = update
            .rows
            .iter()
            .map(|row| (row.key.as_str(), row.observed_at_ms))
            .collect::<Vec<_>>();
        entity_times.sort_unstable_by(|left, right| left.0.cmp(right.0));
        let mut resolved = vec![None; update.nodes.len()];
        let mut statuses = vec![None; update.nodes.len()];

        for &node_index in &plan.order {
            let spec = &update.nodes[node_index];
            let started = Instant::now();
            let values = match &spec.kind {
                NodeKind::Field { field } => compute_field(spec, field, &update.rows),
                NodeKind::Linear {
                    inputs,
                    normalize_weights,
                } => compute_linear(
                    spec,
                    inputs,
                    &plan.dependencies[node_index],
                    *normalize_weights,
                    &entity_times,
                    &resolved,
                ),
            };
            let elapsed = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
            let missing = update.rows.len().saturating_sub(values.len());
            statuses[node_index] = Some(NodeStatus {
                node: spec.id.clone(),
                values: values.len(),
                missing,
                compute_micros: elapsed,
            });
            resolved[node_index] = Some(values);
        }

        let values = resolved
            .into_iter()
            .flatten()
            .flat_map(BTreeMap::into_values)
            .collect();
        let statuses = statuses.into_iter().flatten().collect();
        let snapshot = Snapshot {
            generation: update.generation,
            observed_at_ms: update.observed_at_ms,
            mode: update.mode,
            entity_count: update.rows.len(),
            values,
            statuses,
        };
        self.last_generation = Some(update.generation);
        Ok(snapshot)
    }
}

#[derive(Debug)]
struct ExecutionPlan {
    nodes: Vec<NodeSpec>,
    order: Vec<usize>,
    dependencies: Vec<Vec<Option<usize>>>,
}

impl ExecutionPlan {
    fn compile(nodes: &[NodeSpec]) -> CubeResult<Self> {
        validate_nodes(nodes)?;
        let node_indexes = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id.as_str(), index))
            .collect::<HashMap<_, _>>();
        let dependencies = nodes
            .iter()
            .map(|node| {
                node.dependencies()
                    .map(|dependency| node_indexes.get(dependency.node.as_str()).copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut completed = vec![false; nodes.len()];
        let mut pending = (0..nodes.len()).collect::<Vec<_>>();
        let mut order = Vec::with_capacity(nodes.len());

        while !pending.is_empty() {
            let mut next = Vec::with_capacity(pending.len());
            let mut progressed = false;
            for node_index in pending {
                if dependencies[node_index]
                    .iter()
                    .flatten()
                    .all(|dependency| completed[*dependency])
                {
                    completed[node_index] = true;
                    order.push(node_index);
                    progressed = true;
                } else {
                    next.push(node_index);
                }
            }
            if !progressed {
                let ids = next
                    .iter()
                    .map(|index| nodes[*index].id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(CubeError::DependencyCycle(ids));
            }
            pending = next;
        }

        Ok(Self {
            nodes: nodes.to_vec(),
            order,
            dependencies,
        })
    }
}

fn validate_rows(rows: &[InputRow]) -> CubeResult<()> {
    let mut row_keys = BTreeSet::new();
    for row in rows {
        if row.key.trim().is_empty() {
            return Err(CubeError::EmptyIdentifier { kind: "row key" });
        }
        if !row_keys.insert(row.key.as_str()) {
            return Err(CubeError::DuplicateRow(row.key.clone()));
        }
    }
    Ok(())
}

fn validate_nodes(nodes: &[NodeSpec]) -> CubeResult<()> {
    let mut node_ids = BTreeSet::new();
    for node in nodes {
        if node.id.trim().is_empty() {
            return Err(CubeError::EmptyIdentifier { kind: "node id" });
        }
        if !node_ids.insert(node.id.as_str()) {
            return Err(CubeError::DuplicateNode(node.id.clone()));
        }
        match &node.kind {
            NodeKind::Field { field } if field.trim().is_empty() => {
                return Err(CubeError::EmptyIdentifier { kind: "field" });
            }
            NodeKind::Field { .. } => {}
            NodeKind::Linear { inputs, .. } => {
                if inputs.is_empty() {
                    return Err(CubeError::EmptyLinearNode {
                        node: node.id.clone(),
                    });
                }
                let mut seen = BTreeSet::new();
                for input in inputs {
                    if !input.weight.is_finite() {
                        return Err(CubeError::InvalidWeight {
                            node: node.id.clone(),
                            input: input.node.clone(),
                        });
                    }
                    if !seen.insert(input.node.as_str()) {
                        return Err(CubeError::DuplicateInput {
                            node: node.id.clone(),
                            input: input.node.clone(),
                        });
                    }
                }
            }
        }
    }

    for node in nodes {
        for dependency in node.dependencies().filter(|dependency| dependency.required) {
            if !node_ids.contains(dependency.node.as_str()) {
                return Err(CubeError::UnknownDependency {
                    node: node.id.clone(),
                    dependency: dependency.node.clone(),
                });
            }
        }
    }
    Ok(())
}

fn compute_field(spec: &NodeSpec, field: &str, rows: &[InputRow]) -> BTreeMap<String, CellValue> {
    let raw = rows
        .iter()
        .filter_map(|row| {
            row.fields
                .get(field)
                .copied()
                .filter(|value| value.is_finite())
                .map(|value| RawValue {
                    key: row.key.clone(),
                    value,
                    observed_at_ms: row.observed_at_ms,
                })
        })
        .collect();
    apply_transform(raw, spec.transform)
        .into_iter()
        .map(|value| {
            (
                value.key.clone(),
                CellValue {
                    node: spec.id.clone(),
                    key: value.key,
                    value: value.value,
                    observed_at_ms: value.observed_at_ms,
                },
            )
        })
        .collect()
}

fn compute_linear(
    spec: &NodeSpec,
    inputs: &[WeightedInput],
    dependency_indexes: &[Option<usize>],
    normalize_weights: bool,
    entity_times: &[(&str, i64)],
    resolved: &[Option<BTreeMap<String, CellValue>>],
) -> BTreeMap<String, CellValue> {
    let mut raw = Vec::new();
    for &(key, input_time) in entity_times {
        let mut value = 0.0;
        let mut scale = 0.0;
        let mut observed_at_ms = input_time;
        let mut any = false;
        let mut missing_required = false;
        for (input, dependency_index) in inputs.iter().zip(dependency_indexes) {
            let cell = dependency_index
                .and_then(|index| resolved[index].as_ref())
                .and_then(|values| values.get(key));
            match cell {
                Some(cell) => {
                    value += input.weight * cell.value;
                    scale += input.weight.abs();
                    observed_at_ms = observed_at_ms.min(cell.observed_at_ms);
                    any = true;
                }
                None if input.required => {
                    missing_required = true;
                    break;
                }
                None => {}
            }
        }
        if missing_required || !any || (normalize_weights && scale <= f64::EPSILON) {
            continue;
        }
        if normalize_weights {
            value /= scale;
        }
        if value.is_finite() {
            raw.push(RawValue {
                key: key.to_owned(),
                value,
                observed_at_ms,
            });
        }
    }
    apply_transform(raw, spec.transform)
        .into_iter()
        .map(|value| {
            (
                value.key.clone(),
                CellValue {
                    node: spec.id.clone(),
                    key: value.key,
                    value: value.value,
                    observed_at_ms: value.observed_at_ms,
                },
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
struct RawValue {
    key: String,
    value: f64,
    observed_at_ms: i64,
}

fn apply_transform(mut values: Vec<RawValue>, transform: Transform) -> Vec<RawValue> {
    match transform {
        Transform::Identity => {}
        Transform::ZScore => zscore(&mut values),
        Transform::Rank => rank(&mut values),
        Transform::Percentile => percentile(&mut values),
        Transform::RankZScore => {
            rank(&mut values);
            zscore(&mut values);
        }
    }
    values
}

fn zscore(values: &mut [RawValue]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().map(|value| value.value).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value.value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let standard_deviation = variance.sqrt();
    for value in values {
        value.value = if standard_deviation > f64::EPSILON {
            (value.value - mean) / standard_deviation
        } else {
            0.0
        };
    }
}

fn rank(values: &mut [RawValue]) {
    let ranks = ranks(values);
    for (value, rank) in values.iter_mut().zip(ranks) {
        value.value = rank;
    }
}

fn percentile(values: &mut [RawValue]) {
    if values.len() == 1 {
        values[0].value = 0.5;
        return;
    }
    let denominator = values.len().saturating_sub(1) as f64;
    let ranks = ranks(values);
    for (value, rank) in values.iter_mut().zip(ranks) {
        value.value = (rank - 1.0) / denominator;
    }
}

fn ranks(values: &[RawValue]) -> Vec<f64> {
    let mut ordered = values
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value.value))
        .collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].1 == ordered[start].1 {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 / 2.0;
        for index in start..end {
            ranks[ordered[index].0] = average_rank;
        }
        start = end;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(key: &str, a: f64, b: f64) -> InputRow {
        InputRow::new(key, 1_000)
            .with_field("a", a)
            .with_field("b", b)
    }

    #[test]
    fn field_and_composite_nodes_form_a_deterministic_cube() {
        let nodes = vec![
            NodeSpec::linear(
                "blend",
                vec![
                    WeightedInput::required("a_rank", 0.75),
                    WeightedInput::required("b_rank", -0.25),
                ],
                true,
                Transform::Identity,
            ),
            NodeSpec::field("a_rank", "a", Transform::RankZScore),
            NodeSpec::field("b_rank", "b", Transform::RankZScore),
        ];
        let mut engine = HypercubeEngine::new();
        let snapshot = engine
            .update(Update {
                generation: 1,
                observed_at_ms: 1_000,
                mode: ExecutionMode::Live,
                rows: vec![row("A", 1.0, 3.0), row("B", 2.0, 2.0), row("C", 3.0, 1.0)],
                nodes,
            })
            .unwrap();

        assert_eq!(snapshot.entity_count, 3);
        assert!(snapshot.value("blend", "A").unwrap() < 0.0);
        assert_eq!(snapshot.value("blend", "B"), Some(0.0));
        assert!(snapshot.value("blend", "C").unwrap() > 0.0);
        assert_eq!(snapshot.statuses.len(), 3);
    }

    #[test]
    fn ties_receive_average_ranks() {
        let mut engine = HypercubeEngine::new();
        let snapshot = engine
            .update(Update {
                generation: 1,
                observed_at_ms: 1_000,
                mode: ExecutionMode::Batch,
                rows: vec![row("A", 1.0, 0.0), row("B", 1.0, 0.0), row("C", 3.0, 0.0)],
                nodes: vec![NodeSpec::field("rank", "a", Transform::Rank)],
            })
            .unwrap();
        assert_eq!(snapshot.value("rank", "A"), Some(1.5));
        assert_eq!(snapshot.value("rank", "B"), Some(1.5));
        assert_eq!(snapshot.value("rank", "C"), Some(3.0));
    }

    #[test]
    fn stable_graphs_reuse_the_compiled_plan() {
        let mut engine = HypercubeEngine::new();
        for generation in 1..=3 {
            engine
                .update(Update {
                    generation,
                    observed_at_ms: generation as i64 * 1_000,
                    mode: ExecutionMode::Live,
                    rows: vec![row("A", generation as f64, 0.0)],
                    nodes: vec![NodeSpec::field("value", "a", Transform::Identity)],
                })
                .unwrap();
        }
        assert_eq!(engine.graph_compilations(), 1);

        engine
            .update(Update {
                generation: 4,
                observed_at_ms: 4_000,
                mode: ExecutionMode::Live,
                rows: vec![row("A", 4.0, 0.0)],
                nodes: vec![NodeSpec::field("rank", "a", Transform::Rank)],
            })
            .unwrap();
        assert_eq!(engine.graph_compilations(), 2);
    }

    #[test]
    fn cycles_and_stale_generations_are_rejected() {
        let mut engine = HypercubeEngine::new();
        let cyclic = vec![
            NodeSpec::linear(
                "a",
                vec![WeightedInput::required("b", 1.0)],
                true,
                Transform::Identity,
            ),
            NodeSpec::linear(
                "b",
                vec![WeightedInput::required("a", 1.0)],
                true,
                Transform::Identity,
            ),
        ];
        let error = engine
            .update(Update {
                generation: 1,
                observed_at_ms: 1_000,
                mode: ExecutionMode::Live,
                rows: vec![row("A", 1.0, 2.0)],
                nodes: cyclic,
            })
            .unwrap_err();
        assert!(matches!(error, CubeError::DependencyCycle(_)));

        engine
            .update(Update {
                generation: 1,
                observed_at_ms: 1_000,
                mode: ExecutionMode::Live,
                rows: vec![],
                nodes: vec![],
            })
            .unwrap();
        assert_eq!(
            engine
                .update(Update {
                    generation: 1,
                    observed_at_ms: 2_000,
                    mode: ExecutionMode::Live,
                    rows: vec![],
                    nodes: vec![],
                })
                .unwrap_err(),
            CubeError::StaleGeneration {
                generation: 1,
                previous: 1
            }
        );
    }
}
