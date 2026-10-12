//! Recoverable indexed projections over coherent engine snapshots.
//!
//! A single runtime owner stages graph revisions and evaluates generations.
//! Published frames own immutable contiguous strips and can be shared with
//! readers through `Arc`. No live memory-mapped payload is borrowed here.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::Snapshot;

/// An explicit source namespace; engine nodes and projections may share names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum ProjectionSource {
    /// A primitive or derived node in the supplied engine snapshot.
    Node(String),
    /// Another projection in the same graph revision and generation.
    Projection(String),
}

/// One ordered input membership. Repeated picks are permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pick {
    /// Strip to read.
    pub source: ProjectionSource,
    /// Stable key in the source's declared layout.
    pub key: String,
}

/// Recoverable definition of one stateless materialized view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionSpec {
    /// Unique projection identifier within the graph.
    pub id: String,
    /// Positive version; changing this definition requires a larger version.
    pub version: u64,
    /// Ordered inputs to the calculation, including any repeated memberships.
    pub picks: Vec<Pick>,
    /// Versioned function registry name, such as `weighted_sum@1`.
    pub function: String,
    /// Finite function-specific parameters (weights for `weighted_sum@1`).
    #[serde(default)]
    pub parameters: Vec<f64>,
    /// Ordered, unique keys of the resulting contiguous strip.
    pub output_keys: Vec<String>,
}

/// Complete graph revision, serializable with `serde_json` for recovery.
///
/// Node layouts declare selectable keys, not physical snapshot offsets. Each
/// generation is resolved by key, so reordering or omitted snapshot cells cannot
/// silently redirect a pick. Unselected missing cells are allowed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionConfig {
    /// Positive, monotonically increasing graph revision within a runtime.
    pub revision: u64,
    /// Stable selectable layouts for source engine nodes.
    pub node_layouts: BTreeMap<String, Vec<String>>,
    /// Complete set of projections, in any dependency order.
    pub projections: Vec<ProjectionSpec>,
}

/// Stateless calculation registered under a stable, versioned name.
///
/// Implementations must be deterministic and free of side effects: a failed
/// frame may be retried. `Send + Sync` permits shared ownership, but does not
/// enforce this semantic contract. Use a new registry name when semantics change.
pub trait ProjectionFunction: Send + Sync {
    /// Validate parameters and input/output shapes before activating a graph.
    fn validate(&self, inputs: usize, parameters: &[f64], outputs: usize) -> Result<()>;

    /// Calculate an owned output from finite, ordered input values.
    fn calculate(&self, inputs: &[f64], parameters: &[f64]) -> Result<Vec<f64>>;
}

/// Append-only registry of stateless functions used to compile definitions.
#[derive(Default)]
pub struct FunctionRegistry {
    functions: BTreeMap<String, Arc<dyn ProjectionFunction>>,
}

impl FunctionRegistry {
    /// Register `gather@1` (identity) and `weighted_sum@1` (unnormalized sum).
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        registry
            .functions
            .insert("gather@1".into(), Arc::new(Gather));
        registry
            .functions
            .insert("weighted_sum@1".into(), Arc::new(WeightedSum));
        registry
    }

    /// Register a function without replacing any existing implementation.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        function: impl ProjectionFunction + 'static,
    ) -> Result<()> {
        let name = name.into();
        identifier(&name)?;
        ensure!(
            !self.functions.contains_key(&name),
            "function already registered: {name}"
        );
        self.functions.insert(name, Arc::new(function));
        Ok(())
    }
}

struct Gather;

impl ProjectionFunction for Gather {
    fn validate(&self, inputs: usize, parameters: &[f64], outputs: usize) -> Result<()> {
        ensure!(parameters.is_empty(), "gather takes no parameters");
        ensure!(inputs == outputs, "gather needs one output key per pick");
        Ok(())
    }

    fn calculate(&self, inputs: &[f64], _parameters: &[f64]) -> Result<Vec<f64>> {
        Ok(inputs.to_vec())
    }
}

struct WeightedSum;

impl ProjectionFunction for WeightedSum {
    fn validate(&self, inputs: usize, parameters: &[f64], outputs: usize) -> Result<()> {
        ensure!(
            inputs == parameters.len(),
            "weighted sum needs one weight per pick"
        );
        ensure!(outputs == 1, "weighted sum needs exactly one output key");
        Ok(())
    }

    fn calculate(&self, inputs: &[f64], parameters: &[f64]) -> Result<Vec<f64>> {
        Ok(vec![inputs
            .iter()
            .zip(parameters)
            .map(|(x, w)| x * w)
            .sum()])
    }
}

/// An immutable contiguous result with its complete interpretation attached.
#[derive(Debug)]
pub struct ProjectionStrip {
    id: String,
    definition_version: u64,
    graph_revision: u64,
    generation: u64,
    keys: Vec<String>,
    values: Vec<f64>,
}

impl ProjectionStrip {
    /// Producing projection identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Producing definition version.
    pub fn definition_version(&self) -> u64 {
        self.definition_version
    }
    /// Complete graph revision used to calculate this strip.
    pub fn graph_revision(&self) -> u64 {
        self.graph_revision
    }
    /// Coherent engine generation used by all transitive inputs.
    pub fn generation(&self) -> u64 {
        self.generation
    }
    /// Stable output keys, in value order.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
    /// Contiguous, read-only output values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Look up an output by its stable key.
    pub fn value(&self, key: &str) -> Option<f64> {
        self.keys
            .iter()
            .position(|candidate| candidate == key)
            .map(|slot| self.values[slot])
    }
}

/// All results from one successful evaluation, retaining its inputs and config.
pub struct ProjectionFrame {
    source: Arc<Snapshot>,
    config: Arc<ProjectionConfig>,
    strips: BTreeMap<String, Arc<ProjectionStrip>>,
}

impl ProjectionFrame {
    /// Original coherent input snapshot, retained for this frame's lifetime.
    pub fn source(&self) -> &Arc<Snapshot> {
        &self.source
    }
    /// Exact recoverable definition of the evaluated graph.
    pub fn config(&self) -> &ProjectionConfig {
        &self.config
    }
    /// Input generation shared by every result.
    pub fn generation(&self) -> u64 {
        self.source.generation
    }
    /// Look up a strip; clone its `Arc` to retain it independently of the frame.
    pub fn strip(&self, id: &str) -> Option<&Arc<ProjectionStrip>> {
        self.strips.get(id)
    }
    /// Results in projection identifier order.
    pub fn strips(&self) -> &BTreeMap<String, Arc<ProjectionStrip>> {
        &self.strips
    }
}

/// Single-owner evaluator with transactional graph activation and publication.
///
/// Registration and evaluation require exclusive access. Readers receive owned
/// `Arc` frames/strips, never references into buffers reused by the runtime.
/// Evaluation is synchronous; independent runtimes may execute on other threads.
#[derive(Default)]
pub struct ProjectionRuntime {
    registry: FunctionRegistry,
    active: Option<CompiledGraph>,
    pending: Option<CompiledGraph>,
    latest: Option<Arc<ProjectionFrame>>,
    highest_revision: u64,
    definitions: BTreeMap<String, ProjectionSpec>,
}

impl ProjectionRuntime {
    /// Construct an empty runtime with the supplied function implementations.
    pub fn new(registry: FunctionRegistry) -> Self {
        Self {
            registry,
            ..Self::default()
        }
    }

    /// Register an additional function for future graph revisions.
    pub fn register_function(
        &mut self,
        name: impl Into<String>,
        function: impl ProjectionFunction + 'static,
    ) -> Result<()> {
        self.registry.register(name, function)
    }

    /// Validate and stage a complete revision for the next successful generation.
    ///
    /// This may replace a pending revision. Invalid configs leave all state
    /// unchanged. Definition versions cannot be reused for different definitions,
    /// even after removal or replacement of a pending revision in this runtime.
    pub fn stage(&mut self, config: ProjectionConfig) -> Result<()> {
        ensure!(
            config.revision > self.highest_revision,
            "graph revision must advance beyond {}",
            self.highest_revision
        );
        for spec in &config.projections {
            if let Some(previous) = self.definitions.get(&spec.id) {
                ensure!(
                    spec == previous || spec.version > previous.version,
                    "changed definition {} must advance beyond version {}",
                    spec.id,
                    previous.version
                );
            }
        }
        let graph = CompiledGraph::compile(config, &self.registry)?;
        self.highest_revision = graph.config.revision;
        for spec in &graph.config.projections {
            self.definitions.insert(spec.id.clone(), spec.clone());
        }
        self.pending = Some(graph);
        Ok(())
    }

    /// Last successfully published frame; staging or failures do not replace it.
    pub fn latest(&self) -> Option<&Arc<ProjectionFrame>> {
        self.latest.as_ref()
    }

    /// Configuration awaiting its first successful generation, if any.
    pub fn pending_config(&self) -> Option<&ProjectionConfig> {
        self.pending.as_ref().map(|graph| graph.config.as_ref())
    }

    /// Evaluate all projections from this snapshot and publish only on success.
    ///
    /// Missing/non-finite picked values, invalid function outputs, and stale
    /// generations are errors. A failed generation may be retried; no old values
    /// are substituted. A pending graph becomes active only when evaluation succeeds.
    pub fn evaluate(&mut self, source: Arc<Snapshot>) -> Result<Arc<ProjectionFrame>> {
        if let Some(previous) = &self.latest {
            ensure!(
                source.generation > previous.generation(),
                "generation {} is not newer than {}",
                source.generation,
                previous.generation()
            );
        }
        let graph = self
            .pending
            .as_ref()
            .or(self.active.as_ref())
            .context("no projection graph staged")?;
        let frame = Arc::new(graph.evaluate(source)?);
        if let Some(graph) = self.pending.take() {
            self.active = Some(graph);
        }
        self.latest = Some(Arc::clone(&frame));
        Ok(frame)
    }
}

enum CompiledPick {
    Node { source: usize, slot: usize },
    Projection { source: usize, slot: usize },
}

struct CompiledGraph {
    config: Arc<ProjectionConfig>,
    nodes: Vec<(String, Vec<String>)>,
    picks: Vec<Vec<CompiledPick>>,
    functions: Vec<Arc<dyn ProjectionFunction>>,
    order: Vec<usize>,
}

impl CompiledGraph {
    fn compile(config: ProjectionConfig, registry: &FunctionRegistry) -> Result<Self> {
        let nodes = config
            .node_layouts
            .iter()
            .map(|(id, keys)| (id.clone(), keys.clone()))
            .collect::<Vec<_>>();
        for (id, keys) in &nodes {
            identifier(id)?;
            layout(keys).with_context(|| format!("node {id}"))?;
        }
        let mut ids = BTreeMap::new();
        for (index, spec) in config.projections.iter().enumerate() {
            identifier(&spec.id)?;
            ensure!(
                ids.insert(spec.id.as_str(), index).is_none(),
                "duplicate projection {}",
                spec.id
            );
            ensure!(
                spec.version > 0,
                "projection {} needs a positive version",
                spec.id
            );
            ensure!(
                !spec.picks.is_empty(),
                "projection {} has no picks",
                spec.id
            );
            layout(&spec.output_keys).with_context(|| format!("projection {}", spec.id))?;
            ensure!(
                spec.parameters.iter().all(|p| p.is_finite()),
                "projection {} has non-finite parameters",
                spec.id
            );
        }
        let mut picks = Vec::new();
        let mut functions = Vec::new();
        for spec in &config.projections {
            let function = registry
                .functions
                .get(&spec.function)
                .with_context(|| format!("unknown function {}", spec.function))?;
            function
                .validate(spec.picks.len(), &spec.parameters, spec.output_keys.len())
                .with_context(|| format!("projection {}", spec.id))?;
            functions.push(Arc::clone(function));
            let mut compiled = Vec::new();
            for pick in &spec.picks {
                let (keys, source, is_node) = match &pick.source {
                    ProjectionSource::Node(id) => {
                        let source = nodes
                            .iter()
                            .position(|(name, _)| name == id)
                            .with_context(|| format!("unknown source node {id}"))?;
                        (&nodes[source].1, source, true)
                    }
                    ProjectionSource::Projection(id) => {
                        let source = *ids
                            .get(id.as_str())
                            .with_context(|| format!("unknown projection dependency {id}"))?;
                        (&config.projections[source].output_keys, source, false)
                    }
                };
                let slot = keys
                    .iter()
                    .position(|key| key == &pick.key)
                    .with_context(|| {
                        format!("unknown pick key {} in {:?}", pick.key, pick.source)
                    })?;
                compiled.push(if is_node {
                    CompiledPick::Node { source, slot }
                } else {
                    CompiledPick::Projection { source, slot }
                });
            }
            picks.push(compiled);
        }
        let mut order = Vec::new();
        let mut done = vec![false; picks.len()];
        while order.len() < picks.len() {
            let before = order.len();
            for (index, inputs) in picks.iter().enumerate() {
                if !done[index]
                    && inputs.iter().all(|pick| match pick {
                        CompiledPick::Node { .. } => true,
                        CompiledPick::Projection { source, .. } => done[*source],
                    })
                {
                    done[index] = true;
                    order.push(index);
                }
            }
            ensure!(order.len() > before, "projection dependency cycle");
        }
        Ok(Self {
            config: Arc::new(config),
            nodes,
            picks,
            functions,
            order,
        })
    }

    fn evaluate(&self, source: Arc<Snapshot>) -> Result<ProjectionFrame> {
        let mut cells = BTreeMap::new();
        for cell in &source.values {
            ensure!(
                cells
                    .insert((cell.node.as_str(), cell.key.as_str()), cell.value)
                    .is_none(),
                "duplicate snapshot coordinate ({}, {})",
                cell.node,
                cell.key
            );
        }
        let node_values = self
            .nodes
            .iter()
            .map(|(node, keys)| {
                keys.iter()
                    .map(|key| cells.get(&(node.as_str(), key.as_str())).copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut results: Vec<Option<Arc<ProjectionStrip>>> = vec![None; self.picks.len()];
        for &index in &self.order {
            let spec = &self.config.projections[index];
            let mut inputs = Vec::with_capacity(self.picks[index].len());
            for (pick_index, pick) in self.picks[index].iter().enumerate() {
                let value = match pick {
                    CompiledPick::Node { source, slot } => node_values[*source][*slot],
                    CompiledPick::Projection { source, slot } => {
                        results[*source].as_ref().map(|strip| strip.values[*slot])
                    }
                }
                .filter(|value| value.is_finite())
                .with_context(|| {
                    format!(
                        "projection {} missing finite input {:?}",
                        spec.id, spec.picks[pick_index]
                    )
                })?;
                inputs.push(value);
            }
            let values = self.functions[index]
                .calculate(&inputs, &spec.parameters)
                .with_context(|| format!("calculating projection {}", spec.id))?;
            ensure!(
                values.len() == spec.output_keys.len(),
                "projection {} returned an invalid output length",
                spec.id
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "projection {} returned non-finite output",
                spec.id
            );
            results[index] = Some(Arc::new(ProjectionStrip {
                id: spec.id.clone(),
                definition_version: spec.version,
                graph_revision: self.config.revision,
                generation: source.generation,
                keys: spec.output_keys.clone(),
                values,
            }));
        }
        let strips = results
            .into_iter()
            .flatten()
            .map(|strip| (strip.id.clone(), strip))
            .collect();
        Ok(ProjectionFrame {
            source,
            config: Arc::clone(&self.config),
            strips,
        })
    }
}

fn identifier(value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "identifier cannot be empty");
    Ok(())
}

fn layout(keys: &[String]) -> Result<()> {
    ensure!(!keys.is_empty(), "layout cannot be empty");
    let mut unique = BTreeSet::new();
    for key in keys {
        identifier(key)?;
        if !unique.insert(key) {
            bail!("duplicate output/layout key {key}");
        }
    }
    Ok(())
}
