//! Deterministic synthetic input for examples, integration tests, and demos.

use std::f64::consts::TAU;

use crate::{ExecutionMode, InputRow, NodeSpec, Transform, Update, WeightedInput};

#[derive(Debug, Clone)]
struct EntityState {
    key: String,
    level: f64,
    phase: f64,
}

/// Produces repeatable, correlated semi-live cross-sections without a data
/// vendor or external service.
#[derive(Debug, Clone)]
pub struct SyntheticInjector {
    rng: XorShift64,
    generation: u64,
    entities: Vec<EntityState>,
}

#[derive(Debug, Clone)]
struct OuEntityState {
    key: String,
    anchor_log_price: f64,
    previous_price: f64,
    idiosyncratic_state: f64,
    market_beta: f64,
    sector_beta: f64,
    sector: usize,
    dollar_volume_base: f64,
}

/// One deterministic synthetic market frame produced by [`OuMarketInjector`].
///
/// Rows contain `price`, `return`, `cumulative_return`, `dollar_volume`,
/// `log_dollar_volume`, `model_residual`, factor betas, and `sector` fields.
/// Callers can add application-specific fields before converting the frame
/// into an [`Update`].
#[derive(Debug, Clone, PartialEq)]
pub struct OuMarketFrame {
    /// Strictly increasing synthetic generation.
    pub generation: u64,
    /// Observation time supplied by the caller.
    pub observed_at_ms: i64,
    /// Stable fake symbols and their primitive market fields.
    pub rows: Vec<InputRow>,
}

impl OuMarketFrame {
    /// Convert the market frame into a live engine update.
    pub fn into_update(self, nodes: Vec<NodeSpec>) -> Update {
        Update {
            generation: self.generation,
            observed_at_ms: self.observed_at_ms,
            mode: ExecutionMode::Live,
            rows: self.rows,
            nodes,
        }
    }
}

/// Generates correlated mean-reverting log prices for financial examples.
///
/// The model is compact but financially recognizable. A shared
/// market state, one sector state, and one idiosyncratic state per symbol each
/// follow a discrete Ornstein--Uhlenbeck process. Entity log prices are a
/// weighted sum of those states around a stable anchor. The same seed always
/// produces the same frames.
#[derive(Debug, Clone)]
pub struct OuMarketInjector {
    rng: XorShift64,
    generation: u64,
    market_state: f64,
    sector_states: Vec<f64>,
    entities: Vec<OuEntityState>,
}

impl OuMarketInjector {
    /// Create a synthetic market with at least one symbol and four sectors.
    pub fn new(entity_count: usize, seed: u64) -> Self {
        let count = entity_count.max(1);
        let mut rng = XorShift64::new(seed);
        let entities = (0..count)
            .map(|index| {
                let anchor_price = 35.0 + index as f64 * 0.37 + rng.symmetric() * 4.0;
                let anchor_log_price = anchor_price.max(5.0).ln();
                OuEntityState {
                    key: format!("SIM{:04}", index + 1),
                    anchor_log_price,
                    previous_price: anchor_log_price.exp(),
                    idiosyncratic_state: 0.0,
                    market_beta: 0.75 + (index % 11) as f64 * 0.055,
                    sector_beta: 0.65 + (index % 7) as f64 * 0.07,
                    sector: index % 4,
                    dollar_volume_base: 500_000.0 + (index % 23) as f64 * 90_000.0,
                }
            })
            .collect();
        Self {
            rng,
            generation: 0,
            market_state: 0.0,
            sector_states: vec![0.0; 4],
            entities,
        }
    }

    /// Return the most recently emitted generation, or zero before first use.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return stable fake symbols in layout order.
    pub fn symbols(&self) -> Vec<String> {
        self.entities
            .iter()
            .map(|entity| entity.key.clone())
            .collect()
    }

    /// Advance the OU states by one simulated trading day.
    pub fn next_frame(&mut self, observed_at_ms: i64) -> OuMarketFrame {
        self.generation += 1;
        let dt: f64 = 1.0 / 252.0;
        let sqrt_dt = dt.sqrt();
        let previous_market = self.market_state;
        self.market_state += 1.6 * -self.market_state * dt + 0.12 * sqrt_dt * self.rng.normal();
        let market_move = self.market_state - previous_market;

        let mut sector_moves = Vec::with_capacity(self.sector_states.len());
        for state in &mut self.sector_states {
            let previous = *state;
            *state += 2.4 * -*state * dt + 0.16 * sqrt_dt * self.rng.normal();
            sector_moves.push(*state - previous);
        }

        let rows = self
            .entities
            .iter_mut()
            .map(|entity| {
                entity.idiosyncratic_state +=
                    4.0 * -entity.idiosyncratic_state * dt + 0.22 * sqrt_dt * self.rng.normal();
                let log_price = entity.anchor_log_price
                    + entity.market_beta * self.market_state
                    + entity.sector_beta * self.sector_states[entity.sector]
                    + entity.idiosyncratic_state;
                let price = log_price.exp().max(0.01);
                let tick_return = price / entity.previous_price - 1.0;
                let cumulative_return = price / entity.anchor_log_price.exp() - 1.0;
                let modeled_return = entity.market_beta * market_move
                    + entity.sector_beta * sector_moves[entity.sector];
                let model_residual = tick_return - modeled_return;
                let volume_noise = (0.35 * self.rng.normal()).exp();
                let dollar_volume =
                    entity.dollar_volume_base * volume_noise * (1.0 + 18.0 * tick_return.abs());
                entity.previous_price = price;

                InputRow::new(entity.key.clone(), observed_at_ms)
                    .with_field("price", price)
                    .with_field("return", tick_return)
                    .with_field("cumulative_return", cumulative_return)
                    .with_field("dollar_volume", dollar_volume)
                    .with_field("log_dollar_volume", dollar_volume.ln())
                    .with_field("model_residual", model_residual)
                    .with_field("market_beta", entity.market_beta)
                    .with_field("sector_beta", entity.sector_beta)
                    .with_field("sector", entity.sector as f64)
            })
            .collect();

        OuMarketFrame {
            generation: self.generation,
            observed_at_ms,
            rows,
        }
    }
}

/// Financial node graph used by the browser demonstration.
///
/// The simulated factor residual is
/// `epsilon_i = r_i - beta_market_i * delta_market
///                    - beta_sector_i * delta_sector`.
/// The displayed score is the cross-sectional z-score of
/// `0.75 * rank_z(epsilon_i) + 0.25 * z(log(dollar_volume_i))`.
/// It is an educational residual-move scanner, not a return forecast.
pub fn market_demo_nodes() -> Vec<NodeSpec> {
    vec![
        NodeSpec::field("price", "price", Transform::Identity),
        NodeSpec::field("return", "return", Transform::Identity),
        NodeSpec::field("residual_z", "model_residual", Transform::RankZScore),
        NodeSpec::field("dollar_volume_z", "log_dollar_volume", Transform::ZScore),
        NodeSpec::linear(
            "liquid_residual_score",
            vec![
                WeightedInput::required("residual_z", 0.75),
                WeightedInput::required("dollar_volume_z", 0.25),
            ],
            false,
            Transform::ZScore,
        ),
    ]
}

impl SyntheticInjector {
    /// Create at least one synthetic entity with a repeatable random seed.
    pub fn new(entity_count: usize, seed: u64) -> Self {
        let count = entity_count.max(1);
        let entities = (0..count)
            .map(|index| EntityState {
                key: format!("ENTITY-{:03}", index + 1),
                level: 80.0 + index as f64 * 2.5,
                phase: index as f64 / count as f64 * TAU,
            })
            .collect();
        Self {
            rng: XorShift64::new(seed),
            generation: 0,
            entities,
        }
    }

    /// Return the most recently emitted generation, or zero before first use.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return stable entity keys in layout order.
    pub fn entity_keys(&self) -> Vec<String> {
        self.entities
            .iter()
            .map(|entity| entity.key.clone())
            .collect()
    }

    /// Advance the generator and build one live [`Update`].
    pub fn next_update(&mut self, observed_at_ms: i64, nodes: Vec<NodeSpec>) -> Update {
        self.generation += 1;
        let cycle = self.generation as f64 * 0.09;
        let common = cycle.sin() * 0.003;
        let rows = self
            .entities
            .iter_mut()
            .map(|entity| {
                let noise = self.rng.symmetric();
                let previous = entity.level;
                let change = common + (cycle + entity.phase).sin() * 0.002 + noise * 0.0015;
                entity.level = (entity.level * (1.0 + change)).max(1.0);
                let activity =
                    50.0 + 30.0 * (cycle * 0.7 + entity.phase).cos() + self.rng.symmetric() * 8.0;
                let dispersion = (change - common).abs() * 10_000.0;

                InputRow::new(entity.key.clone(), observed_at_ms)
                    .with_field("level", entity.level)
                    .with_field("change", entity.level / previous - 1.0)
                    .with_field("activity", activity.max(0.0))
                    .with_field("dispersion", dispersion)
            })
            .collect();
        Update {
            generation: self.generation,
            observed_at_ms,
            mode: ExecutionMode::Live,
            rows,
            nodes,
        }
    }
}

/// A small domain-neutral graph retained for tests and API experimentation.
///
/// `level` is a positive simulated state, `change` is its one-step arithmetic
/// return, `activity` is a non-negative periodic intensity, and `dispersion`
/// is the absolute distance from that generation's common change in basis
/// points. Financial examples should prefer [`market_demo_nodes`].
pub fn demo_nodes() -> Vec<NodeSpec> {
    vec![
        NodeSpec::field("level", "level", Transform::Identity),
        NodeSpec::field("change", "change", Transform::RankZScore),
        NodeSpec::field("activity", "activity", Transform::ZScore),
        NodeSpec::field("dispersion", "dispersion", Transform::Percentile),
        NodeSpec::linear(
            "signal",
            vec![
                WeightedInput::required("change", 0.65),
                WeightedInput::required("activity", 0.25),
                WeightedInput::required("dispersion", -0.10),
            ],
            true,
            Transform::ZScore,
        ),
    ]
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn symmetric(&mut self) -> f64 {
        let unit = self.next() as f64 / u64::MAX as f64;
        unit * 2.0 - 1.0
    }

    fn normal(&mut self) -> f64 {
        let first = ((self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let second = (self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        (-2.0 * first.ln()).sqrt() * (TAU * second).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injector_is_repeatable_and_monotonic() {
        let mut left = SyntheticInjector::new(3, 42);
        let mut right = SyntheticInjector::new(3, 42);
        let first = left.next_update(1_000, demo_nodes());
        let same = right.next_update(1_000, demo_nodes());
        assert_eq!(first, same);
        assert_eq!(first.generation, 1);
        assert_eq!(left.next_update(2_000, demo_nodes()).generation, 2);
    }

    #[test]
    fn ou_market_is_repeatable_finite_and_mean_reverting() {
        let mut left = OuMarketInjector::new(8, 42);
        let mut right = OuMarketInjector::new(8, 42);
        let first = left.next_frame(1_000);
        let same = right.next_frame(1_000);
        assert_eq!(first, same);
        assert_eq!(first.generation, 1);
        assert_eq!(first.rows.len(), 8);
        assert!(first.rows.iter().all(|row| {
            row.fields.values().all(|value| value.is_finite()) && row.fields["price"] > 0.0
        }));
        assert_eq!(left.symbols().first().map(String::as_str), Some("SIM0001"));
        assert_eq!(left.next_frame(2_000).generation, 2);
    }
}
