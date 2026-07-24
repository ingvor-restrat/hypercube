//! `pairs` is a top-like statistical-arbitrage monitor built on Hypercube.
//!
//! Each synthetic pair shares a stochastic trend while its log-price residual
//! follows a stationary AR(1) process. Hypercube evaluates a declared,
//! pair-aligned graph and ranks the largest standardized dislocations.

use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::io::{self, IsTerminal, Write};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use hypercube::{ExecutionMode, HypercubeEngine, InputRow, NodeSpec, Snapshot, Transform, Update};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[38;5;45m";
const BLUE: &str = "\x1b[38;5;75m";
const GREEN: &str = "\x1b[38;5;84m";
const RED: &str = "\x1b[38;5;203m";
const AMBER: &str = "\x1b[38;5;221m";
const MUTED: &str = "\x1b[38;5;244m";
const WHITE: &str = "\x1b[38;5;255m";
const SPARKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

#[derive(Debug)]
struct Config {
    pairs: usize,
    top: usize,
    interval: Duration,
    ticks: Option<usize>,
    seed: u64,
    record: bool,
    no_color: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            pairs: 24,
            top: 10,
            interval: Duration::from_millis(180),
            ticks: None,
            seed: 0x5a17,
            record: false,
            no_color: false,
        }
    }
}

#[derive(Debug)]
struct PairState {
    id: String,
    leg_a: String,
    leg_b: String,
    alpha: f64,
    hedge_ratio: f64,
    phi: f64,
    innovation_sigma: f64,
    log_b: f64,
    spread: f64,
    price_a: f64,
    price_b: f64,
    spread_z: f64,
    half_life: f64,
    history: Vec<f64>,
}

#[derive(Debug)]
struct PairRow {
    id: String,
    legs: String,
    price_a: f64,
    price_b: f64,
    hedge_ratio: f64,
    spread_z: f64,
    half_life: f64,
    opportunity_rank: f64,
    history: Vec<f64>,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let mut rng = XorShift64::new(config.seed);
    let mut pairs = build_pairs(config.pairs);
    let mut engine = HypercubeEngine::new();
    let stdout_is_terminal = io::stdout().is_terminal();
    let interactive = stdout_is_terminal && !config.record;
    let color = !config.no_color && (interactive || config.record);
    let mut terminal = TerminalSession::enter(interactive)?;
    let limit = config
        .ticks
        .or(config.record.then_some(32))
        .or((!interactive).then_some(1));
    let mut tick = 0_usize;

    loop {
        tick += 1;
        update_pairs(&mut pairs, &mut rng);
        let observed_at_ms = tick as i64 * 86_400_000;
        let input_rows = pairs
            .iter()
            .map(|pair| {
                InputRow::new(pair.id.clone(), observed_at_ms)
                    .with_field("price_a", pair.price_a)
                    .with_field("price_b", pair.price_b)
                    .with_field("spread_z", pair.spread_z)
                    .with_field("abs_spread_z", pair.spread_z.abs())
                    .with_field("half_life", pair.half_life)
            })
            .collect();
        let snapshot = engine.update(Update {
            generation: tick as u64,
            observed_at_ms,
            mode: ExecutionMode::Live,
            rows: input_rows,
            nodes: demo_nodes(),
        })?;
        let ranked = ranked_rows(&snapshot, &pairs);
        let output = render(&config, &snapshot, &ranked, color);

        if interactive {
            terminal.draw(&output)?;
        } else {
            let mut stdout = io::stdout().lock();
            stdout.write_all(output.as_bytes())?;
            if config.record {
                stdout.write_all(b"\n\x0c\n")?;
            } else if limit.is_some_and(|maximum| maximum > 1) {
                stdout.write_all(b"\n\n")?;
            }
            stdout.flush()?;
        }

        if limit.is_some_and(|maximum| tick >= maximum) {
            break;
        }
        if interactive {
            if wait_for_exit(config.interval)? {
                break;
            }
        } else if !config.interval.is_zero() {
            thread::sleep(config.interval);
        }
    }

    terminal.leave()?;
    Ok(())
}

fn demo_nodes() -> Vec<NodeSpec> {
    vec![
        NodeSpec::field("price_a", "price_a", Transform::Identity),
        NodeSpec::field("price_b", "price_b", Transform::Identity),
        NodeSpec::field("spread_z", "spread_z", Transform::Identity),
        NodeSpec::field("opportunity_rank", "abs_spread_z", Transform::RankZScore),
        NodeSpec::field("half_life", "half_life", Transform::Identity),
    ]
}

fn build_pairs(pair_count: usize) -> Vec<PairState> {
    (0..pair_count)
        .map(|index| {
            let hedge_ratio = 0.85 + (index % 7) as f64 * 0.05;
            let phi = 0.84 + (index % 5) as f64 * 0.025;
            let innovation_sigma = 0.0035 + (index % 4) as f64 * 0.0006;
            let price_b = 32.0 + index as f64 * 1.8;
            let target_a = 28.0 + index as f64 * 2.1;
            let log_b = price_b.ln();
            let alpha = target_a.ln() - hedge_ratio * log_b;
            PairState {
                id: format!("P{:02}", index + 1),
                leg_a: format!("SIM{:04}", index * 2 + 1),
                leg_b: format!("SIM{:04}", index * 2 + 2),
                alpha,
                hedge_ratio,
                phi,
                innovation_sigma,
                log_b,
                spread: 0.0,
                price_a: target_a,
                price_b,
                spread_z: 0.0,
                half_life: 0.5_f64.ln() / phi.ln(),
                history: vec![0.0],
            }
        })
        .collect()
}

fn update_pairs(pairs: &mut [PairState], rng: &mut XorShift64) {
    for pair in pairs {
        pair.log_b += 0.00015 + 0.012 * rng.normal();
        pair.spread = pair.phi * pair.spread + pair.innovation_sigma * rng.normal();
        pair.price_b = pair.log_b.exp();
        pair.price_a = (pair.alpha + pair.hedge_ratio * pair.log_b + pair.spread).exp();
        let stationary_sigma = pair.innovation_sigma / (1.0 - pair.phi * pair.phi).sqrt();
        pair.spread_z = pair.spread / stationary_sigma;
        pair.history.push(pair.spread_z);
        if pair.history.len() > 44 {
            pair.history.remove(0);
        }
    }
}

fn ranked_rows(snapshot: &Snapshot, pairs: &[PairState]) -> Vec<PairRow> {
    let mut rows = pairs
        .iter()
        .filter_map(|pair| {
            Some(PairRow {
                id: pair.id.clone(),
                legs: format!("{}/{}", pair.leg_a, pair.leg_b),
                price_a: snapshot.value("price_a", &pair.id)?,
                price_b: snapshot.value("price_b", &pair.id)?,
                hedge_ratio: pair.hedge_ratio,
                spread_z: snapshot.value("spread_z", &pair.id)?,
                half_life: snapshot.value("half_life", &pair.id)?,
                opportunity_rank: snapshot.value("opportunity_rank", &pair.id)?,
                history: pair.history.clone(),
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .spread_z
            .abs()
            .partial_cmp(&left.spread_z.abs())
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    rows
}

fn render(config: &Config, snapshot: &Snapshot, rows: &[PairRow], color: bool) -> String {
    let mut out = String::new();
    out.push_str(&styled(" HYPERCUBE: PAIRS ", &[BOLD, CYAN], color));
    out.push_str(&styled(
        "  LIVE COINTEGRATION MONITOR\n",
        &[BOLD, WHITE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " generation {:>5}  pairs {:>3}  cells {:>3}\n",
            snapshot.generation,
            snapshot.entity_count,
            snapshot.values.len()
        ),
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " yⱼ,t = log Aⱼ,t − αⱼ − βⱼ log Bⱼ,t    yⱼ,t = φⱼyⱼ,t₋₁ + σⱼηⱼ,t\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        " zⱼ = yⱼ/(σⱼ/√(1−φⱼ²))    half-lifeⱼ = log(1/2)/log(φⱼ)    rank = rank-z(|zⱼ|)\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        "────────────────────────────────────────────────────────────────────────────────────────────────\n",
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " LARGEST STANDARDIZED PAIR DISLOCATIONS\n",
        &[BOLD, AMBER],
        color,
    ));
    out.push_str(&styled(
        " PAIR  LEGS               A PX      B PX      β    SPREAD Z  HALF-LIFE  RANK    POSITION             Z PATH\n",
        &[MUTED],
        color,
    ));

    for row in rows.iter().take(config.top) {
        render_pair_row(&mut out, row, color);
    }
    out.push_str(&styled(
        "\n q quit  •  α, β, and φ are fixed  •  fees, borrow, and breaks are not modeled\n",
        &[DIM, MUTED],
        color,
    ));
    out
}

fn render_pair_row(out: &mut String, row: &PairRow, color: bool) {
    let (row_color, position) = if row.spread_z >= 0.0 {
        (RED, "SHORT A / LONG B")
    } else {
        (GREEN, "LONG A / SHORT B")
    };
    out.push_str(&format!(
        " {:<5} {:<17} {:>8.2}  {:>8.2}  {:>5.2}  {}    {:>6.2}   {:>+6.3}  {:<19}  {}\n",
        row.id,
        row.legs,
        row.price_a,
        row.price_b,
        row.hedge_ratio,
        styled(
            &format!("{:>+8.3}", row.spread_z),
            &[BOLD, row_color],
            color
        ),
        row.half_life,
        row.opportunity_rank,
        styled(position, &[BOLD, row_color], color),
        styled(&sparkline(&row.history, 18), &[row_color], color)
    ));
}

fn sparkline(values: &[f64], width: usize) -> String {
    let start = values.len().saturating_sub(width);
    let values = &values[start..];
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    values
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * (SPARKS.len() - 1) as f64).round() as usize;
            SPARKS[bucket.min(SPARKS.len() - 1)]
        })
        .collect()
}

#[derive(Debug)]
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

    fn normal(&mut self) -> f64 {
        let first = ((self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let second = (self.next() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        (-2.0 * first.ln()).sqrt() * (TAU * second).cos()
    }
}

fn styled(text: &str, codes: &[&str], enabled: bool) -> String {
    if !enabled {
        return text.to_owned();
    }
    format!("{}{}{}", codes.concat(), text, RESET)
}

fn wait_for_exit(timeout: Duration) -> Result<bool> {
    if !event::poll(timeout)? {
        return Ok(false);
    }
    loop {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press
                && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
            {
                return Ok(true);
            }
        }
        if !event::poll(Duration::ZERO)? {
            return Ok(false);
        }
    }
}

struct TerminalSession {
    active: bool,
}

impl TerminalSession {
    fn enter(active: bool) -> Result<Self> {
        if active {
            enable_raw_mode()?;
            execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        }
        Ok(Self { active })
    }

    fn draw(&mut self, frame: &str) -> Result<()> {
        execute!(io::stdout(), MoveTo(0, 0), Clear(ClearType::All))?;
        let mut stdout = io::stdout().lock();
        stdout.write_all(frame.as_bytes())?;
        stdout.flush()?;
        Ok(())
    }

    fn leave(&mut self) -> Result<()> {
        if self.active {
            execute!(io::stdout(), Show, LeaveAlternateScreen)?;
            disable_raw_mode()?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.active {
            let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
            let _ = disable_raw_mode();
        }
    }
}

fn parse_args() -> Result<Config> {
    let mut config = Config::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--pairs" => config.pairs = value(&mut arguments, "--pairs")?,
            "--top" => config.top = value(&mut arguments, "--top")?,
            "--interval-ms" => {
                config.interval = Duration::from_millis(value(&mut arguments, "--interval-ms")?)
            }
            "--ticks" => config.ticks = Some(value(&mut arguments, "--ticks")?),
            "--seed" => config.seed = value(&mut arguments, "--seed")?,
            "--record" => config.record = true,
            "--no-color" => config.no_color = true,
            "-h" | "--help" => {
                println!(
                    "pairs — live synthetic statistical-arbitrage monitor\n\n\
                     cargo run -p hypercube-engine --example pairs -- [options]\n\n\
                     --pairs N          simulated cointegrated pairs (default 24)\n\
                     --top N            displayed dislocations (default 10)\n\
                     --interval-ms N    refresh delay (default 180)\n\
                     --ticks N          stop after N generations\n\
                     --seed N           deterministic random seed\n\
                     --record           emit ANSI frames separated by form-feed\n\
                     --no-color         suppress ANSI color"
                );
                std::process::exit(0);
            }
            unknown => bail!("unknown argument {unknown}; use --help"),
        }
    }
    if !(2..=128).contains(&config.pairs) {
        bail!("--pairs must be between 2 and 128");
    }
    if config.top == 0 {
        bail!("--top must be positive");
    }
    if config.ticks == Some(0) {
        bail!("--ticks must be positive");
    }
    Ok(config)
}

fn value<T>(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    arguments
        .next()
        .ok_or_else(|| anyhow!("{name} requires a value"))?
        .parse()
        .map_err(|error| anyhow!("invalid value for {name}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_prices_reconstruct_the_declared_cointegrating_residual() {
        let mut pairs = build_pairs(12);
        update_pairs(&mut pairs, &mut XorShift64::new(42));
        for pair in pairs {
            let reconstructed =
                pair.price_a.ln() - pair.alpha - pair.hedge_ratio * pair.price_b.ln();
            assert!((reconstructed - pair.spread).abs() < 1e-12);
            assert!(pair.half_life.is_finite() && pair.half_life > 0.0);
            assert!(pair.spread_z.is_finite());
        }
    }
}
