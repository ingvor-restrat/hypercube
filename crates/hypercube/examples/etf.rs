//! `etf` is a top-like ETF arbitrage monitor backed by the public Hypercube API.
//!
//! Correlated constituent returns produce basket fair values through
//! entity-axis dot products. Mean-reverting ETF premiums create synthetic
//! creation/redemption dislocations, and Hypercube evaluates the declared
//! ETF-level graph for every generation.

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
use hypercube::synthetic::OuMarketInjector;
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
    entities: usize,
    funds: usize,
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
            entities: 160,
            funds: 12,
            top: 5,
            interval: Duration::from_millis(180),
            ticks: None,
            seed: 0x51ced,
            record: false,
            no_color: false,
        }
    }
}

#[derive(Debug)]
struct EtfState {
    ticker: String,
    weights: Vec<f64>,
    fair_value: f64,
    market_price: f64,
    premium: f64,
    premium_z: f64,
    history: Vec<f64>,
}

#[derive(Debug)]
struct EtfRow {
    ticker: String,
    fair_value: f64,
    market_price: f64,
    premium_bps: f64,
    premium_z: f64,
    cross_sectional_z: f64,
    history: Vec<f64>,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let mut injector = OuMarketInjector::new(config.entities, config.seed);
    let mut premium_rng = XorShift64::new(config.seed ^ 0xa076_1d64_78bd_642f);
    let mut funds = build_funds(config.entities, config.funds);
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
        let frame = injector.next_frame(tick as i64 * 86_400_000);
        let returns = field_values(&frame.rows, "return")?;
        update_funds(&mut funds, &returns, &mut premium_rng);
        let rows = funds
            .iter()
            .map(|fund| {
                InputRow::new(fund.ticker.clone(), frame.observed_at_ms)
                    .with_field("fair_value", fund.fair_value)
                    .with_field("market_price", fund.market_price)
                    .with_field("premium_bps", fund.premium * 10_000.0)
                    .with_field("premium_z", fund.premium_z)
            })
            .collect::<Vec<_>>();
        let snapshot = engine.update(Update {
            generation: frame.generation,
            observed_at_ms: frame.observed_at_ms,
            mode: ExecutionMode::Live,
            rows,
            nodes: demo_nodes(),
        })?;
        let ranked = ranked_rows(&snapshot, &funds);
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
        NodeSpec::field("fair_value", "fair_value", Transform::Identity),
        NodeSpec::field("market_price", "market_price", Transform::Identity),
        NodeSpec::field("premium_bps", "premium_bps", Transform::Identity),
        NodeSpec::field("premium_z", "premium_z", Transform::Identity),
        NodeSpec::field("cross_sectional_z", "premium_bps", Transform::RankZScore),
    ]
}

fn build_funds(entity_count: usize, fund_count: usize) -> Vec<EtfState> {
    (0..fund_count)
        .map(|fund| {
            let weights = if fund == 0 {
                vec![1.0 / entity_count as f64; entity_count]
            } else if fund <= 4 {
                let sector = fund - 1;
                let members = (0..entity_count)
                    .filter(|entity| entity % 4 == sector)
                    .count();
                (0..entity_count)
                    .map(|entity| {
                        if entity % 4 == sector {
                            1.0 / members as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                let mut raw = (0..entity_count)
                    .map(|entity| {
                        let x = (entity + 1) as f64;
                        0.15 + (x * (fund as f64 + 0.71)).sin().abs()
                    })
                    .collect::<Vec<_>>();
                let gross = raw.iter().sum::<f64>();
                for weight in &mut raw {
                    *weight /= gross;
                }
                raw
            };
            EtfState {
                ticker: match fund {
                    0 => "ALL".to_owned(),
                    1..=4 => format!("SEC{}", fund - 1),
                    _ => format!("MIX{:02}", fund - 4),
                },
                weights,
                fair_value: 100.0,
                market_price: 100.0,
                premium: 0.0,
                premium_z: 0.0,
                history: vec![0.0],
            }
        })
        .collect()
}

fn update_funds(funds: &mut [EtfState], returns: &[f64], rng: &mut XorShift64) {
    const PREMIUM_PHI: f64 = 0.88;
    const INNOVATION_SIGMA: f64 = 0.000_40;
    let stationary_sigma = INNOVATION_SIGMA / (1.0 - PREMIUM_PHI * PREMIUM_PHI).sqrt();

    for fund in funds {
        let basket_return = fund
            .weights
            .iter()
            .zip(returns)
            .map(|(weight, value)| weight * value)
            .sum::<f64>();
        fund.fair_value *= 1.0 + basket_return;
        fund.premium = PREMIUM_PHI * fund.premium + INNOVATION_SIGMA * rng.normal();
        fund.market_price = fund.fair_value * (1.0 + fund.premium);
        fund.premium_z = fund.premium / stationary_sigma;
        fund.history.push(fund.premium * 10_000.0);
        if fund.history.len() > 44 {
            fund.history.remove(0);
        }
    }
}

fn ranked_rows(snapshot: &Snapshot, funds: &[EtfState]) -> Vec<EtfRow> {
    let mut rows = funds
        .iter()
        .filter_map(|fund| {
            Some(EtfRow {
                ticker: fund.ticker.clone(),
                fair_value: snapshot.value("fair_value", &fund.ticker)?,
                market_price: snapshot.value("market_price", &fund.ticker)?,
                premium_bps: snapshot.value("premium_bps", &fund.ticker)?,
                premium_z: snapshot.value("premium_z", &fund.ticker)?,
                cross_sectional_z: snapshot.value("cross_sectional_z", &fund.ticker)?,
                history: fund.history.clone(),
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .premium_z
            .partial_cmp(&left.premium_z)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.ticker.cmp(&right.ticker))
    });
    rows
}

fn render(config: &Config, snapshot: &Snapshot, rows: &[EtfRow], color: bool) -> String {
    let mut out = String::new();
    out.push_str(&styled(" HYPERCUBE: ETF ARBITRAGE ", &[BOLD, CYAN], color));
    out.push_str(&styled(
        "  SYNTHETIC CREATION / REDEMPTION MONITOR\n",
        &[BOLD, WHITE],
        color,
    ));
    out.push_str(&styled(
        &format!(
            " generation {:>5}  constituents {:>4}  ETFs {:>2}  cells {:>3}\n",
            snapshot.generation,
            config.entities,
            snapshot.entity_count,
            snapshot.values.len()
        ),
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " rⱼᴺᴬⱽ = Σᵢ wⱼᵢ rᵢ    Vⱼ,t = Vⱼ,t₋₁(1 + rⱼᴺᴬⱽ)    qⱼ = 10⁴(Pⱼ − Vⱼ)/Vⱼ\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        " premium process: uⱼ,t = 0.88 uⱼ,t₋₁ + 0.00040 ηⱼ,t    zⱼ = uⱼ/(0.00040/√(1−0.88²))\n",
        &[DIM, BLUE],
        color,
    ));
    out.push_str(&styled(
        "────────────────────────────────────────────────────────────────────────────────────────────\n",
        &[MUTED],
        color,
    ));
    out.push_str(&styled(
        " ETF CREATION / REDEMPTION DISLOCATIONS\n",
        &[BOLD, AMBER],
        color,
    ));
    out.push_str(&styled(
        " STATE  ETF       FAIR NAV     MKT PX    PREM bp   MODEL Z   XSEC Z   ETF/BASKET  PREMIUM PATH\n",
        &[MUTED],
        color,
    ));

    let rich = rows
        .iter()
        .filter(|row| row.premium_z >= 0.0)
        .take(config.top)
        .collect::<Vec<_>>();
    let cheap = rows
        .iter()
        .rev()
        .filter(|row| row.premium_z < 0.0)
        .take(config.top)
        .collect::<Vec<_>>();
    for row in rich {
        render_etf_row(&mut out, "RICH ", row, RED, color);
    }
    if !cheap.is_empty() {
        out.push_str(&styled(
            "        ···············································································\n",
            &[DIM, MUTED],
            color,
        ));
    }
    for row in cheap {
        render_etf_row(&mut out, "CHEAP", row, GREEN, color);
    }
    out.push_str(&styled(
        "\n q quit  •  no fees, borrow, or creation-unit frictions  •  educational example\n",
        &[DIM, MUTED],
        color,
    ));
    out
}

fn render_etf_row(out: &mut String, state: &str, row: &EtfRow, state_color: &str, color: bool) {
    let action = if row.premium_z >= 0.0 {
        "SELL/BUY"
    } else {
        "BUY/SELL"
    };
    out.push_str(&format!(
        " {}  {:<7}  {:>9.3}  {:>9.3}  {}  {}  {:>7.3}   {:<8}  {}\n",
        styled(state, &[BOLD, state_color], color),
        row.ticker,
        row.fair_value,
        row.market_price,
        styled(
            &format!("{:>+8.2}", row.premium_bps),
            &[BOLD, state_color],
            color
        ),
        styled(
            &format!("{:>+8.3}", row.premium_z),
            &[BOLD, state_color],
            color
        ),
        row.cross_sectional_z,
        action,
        styled(&sparkline(&row.history, 28), &[state_color], color)
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

fn field_values(rows: &[hypercube::InputRow], field: &str) -> Result<Vec<f64>> {
    rows.iter()
        .map(|row| {
            row.fields
                .get(field)
                .copied()
                .ok_or_else(|| anyhow!("{} is missing field {field}", row.key))
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
            "--entities" => config.entities = value(&mut arguments, "--entities")?,
            "--funds" => config.funds = value(&mut arguments, "--funds")?,
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
                    "etf — live synthetic ETF arbitrage monitor\n\n\
                     cargo run -p hypercube-engine --example etf -- [options]\n\n\
                     --entities N       basket constituents (default 160)\n\
                     --funds N          simulated ETFs (default 12)\n\
                     --top N            rows on each side (default 5)\n\
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
    if config.entities < 8 {
        bail!("--entities must be at least 8");
    }
    if !(2..=32).contains(&config.funds) {
        bail!("--funds must be between 2 and 32");
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
    fn every_synthetic_fund_is_long_only_and_fully_invested() {
        for fund in build_funds(160, 12) {
            assert!(fund.weights.iter().all(|weight| *weight >= 0.0));
            let weight_sum = fund.weights.iter().sum::<f64>();
            assert!((weight_sum - 1.0).abs() < 1e-12);
        }
    }
}
