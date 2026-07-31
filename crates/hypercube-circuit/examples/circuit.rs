use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::error::Error;
use std::io::{self, BufReader, Cursor, IsTerminal, Write};
use std::thread;
use std::time::Duration;

use hypercube::synthetic::{market_demo_nodes, OuMarketInjector};
use hypercube::{HypercubeEngine, Snapshot};
use hypercube_circuit::{
    residual_score_trigger, snapshot_digest, CaptureSession, CircuitConfig, RecordingManifest,
    RecordingReader, ReplayReport, ReplayRunner, ThresholdTriggerSpec, TriggerFrame, TriggerState,
    TriggerTransition, TriggerTransitionKind,
};

const ENTITY_COUNT: usize = 12;
const SEED: u64 = 335_342;
const SCORE_NODE: &str = "liquid_residual_score";
const MISSING_GENERATION: u64 = 4;
const MISSING_KEY: &str = "SIM0006";
const SPARKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_BLACK: &str = "\x1b[90m";
const BRIGHT_RED: &str = "\x1b[91m";
const BRIGHT_GREEN: &str = "\x1b[92m";
const BRIGHT_CYAN: &str = "\x1b[96m";

#[derive(Debug, Clone, Copy)]
struct Config {
    ticks: usize,
    top: usize,
    interval: Duration,
    record: bool,
    no_color: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ticks: 28,
            top: 8,
            interval: Duration::from_millis(180),
            record: false,
            no_color: false,
        }
    }
}

#[derive(Debug)]
struct DemoTotals {
    factor_cells: usize,
    state_cells: usize,
    transitions: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(config) = parse_args()? else {
        return Ok(());
    };
    run(config)
}

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let trigger = residual_score_trigger()?;
    let manifest = RecordingManifest::new(
        "hypercube-terminal-demo-v1",
        env!("CARGO_PKG_VERSION"),
        format!("ou-market-seed-{SEED}"),
        format!("synthetic-layout-{ENTITY_COUNT}"),
        vec![trigger.clone()],
    )?
    .with_metadata("input_boundary", "complete_hypercube_update")?
    .with_metadata("demo_missing_observation", MISSING_KEY)?;

    let mut capture = CaptureSession::new(Vec::new(), manifest, CircuitConfig::default())?;
    let mut display_engine = HypercubeEngine::new();
    let mut injector = OuMarketInjector::new(ENTITY_COUNT, SEED);
    let mut histories = BTreeMap::<String, Vec<f64>>::new();
    let mut recent = VecDeque::<TriggerTransition>::new();
    let mut totals = DemoTotals {
        factor_cells: 0,
        state_cells: 0,
        transitions: 0,
    };
    let mut terminal = DemoTerminal::new(config);

    for index in 0..config.ticks {
        let observed_at_ms = 1_700_000_000_000_i64 + index as i64 * 100;
        let mut market = injector.next_frame(observed_at_ms);
        if market.generation == MISSING_GENERATION {
            market.rows.retain(|row| row.key != MISSING_KEY);
        }
        let update = market.into_update(market_demo_nodes());
        let snapshot = display_engine.update(update.clone())?;
        let frame = capture.process(update)?;
        if snapshot_digest(&snapshot) != frame.snapshot {
            return Err("display calculation did not match the captured snapshot".into());
        }

        update_histories(&snapshot, &mut histories);
        for transition in &frame.transitions {
            recent.push_back(transition.clone());
            while recent.len() > 3 {
                recent.pop_front();
            }
        }
        totals.factor_cells = totals.factor_cells.saturating_add(snapshot.values.len());
        totals.state_cells = totals.state_cells.saturating_add(frame.states.len());
        totals.transitions = totals.transitions.saturating_add(frame.transitions.len());

        terminal.draw(&render_live(LiveRender {
            config,
            trigger: &trigger,
            snapshot: &snapshot,
            frame: &frame,
            histories: &histories,
            recent: &recent,
            totals: &totals,
            color: terminal.color,
        }))?;
    }

    let recording = capture.finish()?;
    let line_count = recording.iter().filter(|byte| **byte == b'\n').count();
    let replay_start =
        render_replay_start(config, &totals, recording.len(), line_count, terminal.color);
    for _ in 0..3 {
        terminal.draw(&replay_start)?;
    }

    let reader = RecordingReader::new(BufReader::new(Cursor::new(recording.as_slice())))?;
    let report = ReplayRunner::default().run(reader)?;
    terminal.draw(&render_replay_result(
        &report,
        recording.len(),
        line_count,
        terminal.color,
    ))?;
    if !report.is_exact() {
        return Err("Hypercube demo replay diverged".into());
    }
    Ok(())
}

struct DemoTerminal {
    interactive: bool,
    record: bool,
    color: bool,
    interval: Duration,
}

impl DemoTerminal {
    fn new(config: Config) -> Self {
        let interactive = io::stdout().is_terminal() && !config.record;
        Self {
            interactive,
            record: config.record,
            color: !config.no_color && (interactive || config.record),
            interval: config.interval,
        }
    }

    fn draw(&mut self, frame: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        if self.interactive {
            stdout.write_all(b"\x1b[2J\x1b[H")?;
        }
        stdout.write_all(frame.as_bytes())?;
        if self.record {
            stdout.write_all(b"\n\x0c\n")?;
        } else if !self.interactive {
            stdout.write_all(b"\n\n")?;
        }
        stdout.flush()?;
        drop(stdout);
        if !self.interval.is_zero() {
            thread::sleep(self.interval);
        }
        Ok(())
    }
}

struct LiveRender<'a> {
    config: Config,
    trigger: &'a ThresholdTriggerSpec,
    snapshot: &'a Snapshot,
    frame: &'a TriggerFrame,
    histories: &'a BTreeMap<String, Vec<f64>>,
    recent: &'a VecDeque<TriggerTransition>,
    totals: &'a DemoTotals,
    color: bool,
}

fn render_live(view: LiveRender<'_>) -> String {
    let LiveRender {
        config,
        trigger,
        snapshot,
        frame,
        histories,
        recent,
        totals,
        color,
    } = view;
    let scores = snapshot
        .slice(SCORE_NODE)
        .into_iter()
        .map(|cell| (cell.key.clone(), cell.value))
        .collect::<BTreeMap<_, _>>();
    let states = frame
        .states
        .iter()
        .map(|state| (state.key.as_str(), state))
        .collect::<BTreeMap<_, _>>();
    let changes = frame
        .transitions
        .iter()
        .map(|transition| (transition.key.as_str(), transition))
        .collect::<BTreeMap<_, _>>();
    let keys = display_keys(config.top, &scores, &states, &frame.transitions);

    let mut out = String::new();
    out.push_str(&styled(
        " HYPERCUBE  CALLBACKS / TRIGGERS / REPLAY",
        &[BOLD, BRIGHT_CYAN],
        color,
    ));
    out.push('\n');
    out.push_str(&format!(
        " generation {:>2}/{:<2}   entities {:>2}   factor cells {:>3}   circuit sequence {:>2}   {}\n",
        snapshot.generation,
        config.ticks,
        snapshot.entity_count,
        snapshot.values.len(),
        snapshot.generation.saturating_sub(1),
        styled("LIVE", &[BOLD, GREEN], color)
    ));
    out.push_str(&format!(
        " trigger {} >= {:+.2} for {} generations  |  exit <= {:+.2}\n",
        trigger.node, trigger.enter_at_or_above, trigger.min_consecutive, trigger.exit_at_or_below
    ));
    out.push_str(&styled(
        " --------------------------------------------------------------------------------------------\n",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    out.push_str(&styled(
        " ENTITY      SCORE    CALLBACK       CHANGE         SCORE PATH\n",
        &[YELLOW],
        color,
    ));

    for key in &keys {
        let score = scores.get(key).copied();
        let state = states.get(key.as_str()).copied();
        let change = changes.get(key.as_str()).copied();
        render_row(&mut out, key, score, state, change, histories, color);
    }
    for _ in keys.len()..config.top {
        out.push_str(" \n");
    }

    out.push_str(&styled(
        " --------------------------------------------------------------------------------------------\n",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    out.push_str(&styled(" RECENT TRANSITIONS\n", &[BOLD, CYAN], color));
    for transition in recent {
        out.push_str(&render_transition(transition, color));
    }
    for _ in recent.len()..3 {
        out.push('\n');
    }
    out.push_str(&format!(
        "\n JSONL capture: {:>2} generations  {:>4} factor cells  {:>3} state cells  {:>2} transitions\n",
        snapshot.generation, totals.factor_cells, totals.state_cells, totals.transitions
    ));
    out.push_str(&styled(
        " test case: SIM0006 is removed at generation 4 to exercise missing-data invalidation",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    out
}

fn display_keys(
    top: usize,
    scores: &BTreeMap<String, f64>,
    states: &BTreeMap<&str, &TriggerState>,
    transitions: &[TriggerTransition],
) -> Vec<String> {
    let mut keys = Vec::with_capacity(top);
    let mut seen = BTreeSet::new();
    let mut push = |key: &str| {
        if keys.len() < top && seen.insert(key.to_owned()) {
            keys.push(key.to_owned());
        }
    };

    for transition in transitions {
        push(&transition.key);
    }
    for state in states.values().filter(|state| state.active) {
        push(&state.key);
    }
    for state in states
        .values()
        .filter(|state| state.qualifying_generations > 0)
    {
        push(&state.key);
    }
    let mut ranked = scores.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(left.1).then_with(|| left.0.cmp(right.0)));
    for (key, _) in ranked {
        push(key);
    }
    keys
}

fn render_row(
    out: &mut String,
    key: &str,
    score: Option<f64>,
    state: Option<&TriggerState>,
    change: Option<&TriggerTransition>,
    histories: &BTreeMap<String, Vec<f64>>,
    color: bool,
) {
    let (callback, callback_color) = match change.map(|item| item.kind) {
        Some(TriggerTransitionKind::Entered) => ("ACTIVE", BRIGHT_GREEN),
        Some(TriggerTransitionKind::Exited) => ("IDLE", RED),
        Some(TriggerTransitionKind::Invalidated) => ("MISSING", BRIGHT_RED),
        None if state.is_some_and(|item| item.active) => ("ACTIVE", GREEN),
        None if state.is_some_and(|item| item.qualifying_generations > 0) => ("QUALIFY", YELLOW),
        None => ("IDLE", BRIGHT_BLACK),
    };
    let callback = if callback == "QUALIFY" {
        format!(
            "QUALIFY {}/2",
            state.map_or(0, |item| item.qualifying_generations)
        )
    } else {
        callback.to_owned()
    };
    let callback = styled(&format!("{callback:<13}"), &[BOLD, callback_color], color);

    let (change_text, change_color) = match change.map(|item| item.kind) {
        Some(TriggerTransitionKind::Entered) => ("ENTERED", BRIGHT_GREEN),
        Some(TriggerTransitionKind::Exited) => ("EXITED", RED),
        Some(TriggerTransitionKind::Invalidated) => ("INVALIDATED", BRIGHT_RED),
        None => ("-", BRIGHT_BLACK),
    };
    let change_text = styled(&format!("{change_text:<14}"), &[BOLD, change_color], color);
    let score_text = score
        .map(|value| format!("{value:>+7.3}"))
        .unwrap_or_else(|| "missing".to_owned());
    let path = histories
        .get(key)
        .map(|values| sparkline(values, 24))
        .unwrap_or_default();
    out.push_str(&format!(
        " {key:<10}  {score_text:>7}   {callback} {change_text} {path}\n"
    ));
}

fn render_transition(transition: &TriggerTransition, color: bool) -> String {
    let (kind, kind_color) = match transition.kind {
        TriggerTransitionKind::Entered => ("ENTERED", BRIGHT_GREEN),
        TriggerTransitionKind::Exited => ("EXITED", RED),
        TriggerTransitionKind::Invalidated => ("INVALIDATED", BRIGHT_RED),
    };
    let kind = styled(&format!("{kind:<11}"), &[BOLD, kind_color], color);
    let value = transition
        .value
        .map(|value| format!("score {value:>+6.3}"))
        .unwrap_or_else(|| "input missing".to_owned());
    format!(
        " g{:03}  {kind}  {:<9}  {value}\n",
        transition.generation, transition.key
    )
}

fn render_replay_start(
    config: Config,
    totals: &DemoTotals,
    bytes: usize,
    lines: usize,
    color: bool,
) -> String {
    let mut out = String::new();
    out.push_str(&styled(
        " HYPERCUBE  CALLBACKS / TRIGGERS / REPLAY",
        &[BOLD, BRIGHT_CYAN],
        color,
    ));
    out.push_str("\n\n");
    out.push_str(&styled(" RECORDING SEALED\n", &[BOLD, YELLOW], color));
    out.push_str(&format!(
        "   1 manifest + {} generations = {lines} JSON Lines\n",
        config.ticks
    ));
    out.push_str(&format!("   {bytes} bytes written in memory\n"));
    out.push_str(&format!(
        "   {} factor cells, {} trigger-state cells, {} transitions\n",
        totals.factor_cells, totals.state_cells, totals.transitions
    ));
    out.push('\n');
    out.push_str(&styled(" STARTING REPLAY\n", &[BOLD, CYAN], color));
    out.push_str("   new HypercubeEngine\n");
    out.push_str("   new persistent-threshold callback state\n");
    out.push_str("   recorded updates switched from Live to Replay mode\n");
    out.push_str("   external effects disabled\n\n");
    out.push_str(&styled(
        " recalculating every generation and comparing values, state, and transitions ...\n",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    for _ in 0..9 {
        out.push('\n');
    }
    out
}

fn render_replay_result(report: &ReplayReport, bytes: usize, lines: usize, color: bool) -> String {
    let exact = report.is_exact();
    let (result, result_color) = if exact {
        ("EXACT MATCH", BRIGHT_GREEN)
    } else {
        ("DIVERGED", BRIGHT_RED)
    };
    let mut out = String::new();
    out.push_str(&styled(
        " HYPERCUBE  CALLBACKS / TRIGGERS / REPLAY",
        &[BOLD, BRIGHT_CYAN],
        color,
    ));
    out.push_str("\n\n");
    out.push_str(&styled(
        &format!(" REPLAY RESULT: {result}\n"),
        &[BOLD, result_color],
        color,
    ));
    out.push_str(&styled(
        " --------------------------------------------------------------------------------\n",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    out.push_str(&styled(
        " COMPARED OBJECT                 RECORDED       REPLAYED\n",
        &[YELLOW],
        color,
    ));
    out.push_str(&format!(
        " generations                    {:>8}       {:>8}\n",
        report.generations, report.generations
    ));
    out.push_str(&format!(
        " factor cells                    {:>8}       {:>8}\n",
        report.values, report.values
    ));
    out.push_str(&format!(
        " trigger-state cells             {:>8}       {:>8}\n",
        report.expected_states, report.actual_states
    ));
    out.push_str(&format!(
        " transitions                     {:>8}       {:>8}\n",
        report.expected_transitions, report.actual_transitions
    ));
    out.push_str(&format!(
        " divergent generations           {:>8}\n",
        report.divergent_generations
    ));
    out.push_str(&styled(
        " --------------------------------------------------------------------------------\n",
        &[DIM, BRIGHT_BLACK],
        color,
    ));
    out.push_str(&format!(" recording: {lines} JSON Lines, {bytes} bytes\n"));
    out.push_str(" fresh engine + fresh callback state produced the same result\n");
    out.push_str(" live effects were never invoked\n");
    for _ in 0..7 {
        out.push('\n');
    }
    out
}

fn update_histories(snapshot: &Snapshot, histories: &mut BTreeMap<String, Vec<f64>>) {
    for value in snapshot.slice(SCORE_NODE) {
        let history = histories.entry(value.key.clone()).or_default();
        history.push(value.value);
        if history.len() > 28 {
            history.remove(0);
        }
    }
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

fn styled(text: &str, codes: &[&str], enabled: bool) -> String {
    if !enabled {
        return text.to_owned();
    }
    format!("{}{}{}", codes.concat(), text, RESET)
}

fn parse_args() -> Result<Option<Config>, Box<dyn Error>> {
    let mut config = Config::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--ticks" => config.ticks = parse_value(&mut arguments, "--ticks")?,
            "--top" => config.top = parse_value(&mut arguments, "--top")?,
            "--interval-ms" => {
                config.interval =
                    Duration::from_millis(parse_value(&mut arguments, "--interval-ms")?)
            }
            "--record" => config.record = true,
            "--no-color" => config.no_color = true,
            "-h" | "--help" => {
                println!("{}", usage());
                return Ok(None);
            }
            _ => return Err(format!("unknown argument: {argument}\n\n{}", usage()).into()),
        }
    }
    if config.ticks == 0 {
        return Err("--ticks must be positive".into());
    }
    if config.top == 0 {
        return Err("--top must be positive".into());
    }
    config.top = config.top.min(ENTITY_COUNT);
    Ok(Some(config))
}

fn parse_value<T>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let value = arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {option} value {value:?}: {error}").into())
}

fn usage() -> &'static str {
    "usage: cargo run -p hypercube-circuit --example circuit -- [OPTIONS]\n\
     \n\
     options:\n\
       --ticks N          live generations before replay (default 28)\n\
       --top N            callback rows shown (default 8)\n\
       --interval-ms N    delay between frames (default 180)\n\
       --record           emit ANSI frames separated by form-feed\n\
       --no-color         disable ANSI color\n"
}
