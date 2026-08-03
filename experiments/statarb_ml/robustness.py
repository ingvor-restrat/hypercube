#!/usr/bin/env python3
"""Repeat the controlled study across deterministically derived random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from study import LANE_ORDER, StudyConfig, aggregate_metrics, run_study

MASTER_SEED = 335_341


def derived_seeds(count: int, master_seed: int = MASTER_SEED) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    if count == 1:
        return [master_seed]
    children = np.random.SeedSequence(master_seed).spawn(count - 1)
    return [master_seed, *[int(child.generate_state(1)[0]) for child in children]]


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = (
        runs.loc[runs.lane == "rolling_z", ["seed", "scenario", "trade_tstat", "total_net"]]
        .rename(
            columns={
                "trade_tstat": "baseline_trade_tstat",
                "total_net": "baseline_total_net",
            }
        )
        .set_index(["seed", "scenario"])
    )
    compared = runs.join(baseline, on=["seed", "scenario"])
    compared["trade_tstat_delta"] = compared.trade_tstat - compared.baseline_trade_tstat
    compared["total_net_delta"] = compared.total_net - compared.baseline_total_net
    compared["beats_baseline_tstat"] = compared.trade_tstat_delta > 0.0
    records = []
    for (scenario, lane), frame in compared.groupby(["scenario", "lane"], sort=False):
        records.append(
            {
                "scenario": scenario,
                "lane": lane,
                "runs": int(len(frame)),
                "mean_trade_tstat": float(frame.trade_tstat.mean()),
                "std_trade_tstat": float(frame.trade_tstat.std(ddof=1)),
                "mean_trade_tstat_delta": float(frame.trade_tstat_delta.mean()),
                "std_trade_tstat_delta": float(frame.trade_tstat_delta.std(ddof=1)),
                "median_trade_tstat_delta": float(frame.trade_tstat_delta.median()),
                "tstat_wins": int(frame.beats_baseline_tstat.sum()),
                "mean_total_net": float(frame.total_net.mean()),
                "mean_total_net_delta": float(frame.total_net_delta.mean()),
            }
        )
    summary = pd.DataFrame.from_records(records)
    return compared, summary.sort_values(["scenario", "lane"]).reset_index(drop=True)


def save_plot(compared: pd.DataFrame, output_path: Path) -> None:
    lanes = [lane for lane in LANE_ORDER if lane != "rolling_z"]
    colors = dict(zip(LANE_ORDER, plt.cm.tab10.colors))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, scenario in zip(axes, ["stationary_ou", "regime_switching"]):
        selected = compared.loc[compared.scenario == scenario]
        for position, lane in enumerate(lanes):
            values = selected.loc[selected.lane == lane, "trade_tstat_delta"].to_numpy()
            jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                color=colors[lane],
                alpha=0.85,
                s=42,
            )
            axis.plot(
                [position - 0.22, position + 0.22],
                [float(np.mean(values)), float(np.mean(values))],
                color="black",
                linewidth=2.0,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(
            np.arange(len(lanes)),
            [lane.replace("_", "\n") for lane in lanes],
            fontsize=8,
        )
        axis.set_ylabel("trade-return t-statistic minus rolling-z baseline")
        axis.set_title(scenario.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Hypercube stat-arb robustness — each dot is an independent deterministic seed",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for robustness CSV, JSON, and PNG artifacts",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--quick", action="store_true", help="use the smoke configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = derived_seeds(args.count, args.master_seed)
    aggregate_runs = []
    versions = None
    for position, seed in enumerate(seeds, start=1):
        print(f"run {position}/{len(seeds)}: seed={seed}", flush=True)
        config = StudyConfig.quick(seed) if args.quick else StudyConfig(seed=seed)
        metrics, _, trades, manifest = run_study(config)
        aggregate = aggregate_metrics(metrics, trades)
        aggregate.insert(0, "seed", seed)
        aggregate_runs.append(aggregate)
        versions = manifest["versions"]

    args.output.mkdir(parents=True, exist_ok=True)
    runs = pd.concat(aggregate_runs, ignore_index=True)
    compared, summary = summarize(runs)
    compared.to_csv(args.output / "robustness_runs.csv", index=False)
    summary.to_csv(args.output / "robustness_summary.csv", index=False)
    save_plot(compared, args.output / "statarb_ml_robustness.png")
    with (args.output / "robustness_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "master_seed": args.master_seed,
                "seed_derivation": "NumPy SeedSequence.spawn; master seed is also run zero",
                "seeds": seeds,
                "quick": args.quick,
                "versions": versions,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    columns = [
        "scenario",
        "lane",
        "mean_trade_tstat_delta",
        "tstat_wins",
        "mean_total_net_delta",
    ]
    print("\n" + summary.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nArtifacts: {args.output}")


if __name__ == "__main__":
    main()
