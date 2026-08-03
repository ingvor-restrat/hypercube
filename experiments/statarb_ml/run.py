#!/usr/bin/env python3
"""Run the controlled Hypercube statistical-arbitrage ML study."""

from __future__ import annotations

import argparse
from pathlib import Path

from study import StudyConfig, run_study, write_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for CSV, JSON, and PNG artifacts",
    )
    parser.add_argument("--seed", type=int, default=335_341)
    parser.add_argument("--quick", action="store_true", help="one small smoke fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StudyConfig.quick(args.seed) if args.quick else StudyConfig(seed=args.seed)
    metrics, diagnostics, trades, manifest = run_study(config)
    aggregate = write_results(args.output, metrics, diagnostics, trades, manifest)
    columns = [
        "scenario",
        "lane",
        "trades",
        "total_net",
        "mean_net",
        "hit_rate",
        "trade_tstat",
        "max_drawdown",
    ]
    print(
        aggregate.loc[:, columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"\nArtifacts: {args.output}")


if __name__ == "__main__":
    main()
