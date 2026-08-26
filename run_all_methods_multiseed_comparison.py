import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


METHODS = {
    "Proposed": {
        "script": "coalition_new.py",
        "output_dir": "proposed",
    },
    "EUBA": {
        "script": "benchmark_1_new.py",
        "output_dir": "equal_b",
    },
    "CSLRA": {
        "script": "benchmark_2_new.py",
        "output_dir": "coalition_based_r",
    },
    "CLRA": {
        "script": "benchmark_4_new.py",
        "output_dir": "constant_r",
    },
    "MiLRA": {
        "script": "benchmark_5_new.py",
        "output_dir": "minimum_r",
    },
    "DBBA": {
        "script": "benchmark_6_new.py",
        "output_dir": "databased",
    },
}

DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def summarize_optimized(df):
    metric_cols = ["cost", "T", "Utility", "loss"]
    numeric_cols = [col for col in metric_cols if col in df.columns]
    summary = df.groupby("option", dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    sort_cols = [col for col in ["Utility_mean", "loss_mean", "cost_mean"] if col in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    return summary


def add_rankings(summary):
    ranked = summary.copy()
    if "loss_mean" in ranked.columns:
        ranked["loss_rank"] = ranked["loss_mean"].rank(method="min", ascending=True).astype(int)
    if "Utility_mean" in ranked.columns:
        ranked["Utility_rank"] = ranked["Utility_mean"].rank(method="min", ascending=True).astype(int)
    if "cost_mean" in ranked.columns:
        ranked["cost_rank"] = ranked["cost_mean"].rank(method="min", ascending=True).astype(int)
    rank_cols = [c for c in ["Utility_rank", "loss_rank", "cost_rank"] if c in ranked.columns]
    if rank_cols:
        ranked = ranked.sort_values(rank_cols)
    return ranked


def run_method_seed(method_name, method, seed, base_output_dir, optimized_csv, python_exe):
    seed_dir = base_output_dir / method["output_dir"] / f"seed_{seed}" / "result"
    seed_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COALITION_SEED"] = str(seed)
    env["COALITION_OUTPUT_DIR"] = str(seed_dir)
    env["COALITION_OPTIMIZED_CSV"] = str(optimized_csv)

    print(f"\n=== Running {method_name} seed {seed} ===")
    print(f"Script: {method['script']}")
    print(f"Output: {seed_dir}")
    subprocess.run([python_exe, method["script"]], check=True, env=env)


def normalize_optimized(df):
    if df.empty:
        return df
    if "cost" not in df.columns and "T" in df.columns:
        df["cost"] = df["T"]
    cols = ["seed", "option", "T", "cost", "Utility", "loss"]
    existing = [col for col in cols if col in df.columns]
    rest = [col for col in df.columns if col not in existing]
    return df[existing + rest]


def main():
    parser = argparse.ArgumentParser(description="Run all coalition benchmarks across seeds and compare optimal metrics.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--base-output-dir", default="results_new1/cifar10/multiseed_comparison")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-run", action="store_true", help="Only summarize the existing combined optimized CSV.")
    parser.add_argument("--resume", action="store_true", help="Keep existing optimized rows and skip completed method/seed runs.")
    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir)
    summary_dir = base_output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    optimized_csv = summary_dir / "optimized_result_by_seed.csv"

    if optimized_csv.exists() and not args.skip_run and not args.resume:
        optimized_csv.unlink()

    completed = set()
    if optimized_csv.exists():
        existing_df = pd.read_csv(optimized_csv)
        if {"seed", "option"}.issubset(existing_df.columns):
            completed = set(zip(existing_df["seed"].astype(int), existing_df["option"].astype(str)))

    if not args.skip_run:
        for method_name in args.methods:
            method = METHODS[method_name]
            for seed in args.seeds:
                if args.resume and (seed, method_name) in completed:
                    print(f"Skipping completed {method_name} seed {seed}")
                    continue
                run_method_seed(method_name, method, seed, base_output_dir, optimized_csv, args.python)

    if not optimized_csv.exists():
        raise FileNotFoundError(f"No optimized result file found: {optimized_csv}")

    optimized_df = normalize_optimized(pd.read_csv(optimized_csv))
    optimized_df.to_csv(summary_dir / "optimized_result_by_seed.csv", index=False)

    summary = summarize_optimized(optimized_df)
    summary.to_csv(summary_dir / "optimized_result_summary.csv", index=False)

    ranked = add_rankings(summary)
    ranked.to_csv(summary_dir / "optimal_metric_comparison.csv", index=False)

    print("\nOptimal metric comparison:")
    display_cols = [
        col for col in [
            "option",
            "Utility_mean", "Utility_std",
            "loss_mean", "loss_std",
            "cost_mean", "cost_std",
            "Utility_rank", "loss_rank", "cost_rank",
        ]
        if col in ranked.columns
    ]
    print(ranked[display_cols].to_string(index=False))
    print(f"\nSummary files written to: {summary_dir}")


if __name__ == "__main__":
    main()
