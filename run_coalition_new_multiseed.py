import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def summarize_numeric(df, group_cols):
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in group_cols and col != "seed"
    ]
    if not numeric_cols:
        return pd.DataFrame()

    summary = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def read_seed_csv(seed_dirs, relative_path):
    frames = []
    for seed, seed_dir in seed_dirs.items():
        csv_path = seed_dir / relative_path
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "seed", seed)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_summary(df, output_path, group_cols):
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.with_name(output_path.stem + "_by_seed.csv"), index=False)
    summary = summarize_numeric(df, group_cols)
    if not summary.empty:
        summary.to_csv(output_path, index=False)


def run_seed(seed, base_output_dir, optimized_csv, python_exe):
    seed_dir = base_output_dir / f"seed_{seed}" / "result"
    seed_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["COALITION_SEED"] = str(seed)
    env["COALITION_OUTPUT_DIR"] = str(seed_dir)
    env["COALITION_OPTIMIZED_CSV"] = str(optimized_csv)

    print(f"\n=== Running seed {seed} ===")
    print(f"Output: {seed_dir}")
    subprocess.run([python_exe, "coalition_new.py"], check=True, env=env)
    return seed_dir


def main():
    parser = argparse.ArgumentParser(description="Run coalition_new.py for multiple partition seeds and summarize results.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--base-output-dir", default="results_new1/cifar10/proposed/multiseed")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-run", action="store_true", help="Only summarize existing seed outputs.")
    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir)
    summary_dir = base_output_dir / "summary"
    optimized_csv = summary_dir / "optimized_result_by_seed.csv"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if optimized_csv.exists() and not args.skip_run:
        optimized_csv.unlink()

    seed_dirs = {}
    for seed in args.seeds:
        seed_dirs[seed] = base_output_dir / f"seed_{seed}" / "result"
        if not args.skip_run:
            seed_dirs[seed] = run_seed(seed, base_output_dir, optimized_csv, args.python)

    optimized_df = pd.read_csv(optimized_csv) if optimized_csv.exists() else pd.DataFrame()
    if not optimized_df.empty:
        optimized_df.to_csv(summary_dir / "optimized_result_by_seed.csv", index=False)
        optimized_summary = summarize_numeric(optimized_df, ["option"])
        optimized_summary.to_csv(summary_dir / "optimized_result_summary.csv", index=False)
        print("\nOptimized result summary:")
        print(optimized_summary.to_string(index=False))

    write_summary(
        read_seed_csv(seed_dirs, "average_loss_vs_T.csv"),
        summary_dir / "average_loss_vs_T_summary.csv",
        ["T_val"],
    )
    write_summary(
        read_seed_csv(seed_dirs, "utility_vs_T.csv"),
        summary_dir / "utility_vs_T_summary.csv",
        ["T_vals"],
    )
    write_summary(
        read_seed_csv(seed_dirs, "optimal_info_Proposed.csv"),
        summary_dir / "optimal_info_Proposed_summary.csv",
        ["Cluster"],
    )
    write_summary(
        read_seed_csv(seed_dirs, "tval_info_Proposed.csv"),
        summary_dir / "tval_info_Proposed_summary.csv",
        ["T_value", "Cluster"],
    )

    for cluster_id in range(10):
        write_summary(
            read_seed_csv(seed_dirs, f"cluster{cluster_id}_gammaB_Loss.csv"),
            summary_dir / f"cluster{cluster_id}_gammaB_Loss_summary.csv",
            ["gammaB"],
        )
        write_summary(
            read_seed_csv(seed_dirs, f"cluster{cluster_id}_gammaB_M_prime.csv"),
            summary_dir / f"cluster{cluster_id}_gammaB_M_prime_summary.csv",
            ["gammaB"],
        )

    print(f"\nSummary files written to: {summary_dir}")


if __name__ == "__main__":
    main()
