"""Analyze multi-seed dose-response annotations for statistical significance.

Loads annotation parquets for all (condition, seed) pairs, computes per-seed
unsafe rates (overall and per category), and reports mean, std, 95% CI, min,
max across seeds. Also runs within-seed bootstrap for comparison.

Usage:
    # Print summary table for C1 and C2
    python analyze_multi_seed.py

    # Save bar chart with error bars
    python analyze_multi_seed.py --save-plot

    # Specify conditions
    python analyze_multi_seed.py --conditions C1 C2

    # Custom seeds
    python analyze_multi_seed.py --seeds 42 137 314 789 1331
"""

import argparse
import json
import os
import sys

from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONDITIONS = ["C1", "C2"]
DEFAULT_SEEDS = [42, 137, 314, 789, 1331]

CATEGORIES = [
    "O1: Hate, Humiliation, Harassment",
    "O2: Violence, Harm, or Cruelty",
    "O3: Sexual Content",
    "O4: Nudity Content",
    "O5: Criminal Planning",
    "O6: Weapons or Substance Abuse",
    "O7: Self-Harm",
    "O8: Animal Cruelty",
    "O9: Disasters or Emergencies",
]


def load_annotations(annotations_dir: str, condition: str, seed: int) -> Optional[pd.DataFrame]:
    """Load annotation parquet for a (condition, seed) pair.

    Checks two layouts:
    1. Single parquet file: annotations_dir/dose_{condition}.parquet
    2. Directory with parquets: annotations_dir/dose_{condition}/*.parquet
    """
    if seed == 42:
        model_id = f"dose_{condition}"
    else:
        model_id = f"dose_{condition}_seed_{seed}"

    # Try single parquet file first
    single_parquet = os.path.join(annotations_dir, f"{model_id}.parquet")
    if os.path.isfile(single_parquet):
        df = pd.read_parquet(single_parquet)
        df["seed"] = seed
        df["condition"] = condition
        return df

    # Fall back to directory with parquets
    parquet_dir = os.path.join(annotations_dir, model_id)
    if not os.path.isdir(parquet_dir):
        return None

    parquet_files = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        return None

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    df["seed"] = seed
    df["condition"] = condition
    return df


def compute_unsafe_rate(df: pd.DataFrame) -> dict:
    """Compute overall and per-category unsafe rates from an annotation DataFrame."""
    total = len(df)
    if total == 0:
        return {"overall": 0.0}

    overall = (df["rating"] == "Unsafe").mean()
    result = {"overall": overall}

    unsafe_df = df[df["rating"] == "Unsafe"]
    for cat in CATEGORIES:
        cat_count = (unsafe_df["category"] == cat).sum()
        result[cat] = cat_count / total

    return result


def bootstrap_unsafe_rate(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Bootstrap confidence interval for unsafe rate within a single seed."""
    rng = np.random.RandomState(seed)
    n = len(df)
    is_unsafe = (df["rating"] == "Unsafe").values

    rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        rates[i] = is_unsafe[idx].mean()

    return {
        "mean": rates.mean(),
        "std": rates.std(),
        "ci_low": np.percentile(rates, 2.5),
        "ci_high": np.percentile(rates, 97.5),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-seed dose-response annotations"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help=f"Conditions to analyze (default: {DEFAULT_CONDITIONS})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Seeds to include (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a bar chart with error bars",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: annotations_output dir from config)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for within-seed CI",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=["generation", "training"],
        help="Variance source: 'generation' (diffusion noise) or 'training' (training seed)",
    )
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    annotations_dir = config["evaluation"]["output_dir"]
    output_dir = args.output_dir or annotations_dir
    mode = args.mode
    suffix = f"_{mode}" if mode == "training" else ""

    print(f"Mode: {mode}-seed variance")
    print(f"Seed label: {'training' if mode == 'training' else 'generation'} seed")

    # Load all annotations
    all_rates = []
    seed_dfs = {}

    for condition in args.conditions:
        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"{'='*70}")

        condition_rates = []
        for seed in args.seeds:
            df = load_annotations(annotations_dir, condition, seed)
            if df is None:
                print(f"  seed={seed}: NOT FOUND")
                continue

            rates = compute_unsafe_rate(df)
            rates["seed"] = seed
            rates["condition"] = condition
            rates["n_images"] = len(df)
            condition_rates.append(rates)
            seed_dfs[(condition, seed)] = df

            print(f"  seed={seed}: {rates['overall']*100:.2f}% unsafe ({len(df)} images)")

        if not condition_rates:
            print("  No data found for any seed.")
            continue

        rates_df = pd.DataFrame(condition_rates)
        all_rates.extend(condition_rates)

        # Summary statistics across seeds
        n_seeds = len(rates_df)
        mean_rate = rates_df["overall"].mean()
        std_rate = rates_df["overall"].std(ddof=1) if n_seeds > 1 else 0.0
        se_rate = std_rate / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
        ci_half = 1.96 * se_rate

        print(f"\n  Between-seed statistics (n={n_seeds} seeds):")
        print(f"    Mean:   {mean_rate*100:.2f}%")
        print(f"    Std:    {std_rate*100:.2f}%")
        print(f"    95% CI: [{(mean_rate - ci_half)*100:.2f}%, {(mean_rate + ci_half)*100:.2f}%]")
        print(f"    Min:    {rates_df['overall'].min()*100:.2f}%")
        print(f"    Max:    {rates_df['overall'].max()*100:.2f}%")

        # Per-category breakdown
        print(f"\n  Per-category unsafe rates (mean +/- std across seeds):")
        for cat in CATEGORIES:
            if cat in rates_df.columns:
                cat_mean = rates_df[cat].mean()
                cat_std = rates_df[cat].std(ddof=1) if n_seeds > 1 else 0.0
                if cat_mean > 0 or cat_std > 0:
                    print(f"    {cat}: {cat_mean*100:.2f}% +/- {cat_std*100:.2f}%")

        # Within-seed bootstrap for comparison
        print(f"\n  Within-seed bootstrap (for comparison):")
        for seed in args.seeds:
            key = (condition, seed)
            if key not in seed_dfs:
                continue
            bs = bootstrap_unsafe_rate(
                seed_dfs[key], n_bootstrap=args.n_bootstrap, seed=seed
            )
            print(
                f"    seed={seed}: {bs['mean']*100:.2f}% "
                f"[{bs['ci_low']*100:.2f}%, {bs['ci_high']*100:.2f}%]"
            )

    if not all_rates:
        print("\nNo annotations found for any condition/seed pair.")
        return

    # Save summary CSV
    summary_df = pd.DataFrame(all_rates)
    csv_path = os.path.join(output_dir, f"multi_seed{suffix}_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # Optional bar chart
    if args.save_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))

            conditions_with_data = []
            means = []
            stds = []

            for condition in args.conditions:
                cond_df = summary_df[summary_df["condition"] == condition]
                if cond_df.empty:
                    continue
                conditions_with_data.append(condition)
                means.append(cond_df["overall"].mean() * 100)
                n = len(cond_df)
                std = cond_df["overall"].std(ddof=1) * 100 if n > 1 else 0.0
                stds.append(std)

            x = np.arange(len(conditions_with_data))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

            ax.set_xlabel("Condition")
            ax.set_ylabel("Unsafe Rate (%)")
            ax.set_title(f"Unsafe Rate by Condition (mean +/- std across {mode} seeds)")
            ax.set_xticks(x)
            ax.set_xticklabels(conditions_with_data)

            # Add individual seed points
            for i, condition in enumerate(conditions_with_data):
                cond_df = summary_df[summary_df["condition"] == condition]
                ax.scatter(
                    [i] * len(cond_df),
                    cond_df["overall"].values * 100,
                    color="darkred",
                    zorder=5,
                    s=30,
                    alpha=0.7,
                )

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"multi_seed{suffix}_error_bars.pdf")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
