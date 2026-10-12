"""
Compare Round Robin, PPO, and CVaR-PPO for 5G RAN Slicing.

Inputs:
    round_robin_evaluation_results.csv
    ppo_evaluation_results.csv
    cvar_ppo_evaluation_results.csv

Outputs:
    final_method_comparison.csv
    comparison_mean_delay.png
    comparison_cvar95.png
    comparison_constraint_satisfaction.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

CVAR_THRESHOLD = 18.0

FILES = {
    "Round Robin": ROOT / "round_robin_evaluation_results.csv",
    "PPO": ROOT / "ppo_evaluation_results.csv",
    "CVaR-PPO": ROOT / "cvar_ppo_evaluation_results.csv",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, possible_names):
    """
    Return the first matching column name.
    Column names differ slightly between evaluation scripts.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    return None


def summarize_method(method_name, file_path):
    """
    Read one method's episode results and calculate summary metrics.
    """

    print(f"\nReading {method_name} results...", flush=True)

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nMissing file: {file_path}\n"
            f"Run the evaluation script for {method_name} first."
        )

    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} episodes from {file_path.name}")

    # Identify metric columns
    delay_col = find_column(
        df,
        ["mean_delay"]
    )

    cvar_col = find_column(
        df,
        ["episode_cvar95", "episode_cvar_95"]
    )

    utilization_col = find_column(
        df,
        ["mean_utilization"]
    )

    fairness_col = find_column(
        df,
        ["mean_fairness"]
    )

    # Validate required columns
    required = {
        "mean_delay": delay_col,
        "episode CVaR95": cvar_col,
        "mean_utilization": utilization_col,
        "mean_fairness": fairness_col,
    }

    missing = [
        name for name, column in required.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            f"\nMissing metrics in {file_path.name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Convert metrics to numeric values
    delays = pd.to_numeric(df[delay_col], errors="coerce")
    cvars = pd.to_numeric(df[cvar_col], errors="coerce")
    utilizations = pd.to_numeric(
        df[utilization_col], errors="coerce"
    )
    fairness = pd.to_numeric(
        df[fairness_col], errors="coerce"
    )

    # Remove invalid CVaR entries for satisfaction calculation
    valid_cvar = cvars.dropna()

    if len(valid_cvar) == 0:
        raise ValueError(
            f"No valid episode CVaR95 values in {file_path.name}"
        )

    # Episode-level constraint satisfaction
    satisfied = valid_cvar <= CVAR_THRESHOLD

    summary = {
        "Method": method_name,

        "Mean Delay": delays.mean(),
        "Delay Std": delays.std(ddof=0),

        "Mean CVaR95": cvars.mean(),
        "CVaR95 Std": cvars.std(ddof=0),

        "Mean Utilization": utilizations.mean(),
        "Mean Fairness": fairness.mean(),

        "CVaR95 Satisfaction (%)": (
            satisfied.mean() * 100
        ),

        "Episodes Evaluated": len(df),
        "Episodes Satisfying CVaR95": int(satisfied.sum()),
    }

    return summary


# =========================================================
# MAIN COMPARISON
# =========================================================

def main():

    print("\n" + "=" * 70)
    print("5G RAN SLICING - METHOD COMPARISON")
    print("=" * 70)

    summaries = []

    # Load and summarize all methods
    for method_name, file_path in FILES.items():

        summary = summarize_method(
            method_name,
            file_path
        )

        summaries.append(summary)

    # Create comparison DataFrame
    comparison = pd.DataFrame(summaries)

    # Save complete comparison
    output_csv = ROOT / "final_method_comparison.csv"

    comparison.to_csv(
        output_csv,
        index=False
    )

    # Display summary table
    display_columns = [
        "Method",
        "Mean Delay",
        "Delay Std",
        "Mean CVaR95",
        "CVaR95 Std",
        "Mean Utilization",
        "Mean Fairness",
        "CVaR95 Satisfaction (%)",
    ]

    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nEpisode constraint satisfaction:")

    for _, row in comparison.iterrows():

        print(
            f"{row['Method']}: "
            f"{int(row['Episodes Satisfying CVaR95'])}/"
            f"{int(row['Episodes Evaluated'])} episodes "
            f"({row['CVaR95 Satisfaction (%)']:.1f}%)"
        )

    print(f"\nSaved comparison CSV:\n{output_csv}")

    # =====================================================
    # GRAPH 1: MEAN DELAY
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Method"],
        comparison["Mean Delay"],
        yerr=comparison["Delay Std"],
        capsize=5
    )

    plt.ylabel("Mean Delay")
    plt.xlabel("Method")
    plt.title("Mean Delay Comparison")

    plt.tight_layout()

    graph1 = ROOT / "comparison_mean_delay.png"

    plt.savefig(
        graph1,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # GRAPH 2: EPISODE CVaR95
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Method"],
        comparison["Mean CVaR95"],
        yerr=comparison["CVaR95 Std"],
        capsize=5
    )

    plt.axhline(
        y=CVAR_THRESHOLD,
        linestyle="--",
        label=f"CVaR95 Threshold = {CVAR_THRESHOLD}"
    )

    plt.ylabel("Episode CVaR95")
    plt.xlabel("Method")
    plt.title("Tail-Risk Comparison")

    plt.legend()
    plt.tight_layout()

    graph2 = ROOT / "comparison_cvar95.png"

    plt.savefig(
        graph2,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # GRAPH 3: CVaR95 CONSTRAINT SATISFACTION
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Method"],
        comparison["CVaR95 Satisfaction (%)"]
    )

    plt.ylabel("Episodes Satisfying CVaR95 (%)")
    plt.xlabel("Method")
    plt.ylim(0, 105)

    plt.title("CVaR95 Constraint Satisfaction")

    plt.tight_layout()

    graph3 = ROOT / "comparison_constraint_satisfaction.png"

    plt.savefig(
        graph3,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # FINISHED
    # =====================================================

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print(f"1. {output_csv.name}")
    print(f"2. {graph1.name}")
    print(f"3. {graph2.name}")
    print(f"4. {graph3.name}")

    print("\nAll files are saved in:")
    print(ROOT)


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":
    main()