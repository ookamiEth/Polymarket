#!/usr/bin/env python3
"""
Calibration Analysis for Binary Option Pricing.

Analyzes how well our Black-Scholes binary pricing predicts actual outcomes.

Key metrics:
1. Calibration plot: Predicted probability vs actual win rate
2. Brier score: Mean squared error of probabilistic predictions
3. Log loss: Logarithmic loss function
4. Breakdown by: time remaining, moneyness, volatility

Output:
- Calibration plots (bid/ask/mid)
- Metrics CSV file
- Detailed analysis by regime
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
RESULTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")
OUTPUT_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/figures")
METRICS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/calibration_metrics.csv")


def load_results() -> pl.DataFrame:
    """Load pilot backtest results."""
    logger.info(f"Loading results from {RESULTS_FILE}")
    df = pl.read_parquet(RESULTS_FILE)

    # Filter to rows with valid prices
    df_valid = df.filter(
        pl.col("price_mid").is_not_null() & pl.col("outcome").is_not_null() & (pl.col("time_remaining") > 0)
    )

    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Valid predictions: {len(df_valid):,} ({len(df_valid) / len(df) * 100:.1f}%)")

    return df_valid


def calculate_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error for probabilistic predictions).

    Brier score = mean((prediction - outcome)²)

    Range: [0, 1] where 0 is perfect, 1 is worst
    """
    return float(np.mean((predictions - outcomes) ** 2))


def calculate_log_loss(predictions: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate log loss (cross-entropy loss).

    Log loss = -mean(outcome*log(p) + (1-outcome)*log(1-p))

    Range: [0, ∞] where 0 is perfect
    """
    # Clip predictions to avoid log(0)
    p = np.clip(predictions, eps, 1 - eps)
    return -float(np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def create_calibration_data(df: pl.DataFrame, price_col: str, n_bins: int = 10) -> dict:
    """
    Create calibration data for plotting.

    Bins predictions and calculates actual win rate in each bin.
    """
    # Convert to numpy for easy binning
    predictions = df[price_col].to_numpy()
    outcomes = df["outcome"].to_numpy()

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate actual win rate in each bin
    actual_rates = []
    counts = []

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])

        count = mask.sum()
        counts.append(count)

        if count > 0:
            actual_rate = outcomes[mask].mean()
            actual_rates.append(actual_rate)
        else:
            actual_rates.append(np.nan)

    return {
        "bin_centers": bin_centers,
        "actual_rates": np.array(actual_rates),
        "counts": np.array(counts),
        "predictions": predictions,
        "outcomes": outcomes,
    }


def plot_calibration(df: pl.DataFrame, output_dir: Path) -> None:
    """Create calibration plots for bid, ask, and mid prices."""
    logger.info("Creating calibration plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (price_col, title) in enumerate(
        [("price_bid", "Bid IV"), ("price_ask", "Ask IV"), ("price_mid", "Mid IV")]
    ):
        ax = axes[idx]

        # Get calibration data
        cal_data = create_calibration_data(df, price_col, n_bins=10)

        # Plot
        ax.scatter(
            cal_data["bin_centers"],
            cal_data["actual_rates"],
            s=cal_data["counts"] / 100,  # Size by sample count
            alpha=0.6,
            label="Actual win rate",
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect calibration")

        # Calculate metrics
        brier = calculate_brier_score(cal_data["predictions"], cal_data["outcomes"])
        log_loss = calculate_log_loss(cal_data["predictions"], cal_data["outcomes"])

        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Actual Win Rate", fontsize=12)
        ax.set_title(f"{title}\nBrier: {brier:.4f}, Log Loss: {log_loss:.4f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    output_path = output_dir / "calibration_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved calibration plot to {output_path}")
    plt.close()


def plot_calibration_by_time(df: pl.DataFrame, output_dir: Path) -> None:
    """Analyze calibration by time remaining."""
    logger.info("Analyzing calibration by time remaining...")

    # Define time buckets
    time_buckets = [
        (0, 60, "<1min"),
        (60, 300, "1-5min"),
        (300, 600, "5-10min"),
        (600, 900, "10-15min"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (min_t, max_t, label) in enumerate(time_buckets):
        ax = axes[idx]

        # Filter to time bucket
        df_bucket = df.filter((pl.col("time_remaining") >= min_t) & (pl.col("time_remaining") < max_t))

        if len(df_bucket) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"Time Remaining: {label}")
            continue

        # Calibration data
        cal_data = create_calibration_data(df_bucket, "price_mid", n_bins=10)

        ax.scatter(cal_data["bin_centers"], cal_data["actual_rates"], s=cal_data["counts"] / 50, alpha=0.6)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)

        brier = calculate_brier_score(cal_data["predictions"], cal_data["outcomes"])
        n_samples = len(df_bucket)

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title(f"Time: {label}\nN={n_samples:,}, Brier={brier:.4f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    output_path = output_dir / "calibration_by_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved time-based calibration to {output_path}")
    plt.close()


def plot_calibration_by_moneyness(df: pl.DataFrame, output_dir: Path) -> None:
    """Analyze calibration by moneyness (S/K)."""
    logger.info("Analyzing calibration by moneyness...")

    # Calculate moneyness
    df = df.with_columns([(pl.col("S") / pl.col("K")).alias("moneyness")])

    # Define moneyness buckets
    moneyness_buckets = [
        (0, 0.995, "OTM (S/K < 0.995)"),
        (0.995, 1.005, "ATM (0.995-1.005)"),
        (1.005, 2, "ITM (S/K > 1.005)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (min_m, max_m, label) in enumerate(moneyness_buckets):
        ax = axes[idx]

        # Filter to moneyness bucket
        df_bucket = df.filter((pl.col("moneyness") >= min_m) & (pl.col("moneyness") < max_m))

        if len(df_bucket) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label)
            continue

        # Calibration data
        cal_data = create_calibration_data(df_bucket, "price_mid", n_bins=10)

        ax.scatter(cal_data["bin_centers"], cal_data["actual_rates"], s=cal_data["counts"] / 50, alpha=0.6)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5)

        brier = calculate_brier_score(cal_data["predictions"], cal_data["outcomes"])
        n_samples = len(df_bucket)

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title(f"{label}\nN={n_samples:,}, Brier={brier:.4f}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    output_path = output_dir / "calibration_by_moneyness.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved moneyness-based calibration to {output_path}")
    plt.close()


def calculate_all_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate comprehensive metrics table."""
    logger.info("Calculating comprehensive metrics...")

    metrics = []

    # Overall metrics
    for price_col, label in [("price_bid", "bid"), ("price_ask", "ask"), ("price_mid", "mid")]:
        valid = df.filter(pl.col(price_col).is_not_null())
        if len(valid) == 0:
            continue

        predictions = valid[price_col].to_numpy()
        outcomes = valid["outcome"].to_numpy()

        brier = calculate_brier_score(predictions, outcomes)
        log_loss_val = calculate_log_loss(predictions, outcomes)
        mean_pred = float(predictions.mean())
        mean_outcome = float(outcomes.mean())

        metrics.append(
            {
                "category": "overall",
                "subset": label,
                "n_samples": len(valid),
                "brier_score": brier,
                "log_loss": log_loss_val,
                "mean_prediction": mean_pred,
                "mean_outcome": mean_outcome,
                "prediction_bias": mean_pred - mean_outcome,
            }
        )

    metrics_df = pl.DataFrame(metrics)
    return metrics_df


def main() -> None:
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    df = load_results()

    # Create plots
    plot_calibration(df, OUTPUT_DIR)
    plot_calibration_by_time(df, OUTPUT_DIR)
    plot_calibration_by_moneyness(df, OUTPUT_DIR)

    # Calculate metrics
    metrics = calculate_all_metrics(df)

    # Save metrics
    metrics.write_csv(METRICS_FILE)
    logger.info(f"\nSaved metrics to {METRICS_FILE}")

    # Display metrics
    logger.info("\nCALIBRATION METRICS:")
    print(metrics)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
