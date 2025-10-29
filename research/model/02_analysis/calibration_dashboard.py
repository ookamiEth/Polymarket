#!/usr/bin/env python3
"""
Comprehensive Calibration Dashboard for Binary Option Pricing.

Enhanced calibration analysis with:
1. Expected Calibration Error (ECE)
2. Maximum Calibration Error (MCE)
3. Regime-based breakdowns (moneyness, time, volatility)
4. Interactive visualizations
5. Detailed error analysis

This dashboard provides a complete picture of model calibration quality
across different market conditions.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Styling
plt.style.use("dark_background")
sns.set_palette("husl")


class CalibrationDashboard:
    """Comprehensive calibration analysis dashboard."""

    def __init__(self, results_file: Path, output_dir: Path):
        """
        Initialize dashboard.

        Args:
            results_file: Path to backtest results parquet file
            output_dir: Directory for output figures and metrics
        """
        self.results_file = results_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pl.DataFrame] = None
        self.metrics: dict = {}

    def load_results(self) -> None:
        """Load and validate backtest results."""
        logger.info(f"Loading results from {self.results_file}")

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        self.df = pl.read_parquet(self.results_file)

        # Filter to valid predictions
        self.df = self.df.filter(
            pl.col("price_mid").is_not_null() & pl.col("outcome").is_not_null() & (pl.col("time_remaining") > 0)
        )

        logger.info(f"Loaded {len(self.df):,} valid predictions")
        logger.info(f"Contracts: {self.df['contract_id'].n_unique():,}")
        logger.info(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

    def calculate_brier_score(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier score (mean squared error for probabilistic predictions).

        Brier score = mean((prediction - outcome)²)
        Range: [0, 1] where 0 is perfect, 1 is worst
        Baseline (random): 0.25
        """
        return float(np.mean((predictions - outcomes) ** 2))

    def calculate_log_loss(self, predictions: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
        """
        Calculate log loss (cross-entropy loss).

        Log loss = -mean(outcome*log(p) + (1-outcome)*log(1-p))
        Range: [0, ∞] where 0 is perfect
        """
        p = np.clip(predictions, eps, 1 - eps)
        return -float(np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

    def calculate_ece(
        self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = weighted mean of absolute calibration error per bin
        ECE = sum(n_i / n * |accuracy_i - confidence_i|)

        Returns:
            ece: Expected Calibration Error
            bin_accuracies: Actual win rate per bin
            bin_confidences: Mean predicted probability per bin
            bin_counts: Sample count per bin
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracies[i] = outcomes[mask].mean()
                bin_confidences[i] = predictions[mask].mean()
                bin_counts[i] = mask.sum()

        # Weighted calibration error
        total = len(predictions)
        ece = np.sum((bin_counts / total) * np.abs(bin_accuracies - bin_confidences))

        return ece, bin_accuracies, bin_confidences, bin_counts

    def calculate_mce(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).

        MCE = max(|accuracy_i - confidence_i|) across all bins
        """
        _, bin_accuracies, bin_confidences, bin_counts = self.calculate_ece(predictions, outcomes, n_bins)

        # Only consider bins with samples
        valid_bins = bin_counts > 0
        if not valid_bins.any():
            return 0.0

        calibration_errors = np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins])
        mce = float(np.max(calibration_errors))

        return mce

    def create_calibration_plot(
        self, predictions: np.ndarray, outcomes: np.ndarray, title: str, n_bins: int = 10
    ) -> tuple:
        """
        Create calibration plot with confidence intervals.

        Returns figure and metrics dictionary.
        """
        # Calculate metrics
        brier = self.calculate_brier_score(predictions, outcomes)
        log_loss = self.calculate_log_loss(predictions, outcomes)
        ece, bin_accuracies, bin_confidences, bin_counts = self.calculate_ece(predictions, outcomes, n_bins)
        mce = self.calculate_mce(predictions, outcomes, n_bins)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot calibration curve
        valid_bins = bin_counts > 0
        sizes = np.clip(bin_counts / 1000, 30, 200)  # Scale point sizes

        scatter = ax.scatter(
            bin_confidences[valid_bins],
            bin_accuracies[valid_bins],
            s=sizes[valid_bins],
            alpha=0.7,
            c=bin_counts[valid_bins],
            cmap="viridis",
            edgecolors="white",
            linewidth=1.5,
            label="Calibration bins",
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, linewidth=2, label="Perfect calibration")

        # Confidence bands (±5%)
        x = np.linspace(0, 1, 100)
        ax.fill_between(x, x - 0.05, x + 0.05, alpha=0.15, color="yellow", label="±5% band")

        # Styling
        ax.set_xlabel("Predicted Probability", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual Win Rate", fontsize=14, fontweight="bold")
        ax.set_title(
            f"{title}\nBrier: {brier:.4f} | ECE: {ece:.4f} | MCE: {mce:.4f} | Log Loss: {log_loss:.4f}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Color bar for sample counts
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sample Count", fontsize=12, fontweight="bold")

        # Add text box with interpretation
        interpretation = self._interpret_calibration(brier, ece, mce)
        ax.text(
            0.02,
            0.98,
            interpretation,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.7},
        )

        plt.tight_layout()

        metrics = {
            "brier_score": brier,
            "log_loss": log_loss,
            "ece": ece,
            "mce": mce,
            "n_samples": len(predictions),
        }

        return fig, metrics

    def _interpret_calibration(self, brier: float, ece: float, mce: float) -> str:
        """Generate interpretation text for calibration metrics."""
        lines = ["Calibration Quality:"]

        # Brier score interpretation
        if brier < 0.10:
            lines.append(f"✅ Excellent Brier ({brier:.3f})")
        elif brier < 0.15:
            lines.append(f"✓ Good Brier ({brier:.3f})")
        elif brier < 0.20:
            lines.append(f"⚠ Fair Brier ({brier:.3f})")
        else:
            lines.append(f"❌ Poor Brier ({brier:.3f})")

        # ECE interpretation
        if ece < 0.05:
            lines.append(f"✅ Excellent ECE ({ece:.3f})")
        elif ece < 0.10:
            lines.append(f"✓ Good ECE ({ece:.3f})")
        elif ece < 0.15:
            lines.append(f"⚠ Fair ECE ({ece:.3f})")
        else:
            lines.append(f"❌ Poor ECE ({ece:.3f})")

        # MCE interpretation
        if mce < 0.10:
            lines.append(f"✅ Excellent MCE ({mce:.3f})")
        elif mce < 0.20:
            lines.append(f"✓ Good MCE ({mce:.3f})")
        elif mce < 0.30:
            lines.append(f"⚠ Fair MCE ({mce:.3f})")
        else:
            lines.append(f"❌ Poor MCE ({mce:.3f})")

        return "\n".join(lines)

    def plot_error_distribution(self) -> None:
        """Plot distribution of prediction errors."""
        logger.info("Creating error distribution plots...")

        assert self.df is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        predictions = self.df["price_mid"].to_numpy()
        outcomes = self.df["outcome"].to_numpy()
        errors = predictions - outcomes

        # 1. Error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=50, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax.axvline(errors.mean(), color="yellow", linestyle="--", linewidth=2, label=f"Mean: {errors.mean():.4f}")
        ax.set_xlabel("Prediction Error (Predicted - Actual)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title("Error Distribution", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 2. Error vs predicted probability
        ax = axes[0, 1]
        sample_idx = np.random.choice(len(errors), size=min(50000, len(errors)), replace=False)
        ax.scatter(
            predictions[sample_idx],
            errors[sample_idx],
            alpha=0.1,
            s=1,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
        ax.set_ylabel("Prediction Error", fontsize=12, fontweight="bold")
        ax.set_title("Error vs Predicted Probability", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Absolute error boxplot by confidence level
        ax = axes[1, 0]
        bin_edges = [0, 0.3, 0.4, 0.6, 0.7, 1.0]
        bin_indices = np.digitize(predictions, bin_edges)
        abs_errors_by_conf = [np.abs(errors[bin_indices == i]) for i in range(1, len(bin_edges))]
        ax.boxplot(abs_errors_by_conf, labels=["0-30%", "30-40%", "40-60%", "60-70%", "70-100%"])
        ax.set_xlabel("Predicted Probability Range", fontsize=12, fontweight="bold")
        ax.set_ylabel("Absolute Error", fontsize=12, fontweight="bold")
        ax.set_title("Absolute Error by Confidence Level", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. Cumulative error distribution
        ax = axes[1, 1]
        sorted_abs_errors = np.sort(np.abs(errors))
        cumsum = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        ax.plot(sorted_abs_errors, cumsum, linewidth=2)
        ax.axvline(0.05, color="yellow", linestyle="--", alpha=0.7, label="±5% threshold")
        ax.axvline(0.10, color="orange", linestyle="--", alpha=0.7, label="±10% threshold")
        ax.set_xlabel("Absolute Error", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight="bold")
        ax.set_title("Cumulative Error Distribution", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "error_distribution.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved error distribution plot: {output_file}")
        plt.close()

    def plot_regime_breakdowns(self) -> None:
        """Plot calibration metrics across different regimes."""
        logger.info("Creating regime breakdown plots...")

        assert self.df is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Calibration by time remaining
        ax = axes[0, 0]
        time_bins = [(0, 60), (60, 300), (300, 600), (600, 900)]
        metrics_by_time = []

        for t_min, t_max in time_bins:
            df_time = self.df.filter((pl.col("time_remaining") >= t_min) & (pl.col("time_remaining") < t_max))
            if len(df_time) > 0:
                pred = df_time["price_mid"].to_numpy()
                out = df_time["outcome"].to_numpy()
                brier = self.calculate_brier_score(pred, out)
                ece, _, _, _ = self.calculate_ece(pred, out)
                metrics_by_time.append((f"{t_min}-{t_max}s", brier, ece, len(df_time)))

        if metrics_by_time:
            labels, briers, eces, counts = zip(*metrics_by_time)
            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, briers, width, label="Brier Score", alpha=0.7)
            ax.bar(x + width / 2, eces, width, label="ECE", alpha=0.7)
            ax.set_xlabel("Time Remaining", fontsize=12, fontweight="bold")
            ax.set_ylabel("Error", fontsize=12, fontweight="bold")
            ax.set_title("Calibration by Time Remaining", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            # Add sample counts as text
            for i, (_, count) in enumerate(zip(labels, counts)):
                ax.text(i, 0.01, f"n={count:,}", ha="center", fontsize=8, rotation=90)

        # 2. Calibration by moneyness
        ax = axes[0, 1]
        df_with_moneyness = self.df.with_columns([(pl.col("S") / pl.col("K")).alias("moneyness")])
        moneyness_bins = [(0, 0.99), (0.99, 0.995), (0.995, 1.005), (1.005, 1.01), (1.01, 2.0)]
        metrics_by_moneyness = []

        for m_min, m_max in moneyness_bins:
            df_mon = df_with_moneyness.filter((pl.col("moneyness") >= m_min) & (pl.col("moneyness") < m_max))
            if len(df_mon) > 0:
                pred = df_mon["price_mid"].to_numpy()
                out = df_mon["outcome"].to_numpy()
                brier = self.calculate_brier_score(pred, out)
                ece, _, _, _ = self.calculate_ece(pred, out)
                metrics_by_moneyness.append((f"{m_min:.3f}-{m_max:.3f}", brier, ece, len(df_mon)))

        if metrics_by_moneyness:
            labels, briers, eces, counts = zip(*metrics_by_moneyness)
            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, briers, width, label="Brier Score", alpha=0.7)
            ax.bar(x + width / 2, eces, width, label="ECE", alpha=0.7)
            ax.set_xlabel("Moneyness (S/K)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Error", fontsize=12, fontweight="bold")
            ax.set_title("Calibration by Moneyness", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            for i, (_, count) in enumerate(zip(labels, counts)):
                ax.text(i, 0.01, f"n={count:,}", ha="center", fontsize=8, rotation=90)

        # 3. Calibration by volatility
        ax = axes[1, 0]
        vol_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 5.0)]
        metrics_by_vol = []

        for v_min, v_max in vol_bins:
            df_vol = self.df.filter((pl.col("sigma_mid") >= v_min) & (pl.col("sigma_mid") < v_max))
            if len(df_vol) > 0:
                pred = df_vol["price_mid"].to_numpy()
                out = df_vol["outcome"].to_numpy()
                brier = self.calculate_brier_score(pred, out)
                ece, _, _, _ = self.calculate_ece(pred, out)
                metrics_by_vol.append((f"{v_min:.1f}-{v_max:.1f}", brier, ece, len(df_vol)))

        if metrics_by_vol:
            labels, briers, eces, counts = zip(*metrics_by_vol)
            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, briers, width, label="Brier Score", alpha=0.7)
            ax.bar(x + width / 2, eces, width, label="ECE", alpha=0.7)
            ax.set_xlabel("Implied Volatility (σ)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Error", fontsize=12, fontweight="bold")
            ax.set_title("Calibration by Volatility", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            for i, (_, count) in enumerate(zip(labels, counts)):
                ax.text(i, 0.01, f"n={count:,}", ha="center", fontsize=8, rotation=90)

        # 4. Win rate by confidence level
        ax = axes[1, 1]
        confidence_bins = [(0, 0.4), (0.4, 0.45), (0.45, 0.55), (0.55, 0.6), (0.6, 1.0)]
        metrics_by_conf = []

        for c_min, c_max in confidence_bins:
            df_conf = self.df.filter((pl.col("price_mid") >= c_min) & (pl.col("price_mid") < c_max))
            if len(df_conf) > 0:
                win_rate = df_conf["outcome"].mean()
                mean_pred = df_conf["price_mid"].mean()
                count = len(df_conf)
                metrics_by_conf.append((f"{c_min:.0%}-{c_max:.0%}", mean_pred, win_rate, count))

        if metrics_by_conf:
            labels, mean_preds, win_rates, counts = zip(*metrics_by_conf)
            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width / 2, mean_preds, width, label="Mean Predicted", alpha=0.7)
            ax.bar(x + width / 2, win_rates, width, label="Actual Win Rate", alpha=0.7)
            ax.set_xlabel("Confidence Range", fontsize=12, fontweight="bold")
            ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
            ax.set_title("Win Rate by Confidence Level", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 1)

            for i, (_, count) in enumerate(zip(labels, counts)):
                ax.text(i, 0.05, f"n={count:,}", ha="center", fontsize=8, rotation=90)

        plt.tight_layout()
        output_file = self.output_dir / "regime_breakdowns.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved regime breakdowns plot: {output_file}")
        plt.close()

    def generate_dashboard(self) -> None:
        """Generate complete calibration dashboard."""
        logger.info("=" * 80)
        logger.info("GENERATING CALIBRATION DASHBOARD")
        logger.info("=" * 80)

        # Load results
        self.load_results()

        assert self.df is not None, "Failed to load results"

        # Overall calibration plot
        logger.info("Creating overall calibration plot...")
        predictions = self.df["price_mid"].to_numpy()
        outcomes = self.df["outcome"].to_numpy()

        fig, metrics = self.create_calibration_plot(predictions, outcomes, "Overall Calibration (Mid IV)", n_bins=10)
        output_file = self.output_dir / "calibration_overall.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved calibration plot: {output_file}")
        plt.close()

        # Store metrics
        self.metrics["overall"] = metrics

        # Error distribution
        self.plot_error_distribution()

        # Regime breakdowns
        self.plot_regime_breakdowns()

        # Save metrics to CSV
        self._save_metrics()

        logger.info("=" * 80)
        logger.info("DASHBOARD GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")

    def _save_metrics(self) -> None:
        """Save metrics to CSV file."""
        metrics_file = self.output_dir / "calibration_metrics.csv"

        metrics_df = pl.DataFrame(
            {
                "metric": list(self.metrics["overall"].keys()),
                "value": [float(v) for v in self.metrics["overall"].values()],
            }
        )

        metrics_df.write_csv(metrics_file)
        logger.info(f"Saved metrics: {metrics_file}")

        # Print summary
        logger.info("\nCALIBRATION METRICS:")
        for metric, value in self.metrics["overall"].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value:,}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate calibration dashboard")
    parser.add_argument(
        "--results-file",
        type=str,
        default="/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet",
        help="Path to backtest results parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/lgierhake/Documents/ETH/BT/research/model/results/figures/calibration",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Create dashboard
    dashboard = CalibrationDashboard(results_file=Path(args.results_file), output_dir=Path(args.output_dir))

    # Generate
    dashboard.generate_dashboard()


if __name__ == "__main__":
    main()
