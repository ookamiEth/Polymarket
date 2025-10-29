#!/usr/bin/env python3
"""
Residual Diagnostics for Black-Scholes Binary Option Pricing.

This script analyzes the systematic errors (residuals) in Black-Scholes pricing
to identify specific biases that can be corrected.

Key analyses:
1. Residuals vs Moneyness - Identify volatility smile bias
2. Residuals vs Time to Expiry - Identify time decay bias
3. Residuals vs Implied Volatility - Identify IV level bias
4. Residuals vs Hour of Day - Identify session effects
5. Residuals vs Realized Volatility - Identify IV staleness

Output:
- Diagnostic plots showing where BS systematically fails
- Statistical tests for bias significance
- Recommendations for model improvements
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Styling
plt.style.use("dark_background")
sns.set_palette("husl")


class ResidualDiagnostics:
    """Analyze Black-Scholes pricing residuals to identify systematic biases."""

    def __init__(self, results_file: Path, output_dir: Path):
        """
        Initialize diagnostics.

        Args:
            results_file: Path to backtest results parquet file
            output_dir: Directory for output figures and reports
        """
        self.results_file = results_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pl.DataFrame] = None
        self.residuals: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None
        self.outcomes: Optional[np.ndarray] = None

    def load_results(self, sample_fraction: float = 1.0) -> None:
        """
        Load and prepare backtest results.

        Args:
            sample_fraction: Fraction of data to sample (1.0 = all data)
        """
        logger.info(f"Loading results from {self.results_file}")

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        # Load with sampling if needed (for memory efficiency)
        if sample_fraction < 1.0:
            logger.info(f"Sampling {sample_fraction:.1%} of data for analysis")
            self.df = pl.read_parquet(self.results_file).sample(fraction=sample_fraction, seed=42)
        else:
            self.df = pl.read_parquet(self.results_file)

        # Filter to valid predictions
        self.df = self.df.filter(
            pl.col("price_mid").is_not_null() & pl.col("outcome").is_not_null() & (pl.col("time_remaining") > 0)
        )

        # Add derived columns
        self.df = self.df.with_columns(
            [
                # Residual: prediction - outcome (error)
                (pl.col("price_mid") - pl.col("outcome")).alias("residual"),
                # Moneyness: S/K
                (pl.col("S") / pl.col("K")).alias("moneyness"),
                # Log moneyness (symmetric)
                (pl.col("S") / pl.col("K")).log().alias("log_moneyness"),
                # Extract hour of day from timestamp (convert seconds to milliseconds first)
                (pl.col("timestamp") * 1000).cast(pl.Datetime(time_unit="ms")).dt.hour().alias("hour_of_day"),
                # Time to expiry in minutes
                (pl.col("time_remaining") / 60).alias("time_to_expiry_minutes"),
            ]
        )

        logger.info(f"Loaded {len(self.df):,} valid predictions")
        logger.info(f"Contracts: {self.df['contract_id'].n_unique():,}")

        # Get date range
        min_ts = self.df["timestamp"].min()
        max_ts = self.df["timestamp"].max()
        logger.info(f"Date range: {min_ts} to {max_ts}")

        # Extract key arrays for analysis
        self.residuals = self.df["residual"].to_numpy()
        self.predictions = self.df["price_mid"].to_numpy()
        self.outcomes = self.df["outcome"].to_numpy()

        # Summary statistics
        logger.info("\nResidual Statistics:")
        logger.info(f"  Mean: {self.residuals.mean():.6f}")
        logger.info(f"  Median: {np.median(self.residuals):.6f}")
        logger.info(f"  Std Dev: {self.residuals.std():.6f}")
        logger.info(f"  Min: {self.residuals.min():.6f}")
        logger.info(f"  Max: {self.residuals.max():.6f}")

    def plot_residuals_vs_moneyness(self) -> None:
        """Plot residuals vs moneyness to identify smile bias."""
        logger.info("Analyzing residuals vs moneyness...")

        assert self.df is not None, "Must call load_results() first"
        assert self.residuals is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        moneyness = self.df["moneyness"].to_numpy()
        log_moneyness = self.df["log_moneyness"].to_numpy()

        # 1. Scatter plot (sampled)
        ax = axes[0, 0]
        sample_size = min(50000, len(self.residuals))
        sample_idx = np.random.choice(len(self.residuals), size=sample_size, replace=False)

        ax.scatter(
            moneyness[sample_idx],
            self.residuals[sample_idx],
            alpha=0.05,
            s=1,
            c=self.residuals[sample_idx],
            cmap="RdYlGn_r",
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Moneyness (S/K)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual (Predicted - Actual)", fontsize=12, fontweight="bold")
        ax.set_title("Residuals vs Moneyness", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.95, 1.05)

        # 2. Binned mean residual
        ax = axes[0, 1]
        moneyness_bins = np.linspace(0.97, 1.03, 30)
        bin_indices = np.digitize(moneyness, moneyness_bins)
        bin_means = [
            self.residuals[bin_indices == i].mean()
            for i in range(1, len(moneyness_bins))
            if (bin_indices == i).sum() > 100
        ]
        bin_centers = [
            moneyness_bins[i - 1 : i + 1].mean()
            for i in range(1, len(moneyness_bins))
            if (bin_indices == i).sum() > 100
        ]

        ax.plot(bin_centers, bin_means, marker="o", linewidth=2, markersize=6)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.fill_between(bin_centers, 0, bin_means, alpha=0.3)
        ax.set_xlabel("Moneyness (S/K)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Residual", fontsize=12, fontweight="bold")
        ax.set_title("Mean Residual by Moneyness Bin", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add interpretation
        max_abs_bias = np.max(np.abs(bin_means))
        ax.text(
            0.05,
            0.95,
            f"Max Absolute Bias: {max_abs_bias:.4f}\nPattern: {'Smile' if max_abs_bias > 0.02 else 'Flat'}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.7},
        )

        # 3. Residuals vs log moneyness
        ax = axes[1, 0]
        ax.scatter(
            log_moneyness[sample_idx],
            self.residuals[sample_idx],
            alpha=0.05,
            s=1,
            c=self.residuals[sample_idx],
            cmap="RdYlGn_r",
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvline(0, color="yellow", linestyle="--", linewidth=1, alpha=0.5, label="ATM")
        ax.set_xlabel("Log Moneyness ln(S/K)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residuals vs Log Moneyness (Symmetric)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.03, 0.03)

        # 4. Distribution by moneyness regime
        ax = axes[1, 1]
        otm_mask = moneyness < 0.995
        atm_mask = (moneyness >= 0.995) & (moneyness <= 1.005)
        itm_mask = moneyness > 1.005

        data_to_plot = [
            self.residuals[otm_mask],
            self.residuals[atm_mask],
            self.residuals[itm_mask],
        ]
        labels = [
            f"OTM (<0.995)\nn={otm_mask.sum():,}",
            f"ATM (0.995-1.005)\nn={atm_mask.sum():,}",
            f"ITM (>1.005)\nn={itm_mask.sum():,}",
        ]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)

        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residual Distribution by Moneyness Regime", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add mean markers
        for i, data in enumerate(data_to_plot, 1):
            mean_val = data.mean()
            ax.plot(i, mean_val, "r*", markersize=15, label="Mean" if i == 1 else "")

        ax.legend(fontsize=10)

        plt.tight_layout()
        output_file = self.output_dir / "residuals_vs_moneyness.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

        # Statistical test for bias
        logger.info("\nMoneyness Bias Test:")
        logger.info(f"  OTM Mean Residual: {self.residuals[otm_mask].mean():.6f}")
        logger.info(f"  ATM Mean Residual: {self.residuals[atm_mask].mean():.6f}")
        logger.info(f"  ITM Mean Residual: {self.residuals[itm_mask].mean():.6f}")

        # t-test: OTM vs ITM
        result = stats.ttest_ind(self.residuals[otm_mask], self.residuals[itm_mask], equal_var=False)
        t_stat = float(result[0])  # type: ignore[arg-type]
        p_value = float(result[1])  # type: ignore[arg-type]
        logger.info(f"  t-test (OTM vs ITM): t={t_stat:.4f}, p={p_value:.6f}")
        if p_value < 0.001:
            logger.info("  ✓ Significant moneyness bias detected!")
        else:
            logger.info("  ✗ No significant moneyness bias")

    def plot_residuals_vs_time(self) -> None:
        """Plot residuals vs time to expiry to identify time decay bias."""
        logger.info("Analyzing residuals vs time to expiry...")

        assert self.df is not None, "Must call load_results() first"
        assert self.residuals is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        time_remaining = self.df["time_remaining"].to_numpy()
        time_minutes = self.df["time_to_expiry_minutes"].to_numpy()

        # 1. Scatter plot (sampled)
        ax = axes[0, 0]
        sample_size = min(50000, len(self.residuals))
        sample_idx = np.random.choice(len(self.residuals), size=sample_size, replace=False)

        ax.scatter(
            time_minutes[sample_idx],
            self.residuals[sample_idx],
            alpha=0.05,
            s=1,
            c=self.residuals[sample_idx],
            cmap="RdYlGn_r",
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Time to Expiry (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residuals vs Time to Expiry", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Binned mean residual
        ax = axes[0, 1]
        time_bins = np.array([0, 1, 3, 5, 10, 15])
        bin_indices = np.digitize(time_minutes, time_bins)
        bin_means = [
            self.residuals[bin_indices == i].mean() for i in range(1, len(time_bins)) if (bin_indices == i).sum() > 100
        ]
        bin_labels = [
            f"{time_bins[i - 1]:.0f}-{time_bins[i]:.0f}min"
            for i in range(1, len(time_bins))
            if (bin_indices == i).sum() > 100
        ]
        bin_counts = [(bin_indices == i).sum() for i in range(1, len(time_bins)) if (bin_indices == i).sum() > 100]

        x_pos = np.arange(len(bin_means))
        bars = ax.bar(x_pos, bin_means, alpha=0.7)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.set_ylabel("Mean Residual", fontsize=12, fontweight="bold")
        ax.set_title("Mean Residual by Time Bin", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Color bars by sign
        for bar, mean_val in zip(bars, bin_means):
            bar.set_color("red" if mean_val > 0 else "green")

        # Add sample counts
        for i, (_, count) in enumerate(zip(bin_means, bin_counts)):
            ax.text(
                i,
                0.001,
                f"n={count:,}",
                ha="center",
                fontsize=8,
                rotation=90,
                color="white",
            )

        # 3. Absolute residual vs time (identify uncertainty)
        ax = axes[1, 0]
        abs_residuals = np.abs(self.residuals)
        ax.scatter(
            time_minutes[sample_idx],
            abs_residuals[sample_idx],
            alpha=0.05,
            s=1,
            c=time_minutes[sample_idx],
            cmap="viridis",
        )
        ax.set_xlabel("Time to Expiry (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Absolute Residual", fontsize=12, fontweight="bold")
        ax.set_title("Absolute Error vs Time (Uncertainty)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 4. Rolling mean residual over time
        ax = axes[1, 1]
        # Sort by time and compute rolling mean
        sort_idx = np.argsort(time_remaining)
        window_size = len(self.residuals) // 100  # 100 windows
        rolling_means = []
        rolling_times = []

        for i in range(0, len(self.residuals) - window_size, window_size):
            idx_window = sort_idx[i : i + window_size]
            rolling_means.append(self.residuals[idx_window].mean())
            rolling_times.append(time_minutes[idx_window].mean())

        ax.plot(rolling_times, rolling_means, linewidth=2, alpha=0.8)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.fill_between(rolling_times, 0, rolling_means, alpha=0.3)
        ax.set_xlabel("Time to Expiry (minutes)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Rolling Mean Residual", fontsize=12, fontweight="bold")
        ax.set_title("Rolling Mean Residual (100 windows)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "residuals_vs_time.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

        # Statistical test
        logger.info("\nTime Decay Bias Test:")
        for i in range(1, len(time_bins)):
            if (bin_indices == i).sum() > 100:
                mean_res = self.residuals[bin_indices == i].mean()
                logger.info(f"  {bin_labels[i - 1]}: Mean Residual = {mean_res:.6f}")

    def plot_residuals_vs_volatility(self) -> None:
        """Plot residuals vs implied volatility to identify IV level bias."""
        logger.info("Analyzing residuals vs implied volatility...")

        assert self.df is not None, "Must call load_results() first"
        assert self.residuals is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        sigma = self.df["sigma_mid"].to_numpy()

        # 1. Scatter plot
        ax = axes[0, 0]
        sample_size = min(50000, len(self.residuals))
        sample_idx = np.random.choice(len(self.residuals), size=sample_size, replace=False)

        ax.scatter(
            sigma[sample_idx],
            self.residuals[sample_idx],
            alpha=0.05,
            s=1,
            c=self.residuals[sample_idx],
            cmap="RdYlGn_r",
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Implied Volatility (σ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residuals vs Implied Volatility", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Binned mean residual
        ax = axes[0, 1]
        vol_bins = np.array([0, 0.3, 0.5, 0.7, 1.0, 5.0])
        bin_indices = np.digitize(sigma, vol_bins)
        bin_means = [
            self.residuals[bin_indices == i].mean() for i in range(1, len(vol_bins)) if (bin_indices == i).sum() > 100
        ]
        bin_labels = [
            f"{vol_bins[i - 1]:.1f}-{vol_bins[i]:.1f}"
            for i in range(1, len(vol_bins))
            if (bin_indices == i).sum() > 100
        ]
        bin_counts = [(bin_indices == i).sum() for i in range(1, len(vol_bins)) if (bin_indices == i).sum() > 100]

        x_pos = np.arange(len(bin_means))
        bars = ax.bar(x_pos, bin_means, alpha=0.7)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.set_ylabel("Mean Residual", fontsize=12, fontweight="bold")
        ax.set_title("Mean Residual by Volatility Bin", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, mean_val in zip(bars, bin_means):
            bar.set_color("red" if mean_val > 0 else "green")

        for i, (_, count) in enumerate(zip(bin_means, bin_counts)):
            ax.text(i, 0.001, f"n={count:,}", ha="center", fontsize=8, rotation=90)

        # 3. Hexbin plot for density
        ax = axes[1, 0]
        hb = ax.hexbin(
            sigma[sample_idx],
            self.residuals[sample_idx],
            gridsize=50,
            cmap="viridis",
            mincnt=1,
        )
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Implied Volatility (σ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residual Density (Hexbin)", fontsize=14, fontweight="bold")
        plt.colorbar(hb, ax=ax, label="Count")

        # 4. Correlation test
        ax = axes[1, 1]
        correlation = np.corrcoef(sigma, self.residuals)[0, 1]

        # Scatter with trend line
        ax.scatter(sigma[sample_idx], self.residuals[sample_idx], alpha=0.05, s=1)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)

        # Add trend line
        z = np.polyfit(sigma[sample_idx], self.residuals[sample_idx], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(sigma.min(), sigma.max(), 100)
        ax.plot(x_trend, p(x_trend), "r-", linewidth=2, label="Linear Fit")

        ax.set_xlabel("Implied Volatility (σ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Residuals vs IV with Trend\nCorrelation: {correlation:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "residuals_vs_volatility.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

        logger.info(f"\nVolatility Correlation: {correlation:.6f}")

    def plot_residuals_vs_hour(self) -> None:
        """Plot residuals vs hour of day to identify session effects."""
        logger.info("Analyzing residuals vs hour of day...")

        assert self.df is not None, "Must call load_results() first"
        assert self.residuals is not None, "Must call load_results() first"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        hour = self.df["hour_of_day"].to_numpy()

        # 1. Mean residual by hour
        ax = axes[0, 0]
        hours = np.arange(24)
        hour_means = [self.residuals[hour == h].mean() if (hour == h).sum() > 0 else 0 for h in hours]
        hour_counts = [(hour == h).sum() for h in hours]

        bars = ax.bar(hours, hour_means, alpha=0.7)
        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Hour of Day (UTC)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Residual", fontsize=12, fontweight="bold")
        ax.set_title("Mean Residual by Hour of Day", fontsize=14, fontweight="bold")
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, mean_val in zip(bars, hour_means):
            bar.set_color("red" if mean_val > 0 else "green")

        # 2. Boxplot by hour
        ax = axes[0, 1]
        data_by_hour = [self.residuals[hour == h] for h in hours if (hour == h).sum() > 100]
        valid_hours = [h for h in hours if (hour == h).sum() > 100]

        bp = ax.boxplot(data_by_hour, positions=valid_hours, widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)

        ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Hour of Day (UTC)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual", fontsize=12, fontweight="bold")
        ax.set_title("Residual Distribution by Hour", fontsize=14, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Sample count by hour
        ax = axes[1, 0]
        ax.bar(hours, hour_counts, alpha=0.7, color="steelblue")
        ax.set_xlabel("Hour of Day (UTC)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sample Count", fontsize=12, fontweight="bold")
        ax.set_title("Data Distribution by Hour", fontsize=14, fontweight="bold")
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, axis="y")

        # Add session markers
        ax.axvspan(0, 8, alpha=0.1, color="blue", label="Asian Session")
        ax.axvspan(8, 16, alpha=0.1, color="green", label="European Session")
        ax.axvspan(16, 24, alpha=0.1, color="red", label="US Session")
        ax.legend(fontsize=10)

        # 4. Heatmap: hour vs residual bins
        ax = axes[1, 1]
        residual_bins = np.array([-1, -0.2, -0.1, 0, 0.1, 0.2, 1])
        heatmap_data = np.zeros((len(hours), len(residual_bins) - 1))

        for i, h in enumerate(hours):
            hour_residuals = self.residuals[hour == h]
            if len(hour_residuals) > 0:
                for j in range(len(residual_bins) - 1):
                    mask = (hour_residuals >= residual_bins[j]) & (hour_residuals < residual_bins[j + 1])
                    heatmap_data[i, j] = mask.sum() / len(hour_residuals)

        im = ax.imshow(heatmap_data.T, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xlabel("Hour of Day (UTC)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual Bin", fontsize=12, fontweight="bold")
        ax.set_title("Residual Distribution Heatmap", fontsize=14, fontweight="bold")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_yticks(range(len(residual_bins) - 1))
        ax.set_yticklabels([f"{residual_bins[i]:.1f}" for i in range(len(residual_bins) - 1)])
        plt.colorbar(im, ax=ax, label="Fraction of Data")

        plt.tight_layout()
        output_file = self.output_dir / "residuals_vs_hour.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_file}")
        plt.close()

        # Statistical test
        logger.info("\nHour-of-Day Bias Test:")
        asian = self.residuals[(hour >= 0) & (hour < 8)].mean()
        european = self.residuals[(hour >= 8) & (hour < 16)].mean()
        us = self.residuals[(hour >= 16) & (hour < 24)].mean()

        logger.info(f"  Asian Session (0-8 UTC): {asian:.6f}")
        logger.info(f"  European Session (8-16 UTC): {european:.6f}")
        logger.info(f"  US Session (16-24 UTC): {us:.6f}")

    def generate_summary_report(self) -> None:
        """Generate summary report of all diagnostics."""
        logger.info("Generating summary report...")

        assert self.df is not None and self.residuals is not None, "Must call load_results() first"

        # Overall statistics
        mean_residual = self.residuals.mean()
        median_residual = np.median(self.residuals)
        std_residual = self.residuals.std()
        mae = np.abs(self.residuals).mean()
        rmse = np.sqrt((self.residuals**2).mean())

        # Statistical test for zero mean
        result = stats.ttest_1samp(self.residuals, 0)
        t_stat = float(result[0])  # type: ignore[arg-type]
        p_value = float(result[1])  # type: ignore[arg-type]

        # Create summary figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            "Black-Scholes Residual Diagnostics Summary",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # 1. Overall statistics (text box)
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        stats_text = f"""
        OVERALL RESIDUAL STATISTICS

        Mean Residual: {mean_residual:.6f}  (Should be ~0 for unbiased model)
        Median Residual: {median_residual:.6f}
        Standard Deviation: {std_residual:.6f}
        Mean Absolute Error (MAE): {mae:.6f}
        Root Mean Square Error (RMSE): {rmse:.6f}

        t-test for zero mean: t={t_stat:.4f}, p={p_value:.6f}
        {"✓ Model is unbiased (p > 0.05)" if p_value > 0.05 else "✗ Model shows systematic bias (p < 0.05)"}

        Sample Size: {len(self.residuals):,} predictions
        Contracts: {self.df["contract_id"].n_unique():,}
        """

        ax.text(
            0.5,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            horizontalalignment="center",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
        )

        # 2. Residual histogram
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(self.residuals, bins=100, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="white", linestyle="--", linewidth=2, label="Zero")
        ax.axvline(mean_residual, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_residual:.4f}")
        ax.set_xlabel("Residual", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Q-Q plot (test normality)
        ax = fig.add_subplot(gs[1, 1])
        sample_size = min(10000, len(self.residuals))
        sample = np.random.choice(self.residuals, size=sample_size, replace=False)
        stats.probplot(sample, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normality Test)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 4. Cumulative residual
        ax = fig.add_subplot(gs[1, 2])
        sorted_residuals = np.sort(self.residuals)
        cumsum_residuals = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
        ax.plot(sorted_residuals, cumsum_residuals, linewidth=2)
        ax.axvline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Residual", fontsize=10, fontweight="bold")
        ax.set_ylabel("Cumulative Probability", fontsize=10, fontweight="bold")
        ax.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 5-7. Key findings
        ax = fig.add_subplot(gs[2, :])
        ax.axis("off")

        # Compute key biases
        moneyness = self.df["moneyness"].to_numpy()
        otm_bias = self.residuals[moneyness < 0.995].mean()
        atm_bias = self.residuals[(moneyness >= 0.995) & (moneyness <= 1.005)].mean()
        itm_bias = self.residuals[moneyness > 1.005].mean()

        time_minutes = self.df["time_to_expiry_minutes"].to_numpy()
        early_bias = self.residuals[time_minutes > 10].mean()
        late_bias = self.residuals[time_minutes <= 1].mean()

        findings = f"""
        KEY FINDINGS & RECOMMENDATIONS

        1. MONEYNESS BIAS:
           - OTM (<0.995): {otm_bias:+.6f}  {"⚠️ Underpricing" if otm_bias < -0.01 else "✓ OK"}
           - ATM (0.995-1.005): {atm_bias:+.6f}  {"⚠️ Bias detected" if abs(atm_bias) > 0.01 else "✓ OK"}
           - ITM (>1.005): {itm_bias:+.6f}  {"⚠️ Overpricing" if itm_bias > 0.01 else "✓ OK"}
           {"→ Recommendation: Apply volatility smile adjustment" if abs(otm_bias - itm_bias) > 0.02 else "→ No smile adjustment needed"}

        2. TIME DECAY BIAS:
           - Early (>10 min): {early_bias:+.6f}
           - Late (≤1 min): {late_bias:+.6f}
           {"→ Recommendation: Adjust near-expiry pricing" if abs(late_bias) > 0.02 else "→ Time decay modeling is adequate"}

        3. OVERALL BIAS:
           {"✓ Model is well-calibrated (mean bias < 1%)" if abs(mean_residual) < 0.01 else "⚠️ Systematic bias detected - consider recalibration"}
        """

        ax.text(
            0.5,
            0.5,
            findings,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            horizontalalignment="center",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
        )

        output_file = self.output_dir / "residual_diagnostics_summary.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved summary: {output_file}")
        plt.close()

    def run_full_diagnostics(self, sample_fraction: float = 1.0) -> None:
        """Run complete diagnostic suite."""
        logger.info("=" * 80)
        logger.info("RESIDUAL DIAGNOSTICS")
        logger.info("=" * 80)

        self.load_results(sample_fraction=sample_fraction)

        self.plot_residuals_vs_moneyness()
        self.plot_residuals_vs_time()
        self.plot_residuals_vs_volatility()
        self.plot_residuals_vs_hour()
        self.generate_summary_report()

        logger.info("=" * 80)
        logger.info("DIAGNOSTICS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run residual diagnostics on backtest results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet",
        help="Path to backtest results parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/lgierhake/Documents/ETH/BT/research/model/results/figures/diagnostics",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of data to sample (0.0-1.0, default 1.0 = all data)",
    )

    args = parser.parse_args()

    # Create diagnostics
    diagnostics = ResidualDiagnostics(results_file=Path(args.results_file), output_dir=Path(args.output_dir))

    # Run full suite
    diagnostics.run_full_diagnostics(sample_fraction=args.sample)


if __name__ == "__main__":
    main()
