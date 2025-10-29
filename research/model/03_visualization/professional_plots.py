#!/usr/bin/env python3
"""
Professional Quant-Grade Visualizations for Binary Option Pricing Model

This module provides publication-quality visualizations with:
1. Dark theme for professional trading desk look
2. Confidence intervals and statistical annotations
3. Interactive plots using plotly
4. Comprehensive error analysis
5. Time series evolution plots
6. PnL analysis visualizations

All plots follow quantitative finance best practices for clarity and information density.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import polars as pl
import seaborn as sns
from scipy import stats
from sklearn.metrics import brier_score_loss

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set professional dark theme
plt.style.use('dark_background')
mplstyle.use('fast')

# Professional color palette
COLORS = {
    'primary': '#00D4FF',      # Cyan
    'secondary': '#FF00FF',    # Magenta
    'success': '#00FF88',      # Green
    'danger': '#FF3366',       # Red
    'warning': '#FFB000',      # Orange
    'info': '#00B4FF',         # Light Blue
    'perfect': '#888888',      # Gray for reference lines
    'grid': '#333333',         # Dark gray for grid
}

# Font settings for professional look
FONT_TITLE = {'family': 'Arial', 'size': 16, 'weight': 'bold'}
FONT_LABEL = {'family': 'Arial', 'size': 12}
FONT_TICK = {'family': 'Arial', 'size': 10}

# File paths
RESULTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")
OUTPUT_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/figures/professional")


class ProfessionalVisualizer:
    """Professional visualization generator for binary option pricing analysis."""

    def __init__(self, results_file: Path = RESULTS_FILE, output_dir: Path = OUTPUT_DIR):
        """Initialize visualizer with data and output paths."""
        self.results_file = results_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and prepare data for visualization."""
        logger.info(f"Loading data from {self.results_file}")
        self.df = pl.read_parquet(self.results_file)

        # Filter to valid predictions
        self.df = self.df.filter(
            pl.col("price_mid").is_not_null()
            & pl.col("outcome").is_not_null()
            & (pl.col("time_remaining") > 0)
        )

        # Add moneyness column
        self.df = self.df.with_columns([(pl.col("S") / pl.col("K")).alias("moneyness")])

        logger.info(f"Loaded {len(self.df):,} valid predictions")

    def create_enhanced_calibration_plot(self) -> None:
        """
        Create professional calibration plot with confidence intervals and statistics.
        """
        logger.info("Creating enhanced calibration plot...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#0a0a0a')
        fig.suptitle('Binary Option Pricing Calibration Analysis',
                     fontsize=20, fontweight='bold', color=COLORS['primary'])

        for idx, (price_col, title, color) in enumerate([
            ("price_bid", "Bid IV Pricing", COLORS['danger']),
            ("price_ask", "Ask IV Pricing", COLORS['warning']),
            ("price_mid", "Mid IV Pricing", COLORS['success'])
        ]):
            ax = axes[idx]
            ax.set_facecolor('#111111')

            # Get calibration data with bootstrap confidence intervals
            cal_data = self._calculate_calibration_with_ci(price_col, n_bins=15, n_bootstrap=1000)

            # Plot main calibration points
            # Scale dot size: min 30, max 300 based on sample count
            sizes = np.clip(cal_data['counts'] / 1000, 30, 300)
            ax.scatter(
                cal_data['bin_centers'],
                cal_data['actual_rates'],
                s=sizes,
                c=color,
                alpha=0.8,
                edgecolors='white',
                linewidth=1,
                label='Empirical',
                zorder=5
            )

            # Add confidence intervals
            ax.fill_between(
                cal_data['bin_centers'],
                cal_data['ci_lower'],
                cal_data['ci_upper'],
                alpha=0.2,
                color=color,
                label='95% CI'
            )

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], '--', color=COLORS['perfect'],
                    linewidth=2, alpha=0.6, label='Perfect')

            # Add regression line
            mask = ~np.isnan(cal_data['actual_rates'])
            if mask.sum() > 1:
                z = np.polyfit(cal_data['bin_centers'][mask],
                              cal_data['actual_rates'][mask], 1,
                              w=np.sqrt(cal_data['counts'][mask]))
                p = np.poly1d(z)
                x_reg = np.linspace(0, 1, 100)
                ax.plot(x_reg, p(x_reg), '-', color=COLORS['info'],
                       linewidth=1.5, alpha=0.8, label=f'Fit: {z[0]:.2f}x + {z[1]:.3f}')

            # Calculate metrics
            predictions = self.df[price_col].to_numpy()
            outcomes = self.df["outcome"].to_numpy()
            brier = brier_score_loss(outcomes, predictions)
            ece = self._calculate_ece(predictions, outcomes)
            mce = self._calculate_mce(predictions, outcomes)

            # Style the plot
            ax.set_xlabel('Predicted Probability', fontdict=FONT_LABEL, color='white')
            ax.set_ylabel('Actual Win Rate', fontdict=FONT_LABEL, color='white')
            ax.set_title(f'{title}\nBrier: {brier:.4f} | ECE: {ece:.4f} | MCE: {mce:.4f}',
                        fontdict=FONT_TITLE, color=color, pad=15)

            ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc='upper left', frameon=True, fancybox=True,
                     framealpha=0.9, edgecolor=color)

            # Add sample size annotation
            total_samples = len(self.df)
            ax.text(0.95, 0.05, f'N = {total_samples:,}',
                   transform=ax.transAxes, fontsize=10,
                   ha='right', color='white', alpha=0.7)

        plt.tight_layout()
        output_path = self.output_dir / "calibration_enhanced.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        logger.info(f"Saved enhanced calibration plot to {output_path}")
        plt.close()

    def create_time_series_evolution_plot(self, n_examples: int = 6) -> None:
        """
        Create time series plots showing binary option price evolution during contracts.
        """
        logger.info("Creating time series evolution plots...")

        # Select diverse example contracts
        contracts = self._select_diverse_contracts(n_examples)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='#0a0a0a')
        axes = axes.flatten()
        fig.suptitle('Binary Option Price Evolution During 15-Min Contracts',
                     fontsize=20, fontweight='bold', color=COLORS['primary'])

        for idx, contract_id in enumerate(contracts):
            ax = axes[idx]
            ax.set_facecolor('#111111')

            # Get contract data
            contract_data = self.df.filter(pl.col("contract_id") == contract_id).sort("seconds_offset")

            if len(contract_data) == 0:
                continue

            # Convert to numpy for plotting
            seconds = contract_data["seconds_offset"].to_numpy()
            spot = contract_data["S"].to_numpy()
            strike = contract_data["K"].to_numpy()[0]
            price_mid = contract_data["price_mid"].to_numpy()
            price_bid = contract_data["price_bid"].to_numpy()
            price_ask = contract_data["price_ask"].to_numpy()
            outcome = contract_data["outcome"].to_numpy()[0]

            # Normalize spot for dual axis
            spot_norm = (spot - strike) / strike * 100  # Percentage from strike

            # Plot binary prices
            ax.plot(seconds, price_mid, color=COLORS['primary'], linewidth=2, label='Mid Price')
            ax.fill_between(seconds, price_bid, price_ask,
                           alpha=0.2, color=COLORS['primary'], label='Bid-Ask Spread')

            # Add spot price on secondary axis
            ax2 = ax.twinx()
            ax2.plot(seconds, spot_norm, color=COLORS['warning'],
                    linewidth=1, alpha=0.6, linestyle='--', label='Spot vs Strike')
            ax2.axhline(y=0, color=COLORS['perfect'], linestyle=':', alpha=0.5)
            ax2.set_ylabel('Spot vs Strike (%)', color=COLORS['warning'])
            ax2.tick_params(axis='y', labelcolor=COLORS['warning'])

            # Add outcome indicator
            outcome_color = COLORS['success'] if outcome == 1 else COLORS['danger']
            outcome_text = "WIN" if outcome == 1 else "LOSS"
            ax.text(0.95, 0.95, outcome_text, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', ha='right', va='top',
                   color=outcome_color, bbox=dict(boxstyle="round,pad=0.3",
                                                  facecolor=outcome_color,
                                                  alpha=0.2))

            # Add final values
            final_spot = spot[-1]
            final_price = price_mid[-1] if not np.isnan(price_mid[-1]) else 0
            ax.text(0.02, 0.95, f'Strike: ${strike:.0f}\nFinal Spot: ${final_spot:.0f}\nFinal Price: {final_price:.3f}',
                   transform=ax.transAxes, fontsize=9, va='top',
                   color='white', alpha=0.8)

            # Styling
            ax.set_xlabel('Time (seconds)', fontdict=FONT_LABEL, color='white')
            ax.set_ylabel('Binary Option Price', fontdict=FONT_LABEL, color='white')
            ax.set_title(f'Contract {contract_id}', fontdict={'size': 12}, color='white')
            ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
            ax.set_xlim(0, 900)
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        output_path = self.output_dir / "time_series_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        logger.info(f"Saved time series evolution plot to {output_path}")
        plt.close()

    def create_error_distribution_analysis(self) -> None:
        """
        Create comprehensive error distribution analysis plots.
        """
        logger.info("Creating error distribution analysis...")

        fig = plt.figure(figsize=(20, 12), facecolor='#0a0a0a')
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Calculate errors
        self.df = self.df.with_columns([
            (pl.col("outcome") - pl.col("price_mid")).alias("error"),
            (pl.col("outcome") - pl.col("price_mid")).abs().alias("abs_error")
        ])

        # 1. Overall error distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#111111')
        errors = self.df["error"].to_numpy()
        errors_clean = errors[~np.isnan(errors)]

        n, bins, patches = ax1.hist(errors_clean, bins=50, alpha=0.7,
                                    color=COLORS['primary'], edgecolor='white', linewidth=0.5)

        # Color negative/positive differently
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor(COLORS['danger'])
            else:
                patch.set_facecolor(COLORS['success'])

        ax1.axvline(x=0, color=COLORS['perfect'], linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(x=np.mean(errors_clean), color=COLORS['warning'],
                   linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors_clean):.3f}')

        ax1.set_xlabel('Prediction Error (Outcome - Prediction)', fontdict=FONT_LABEL)
        ax1.set_ylabel('Frequency', fontdict=FONT_LABEL)
        ax1.set_title('Overall Prediction Error Distribution', fontdict=FONT_TITLE, color=COLORS['primary'])
        ax1.legend()
        ax1.grid(True, alpha=0.2, color=COLORS['grid'])

        # 2. QQ plot
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor('#111111')
        stats.probplot(errors_clean, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_color(COLORS['info'])
        ax2.get_lines()[1].set_color(COLORS['danger'])
        ax2.set_title('Q-Q Plot', fontdict=FONT_TITLE, color=COLORS['info'])
        ax2.grid(True, alpha=0.2, color=COLORS['grid'])

        # 3. Error by time remaining
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor('#111111')

        time_buckets = [60, 300, 600, 900]
        time_labels = ['<1min', '1-5min', '5-10min', '10-15min']
        positions = []
        errors_by_time = []

        for i, (max_t, label) in enumerate(zip(time_buckets, time_labels)):
            min_t = 0 if i == 0 else time_buckets[i-1]
            bucket_errors = self.df.filter(
                (pl.col("time_remaining") >= min_t) &
                (pl.col("time_remaining") < max_t)
            )["abs_error"].to_numpy()
            bucket_errors = bucket_errors[~np.isnan(bucket_errors)]
            errors_by_time.append(bucket_errors)
            positions.append(i)

        bp = ax3.boxplot(errors_by_time, positions=positions, widths=0.6,
                         patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], [COLORS['danger'], COLORS['warning'],
                                              COLORS['info'], COLORS['success']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_xticklabels(time_labels)
        ax3.set_xlabel('Time Remaining', fontdict=FONT_LABEL)
        ax3.set_ylabel('Absolute Error', fontdict=FONT_LABEL)
        ax3.set_title('Prediction Error by Time to Expiry', fontdict=FONT_TITLE, color=COLORS['primary'])
        ax3.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')

        # 4. Error by moneyness
        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_facecolor('#111111')

        moneyness_buckets = [(0, 0.99, 'OTM'), (0.99, 1.01, 'ATM'), (1.01, 2, 'ITM')]
        errors_by_moneyness = []
        moneyness_labels = []

        for min_m, max_m, label in moneyness_buckets:
            bucket_errors = self.df.filter(
                (pl.col("moneyness") >= min_m) &
                (pl.col("moneyness") < max_m)
            )["abs_error"].to_numpy()
            bucket_errors = bucket_errors[~np.isnan(bucket_errors)]
            if len(bucket_errors) > 0:
                errors_by_moneyness.append(bucket_errors)
                moneyness_labels.append(f'{label}\n(n={len(bucket_errors):,})')

        bp2 = ax4.boxplot(errors_by_moneyness, labels=moneyness_labels, widths=0.5,
                          patch_artist=True, showfliers=False)

        for patch in bp2['boxes']:
            patch.set_facecolor(COLORS['secondary'])
            patch.set_alpha(0.7)

        ax4.set_xlabel('Moneyness', fontdict=FONT_LABEL)
        ax4.set_ylabel('Absolute Error', fontdict=FONT_LABEL)
        ax4.set_title('Prediction Error by Moneyness', fontdict=FONT_TITLE, color=COLORS['primary'])
        ax4.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')

        plt.suptitle('Comprehensive Error Analysis', fontsize=22, fontweight='bold',
                    color=COLORS['primary'], y=0.98)

        output_path = self.output_dir / "error_distribution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        logger.info(f"Saved error distribution analysis to {output_path}")
        plt.close()

    def create_interactive_calibration_dashboard(self) -> None:
        """
        Create interactive Plotly dashboard for calibration analysis.
        """
        logger.info("Creating interactive calibration dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Calibration', 'Calibration by Time',
                          'Calibration by Moneyness', 'Error Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'histogram'}]]
        )

        # 1. Overall calibration
        cal_data = self._calculate_calibration_with_ci("price_mid", n_bins=20)

        fig.add_trace(
            go.Scatter(
                x=cal_data['bin_centers'],
                y=cal_data['actual_rates'],
                mode='markers',
                marker=dict(
                    size=np.sqrt(cal_data['counts']) / 5,
                    color=cal_data['actual_rates'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Win Rate", x=1.15)
                ),
                name='Empirical',
                text=[f'Predicted: {p:.2f}<br>Actual: {a:.2f}<br>Count: {c:,}'
                     for p, a, c in zip(cal_data['bin_centers'],
                                       cal_data['actual_rates'],
                                       cal_data['counts'])],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect',
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. Calibration by time
        time_buckets = [(0, 60), (60, 300), (300, 600), (600, 900)]
        colors_time = ['red', 'orange', 'yellow', 'green']

        for (min_t, max_t), color in zip(time_buckets, colors_time):
            df_bucket = self.df.filter(
                (pl.col("time_remaining") >= min_t) &
                (pl.col("time_remaining") < max_t)
            )
            if len(df_bucket) > 0:
                cal_time = self._calculate_calibration_with_ci("price_mid", n_bins=10,
                                                               df_subset=df_bucket)
                fig.add_trace(
                    go.Scatter(
                        x=cal_time['bin_centers'],
                        y=cal_time['actual_rates'],
                        mode='lines+markers',
                        name=f'{min_t//60}-{max_t//60}min',
                        line=dict(color=color),
                        showlegend=True
                    ),
                    row=1, col=2
                )

        # 3. Calibration by moneyness
        moneyness_buckets = [(0.95, 0.99), (0.99, 1.01), (1.01, 1.05)]
        colors_money = ['blue', 'purple', 'pink']
        labels_money = ['OTM', 'ATM', 'ITM']

        for (min_m, max_m), color, label in zip(moneyness_buckets, colors_money, labels_money):
            df_bucket = self.df.filter(
                (pl.col("moneyness") >= min_m) &
                (pl.col("moneyness") < max_m)
            )
            if len(df_bucket) > 0:
                cal_money = self._calculate_calibration_with_ci("price_mid", n_bins=10,
                                                               df_subset=df_bucket)
                fig.add_trace(
                    go.Scatter(
                        x=cal_money['bin_centers'],
                        y=cal_money['actual_rates'],
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color),
                        showlegend=True
                    ),
                    row=2, col=1
                )

        # 4. Error distribution
        errors = (self.df["outcome"] - self.df["price_mid"]).to_numpy()
        errors_clean = errors[~np.isnan(errors)]

        fig.add_trace(
            go.Histogram(
                x=errors_clean,
                nbinsx=50,
                name='Errors',
                marker=dict(color='lightblue'),
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Interactive Binary Option Calibration Dashboard",
            showlegend=True,
            height=900,
            template='plotly_dark',
            font=dict(family="Arial", size=12),
            hovermode='closest'
        )

        # Update axes
        fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
        fig.update_yaxes(title_text="Actual Win Rate", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Probability", row=1, col=2)
        fig.update_yaxes(title_text="Actual Win Rate", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
        fig.update_yaxes(title_text="Actual Win Rate", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Error", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)

        # Save interactive plot
        output_path = self.output_dir / "interactive_calibration_dashboard.html"
        fig.write_html(str(output_path))
        logger.info(f"Saved interactive dashboard to {output_path}")

    def create_pnl_analysis_plot(self, threshold: float = 0.05) -> None:
        """
        Create PnL analysis assuming we trade when model disagrees with 50% by threshold.
        """
        logger.info(f"Creating PnL analysis with threshold {threshold}...")

        # Simulate trading strategy
        self.df = self.df.with_columns([
            pl.when(pl.col("price_mid") > (0.5 + threshold))
            .then(1)  # Buy
            .when(pl.col("price_mid") < (0.5 - threshold))
            .then(-1)  # Sell
            .otherwise(0)  # No trade
            .alias("signal")
        ])

        # Calculate PnL for each trade
        self.df = self.df.with_columns([
            pl.when(pl.col("signal") == 1)
            .then((pl.col("outcome") - pl.col("price_mid")))
            .when(pl.col("signal") == -1)
            .then((pl.col("price_mid") - pl.col("outcome")))
            .otherwise(0)
            .alias("pnl")
        ])

        # Get trades only
        trades = self.df.filter(pl.col("signal") != 0)

        if len(trades) == 0:
            logger.warning("No trades generated with current threshold")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')

        # 1. Cumulative PnL
        ax1 = axes[0, 0]
        ax1.set_facecolor('#111111')

        pnl_cumsum = trades.sort("timestamp")["pnl"].to_numpy().cumsum()
        ax1.plot(pnl_cumsum, color=COLORS['primary'], linewidth=2)
        ax1.fill_between(range(len(pnl_cumsum)), 0, pnl_cumsum,
                        where=(pnl_cumsum >= 0), color=COLORS['success'], alpha=0.3)
        ax1.fill_between(range(len(pnl_cumsum)), 0, pnl_cumsum,
                        where=(pnl_cumsum < 0), color=COLORS['danger'], alpha=0.3)

        ax1.set_xlabel('Trade Number', fontdict=FONT_LABEL)
        ax1.set_ylabel('Cumulative PnL', fontdict=FONT_LABEL)
        ax1.set_title(f'Cumulative PnL (Threshold: {threshold:.1%})', fontdict=FONT_TITLE)
        ax1.grid(True, alpha=0.2, color=COLORS['grid'])

        # Add statistics
        total_pnl = pnl_cumsum[-1]
        sharpe = np.mean(trades["pnl"].to_numpy()) / np.std(trades["pnl"].to_numpy()) if len(trades) > 1 else 0
        win_rate = (trades["pnl"] > 0).sum() / len(trades)

        ax1.text(0.02, 0.98, f'Total PnL: {total_pnl:.3f}\nSharpe: {sharpe:.2f}\nWin Rate: {win_rate:.1%}',
                transform=ax1.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

        # 2. PnL Distribution
        ax2 = axes[0, 1]
        ax2.set_facecolor('#111111')

        pnl_values = trades["pnl"].to_numpy()
        ax2.hist(pnl_values, bins=30, color=COLORS['secondary'],
                alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.axvline(x=0, color=COLORS['perfect'], linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(pnl_values), color=COLORS['warning'],
                   linestyle='-', linewidth=2, label=f'Mean: {np.mean(pnl_values):.3f}')

        ax2.set_xlabel('PnL per Trade', fontdict=FONT_LABEL)
        ax2.set_ylabel('Frequency', fontdict=FONT_LABEL)
        ax2.set_title('PnL Distribution', fontdict=FONT_TITLE)
        ax2.legend()
        ax2.grid(True, alpha=0.2, color=COLORS['grid'])

        # 3. PnL by Prediction Confidence
        ax3 = axes[1, 0]
        ax3.set_facecolor('#111111')

        trades_with_conf = trades.with_columns([
            (pl.col("price_mid") - 0.5).abs().alias("confidence")
        ])

        conf_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        avg_pnls = []
        counts = []

        for i in range(len(conf_bins) - 1):
            bin_trades = trades_with_conf.filter(
                (pl.col("confidence") >= conf_bins[i]) &
                (pl.col("confidence") < conf_bins[i + 1])
            )
            if len(bin_trades) > 0:
                avg_pnls.append(bin_trades["pnl"].mean())
                counts.append(len(bin_trades))
            else:
                avg_pnls.append(0)
                counts.append(0)

        x_pos = np.arange(len(avg_pnls))
        bars = ax3.bar(x_pos, avg_pnls, color=COLORS['info'], alpha=0.7)

        # Color bars by profit/loss
        for bar, pnl in zip(bars, avg_pnls):
            if pnl < 0:
                bar.set_color(COLORS['danger'])
            else:
                bar.set_color(COLORS['success'])

        ax3.set_xlabel('Confidence Level', fontdict=FONT_LABEL)
        ax3.set_ylabel('Average PnL', fontdict=FONT_LABEL)
        ax3.set_title('PnL by Prediction Confidence', fontdict=FONT_TITLE)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{conf_bins[i]:.1f}-{conf_bins[i+1]:.1f}'
                            for i in range(len(conf_bins)-1)])
        ax3.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')

        # Add trade counts
        for i, (pnl, count) in enumerate(zip(avg_pnls, counts)):
            ax3.text(i, pnl, f'n={count}', ha='center', va='bottom' if pnl > 0 else 'top',
                    fontsize=8, color='white', alpha=0.7)

        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        ax4.set_facecolor('#111111')

        # Calculate drawdown
        cumsum = pnl_cumsum
        running_max = np.maximum.accumulate(cumsum)
        drawdown = cumsum - running_max

        ax4.fill_between(range(len(drawdown)), 0, drawdown,
                        color=COLORS['danger'], alpha=0.5)
        ax4.plot(drawdown, color=COLORS['danger'], linewidth=1.5)

        ax4.set_xlabel('Trade Number', fontdict=FONT_LABEL)
        ax4.set_ylabel('Drawdown', fontdict=FONT_LABEL)
        ax4.set_title('Drawdown Analysis', fontdict=FONT_TITLE)
        ax4.grid(True, alpha=0.2, color=COLORS['grid'])

        # Add max drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        ax4.scatter([max_dd_idx], [max_dd], color=COLORS['warning'], s=100, zorder=5)
        ax4.text(max_dd_idx, max_dd, f'Max DD: {max_dd:.3f}',
                ha='right', va='top', fontsize=10, color=COLORS['warning'])

        plt.suptitle(f'Trading Strategy PnL Analysis (Threshold: {threshold:.1%})',
                    fontsize=18, fontweight='bold', color=COLORS['primary'], y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "pnl_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        logger.info(f"Saved PnL analysis to {output_path}")
        plt.close()

    def create_implied_volatility_analysis(self) -> None:
        """
        Create implied volatility surface and smile analysis plots.
        """
        logger.info("Creating implied volatility analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0a0a0a')

        # Filter to data with IV
        iv_data = self.df.filter(pl.col("sigma_mid").is_not_null())

        # 1. IV by Moneyness
        ax1 = axes[0, 0]
        ax1.set_facecolor('#111111')

        moneyness_bins = np.linspace(0.95, 1.05, 20)
        iv_by_moneyness = []
        moneyness_centers = []

        for i in range(len(moneyness_bins) - 1):
            bin_data = iv_data.filter(
                (pl.col("moneyness") >= moneyness_bins[i]) &
                (pl.col("moneyness") < moneyness_bins[i + 1])
            )
            if len(bin_data) > 0:
                iv_by_moneyness.append(bin_data["sigma_mid"].mean())
                moneyness_centers.append((moneyness_bins[i] + moneyness_bins[i + 1]) / 2)

        ax1.plot(moneyness_centers, iv_by_moneyness, color=COLORS['primary'],
                linewidth=2, marker='o', markersize=6)
        ax1.set_xlabel('Moneyness (S/K)', fontdict=FONT_LABEL)
        ax1.set_ylabel('Implied Volatility', fontdict=FONT_LABEL)
        ax1.set_title('Volatility Smile', fontdict=FONT_TITLE)
        ax1.grid(True, alpha=0.2, color=COLORS['grid'])

        # 2. IV Term Structure
        ax2 = axes[0, 1]
        ax2.set_facecolor('#111111')

        time_bins = [0, 60, 180, 300, 450, 600, 750, 900]
        iv_by_time = []
        time_centers = []

        for i in range(len(time_bins) - 1):
            bin_data = iv_data.filter(
                (pl.col("time_remaining") >= time_bins[i]) &
                (pl.col("time_remaining") < time_bins[i + 1])
            )
            if len(bin_data) > 0:
                iv_by_time.append(bin_data["sigma_mid"].mean())
                time_centers.append((time_bins[i] + time_bins[i + 1]) / 2 / 60)  # Convert to minutes

        ax2.plot(time_centers, iv_by_time, color=COLORS['warning'],
                linewidth=2, marker='s', markersize=6)
        ax2.set_xlabel('Time to Expiry (minutes)', fontdict=FONT_LABEL)
        ax2.set_ylabel('Implied Volatility', fontdict=FONT_LABEL)
        ax2.set_title('IV Term Structure', fontdict=FONT_TITLE)
        ax2.grid(True, alpha=0.2, color=COLORS['grid'])

        # 3. IV Distribution
        ax3 = axes[1, 0]
        ax3.set_facecolor('#111111')

        iv_values = iv_data["sigma_mid"].to_numpy()
        ax3.hist(iv_values, bins=50, color=COLORS['info'], alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax3.axvline(x=np.mean(iv_values), color=COLORS['warning'],
                   linestyle='-', linewidth=2, label=f'Mean: {np.mean(iv_values):.3f}')
        ax3.axvline(x=np.median(iv_values), color=COLORS['success'],
                   linestyle='--', linewidth=2, label=f'Median: {np.median(iv_values):.3f}')

        ax3.set_xlabel('Implied Volatility', fontdict=FONT_LABEL)
        ax3.set_ylabel('Frequency', fontdict=FONT_LABEL)
        ax3.set_title('IV Distribution', fontdict=FONT_TITLE)
        ax3.legend()
        ax3.grid(True, alpha=0.2, color=COLORS['grid'])

        # 4. IV vs Realized Volatility (if we had it)
        ax4 = axes[1, 1]
        ax4.set_facecolor('#111111')

        # For now, show IV stability over time
        hourly_iv = iv_data.group_by(
            pl.col("timestamp") // 3600  # Group by hour
        ).agg(pl.col("sigma_mid").mean())

        hours = hourly_iv.select(pl.col("timestamp")).to_numpy().flatten()
        iv_hourly = hourly_iv.select(pl.col("sigma_mid")).to_numpy().flatten()

        ax4.plot(hours[:100], iv_hourly[:100], color=COLORS['secondary'],
                linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Hour', fontdict=FONT_LABEL)
        ax4.set_ylabel('Average IV', fontdict=FONT_LABEL)
        ax4.set_title('IV Stability Over Time', fontdict=FONT_TITLE)
        ax4.grid(True, alpha=0.2, color=COLORS['grid'])

        plt.suptitle('Implied Volatility Analysis',
                    fontsize=18, fontweight='bold', color=COLORS['primary'], y=1.02)
        plt.tight_layout()

        output_path = self.output_dir / "iv_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='#0a0a0a', edgecolor='none')
        logger.info(f"Saved IV analysis to {output_path}")
        plt.close()

    # Helper methods
    def _calculate_calibration_with_ci(
        self, price_col: str, n_bins: int = 10, n_bootstrap: int = 1000,
        df_subset: Optional[pl.DataFrame] = None
    ) -> dict:
        """Calculate calibration with bootstrap confidence intervals."""
        df = df_subset if df_subset is not None else self.df

        predictions = df[price_col].to_numpy()
        outcomes = df["outcome"].to_numpy()

        # Remove NaNs
        mask = ~np.isnan(predictions)
        predictions = predictions[mask]
        outcomes = outcomes[mask]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate empirical rates
        actual_rates = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])
            else:
                mask = (predictions >= bins[i]) & (predictions < bins[i + 1])

            counts[i] = mask.sum()
            if counts[i] > 0:
                actual_rates[i] = outcomes[mask].mean()
            else:
                actual_rates[i] = np.nan

        # Bootstrap confidence intervals
        ci_lower = np.zeros(n_bins)
        ci_upper = np.zeros(n_bins)

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])
            else:
                mask = (predictions >= bins[i]) & (predictions < bins[i + 1])

            if mask.sum() > 0:
                bin_outcomes = outcomes[mask]
                bootstrap_means = []

                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(bin_outcomes,
                                                       size=len(bin_outcomes),
                                                       replace=True)
                    bootstrap_means.append(bootstrap_sample.mean())

                ci_lower[i] = np.percentile(bootstrap_means, 2.5)
                ci_upper[i] = np.percentile(bootstrap_means, 97.5)
            else:
                ci_lower[i] = np.nan
                ci_upper[i] = np.nan

        return {
            'bin_centers': bin_centers,
            'actual_rates': actual_rates,
            'counts': counts,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'predictions': predictions,
            'outcomes': outcomes
        }

    def _calculate_ece(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])
            else:
                mask = (predictions >= bins[i]) & (predictions < bins[i + 1])

            if mask.sum() > 0:
                bin_pred = predictions[mask].mean()
                bin_actual = outcomes[mask].mean()
                bin_weight = mask.sum() / len(predictions)
                ece += bin_weight * abs(bin_pred - bin_actual)

        return ece

    def _calculate_mce(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        max_error = 0

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])
            else:
                mask = (predictions >= bins[i]) & (predictions < bins[i + 1])

            if mask.sum() > 0:
                bin_pred = predictions[mask].mean()
                bin_actual = outcomes[mask].mean()
                max_error = max(max_error, abs(bin_pred - bin_actual))

        return max_error

    def _select_diverse_contracts(self, n: int = 6) -> list:
        """Select diverse contracts for visualization."""
        contracts = self.df["contract_id"].unique().to_list()

        # Try to get a mix of outcomes
        wins = self.df.filter(pl.col("outcome") == 1)["contract_id"].unique().to_list()
        losses = self.df.filter(pl.col("outcome") == 0)["contract_id"].unique().to_list()

        selected = []
        for i in range(min(n // 2, len(wins))):
            selected.append(wins[i])
        for i in range(min(n // 2, len(losses))):
            selected.append(losses[i])

        # Fill remaining with any contracts if needed
        while len(selected) < n and len(contracts) > len(selected):
            for contract in contracts:
                if contract not in selected:
                    selected.append(contract)
                    break

        return selected[:n]

    def generate_all_visualizations(self) -> None:
        """Generate all professional visualizations."""
        logger.info("Generating all professional visualizations...")

        # Generate all plots
        self.create_enhanced_calibration_plot()
        self.create_time_series_evolution_plot()
        self.create_error_distribution_analysis()
        self.create_pnl_analysis_plot(threshold=0.05)
        self.create_implied_volatility_analysis()
        self.create_interactive_calibration_dashboard()

        logger.info(f"\n✅ All visualizations saved to {self.output_dir}")
        logger.info("Generated files:")
        for file in self.output_dir.glob("*"):
            logger.info(f"  - {file.name}")


def main() -> None:
    """Main entry point for visualization generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate professional visualizations")
    parser.add_argument("--input", type=str, default=str(RESULTS_FILE),
                       help="Input results parquet file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for plots")

    args = parser.parse_args()

    # Create visualizer and generate all plots
    visualizer = ProfessionalVisualizer(
        results_file=Path(args.input),
        output_dir=Path(args.output)
    )

    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()