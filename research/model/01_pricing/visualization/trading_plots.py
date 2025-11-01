#!/usr/bin/env python3
"""
Trading simulation plots for model evaluation.

Provides equity curve and P&L analysis based on simple threshold strategies.

Functions:
    - plot_equity_curve(): Equity curve with cumulative P&L and drawdown
    - generate_trading_report(): Orchestrates all trading plots
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from visualization.plot_config import COLORS, FONT_SIZES, apply_plot_style, get_plot_output_path
from visualization.wandb_integration import upload_plot

logger = logging.getLogger(__name__)


def _calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 365 * 24 * 4) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns (P&L per trade)
        periods_per_year: Number of periods per year (default: 15-min bars)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return float(sharpe)


def _calculate_max_drawdown(cumulative_pnl: np.ndarray) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        cumulative_pnl: Array of cumulative P&L

    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    cumulative_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - cumulative_max
    max_dd = float(np.min(drawdown))
    end_idx = int(np.argmin(drawdown))

    # Find start of drawdown (last peak before max drawdown)
    start_idx = 0
    if end_idx > 0:
        start_idx = int(np.argmax(cumulative_pnl[:end_idx]))

    return max_dd, start_idx, end_idx


def plot_equity_curve(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    bet_size: float = 1.0,
) -> dict[str, Any]:
    """
    Plot equity curve from simple threshold trading strategy.

    Strategy:
        - LONG if predicted_prob > long_threshold
        - SHORT if predicted_prob < short_threshold
        - FLAT otherwise

    Args:
        test_df: Test predictions with columns:
            - final_prob (model predictions)
            - prob_mid (baseline predictions)
            - outcome (binary outcome, 1 = UP)
            - timestamp (for x-axis)
        output_path: Path to save plot (optional, auto-generated if None)
        wandb_log: Whether to upload to W&B
        long_threshold: Threshold for long position (default: 0.55)
        short_threshold: Threshold for short position (default: 0.45)
        bet_size: Fixed bet size per trade (default: 1.0)

    Returns:
        Dictionary with trading statistics
    """
    logger.info("Generating equity curve plot...")

    # Sort by timestamp
    test_df = test_df.sort("timestamp")

    # Extract arrays
    timestamps = test_df["timestamp"].to_numpy()
    final_prob_model = test_df["final_prob"].to_numpy()
    prob_mid_baseline = test_df["prob_mid"].to_numpy()
    outcomes = test_df["outcome"].to_numpy()

    # Generate trading signals (model)
    # LONG (1) if prob > long_threshold, SHORT (-1) if prob < short_threshold, FLAT (0) otherwise
    signals_model = np.where(
        final_prob_model > long_threshold,
        1,
        np.where(final_prob_model < short_threshold, -1, 0),
    )

    # Generate trading signals (baseline)
    signals_baseline = np.where(
        prob_mid_baseline > long_threshold,
        1,
        np.where(prob_mid_baseline < short_threshold, -1, 0),
    )

    # Calculate P&L per trade
    # If LONG and outcome=1 (UP): +bet_size, if LONG and outcome=0 (DOWN): -bet_size
    # If SHORT and outcome=0 (DOWN): +bet_size, if SHORT and outcome=1 (UP): -bet_size
    # If FLAT: 0
    pnl_model = np.where(
        signals_model == 1,
        np.where(outcomes == 1, bet_size, -bet_size),
        np.where(signals_model == -1, np.where(outcomes == 0, bet_size, -bet_size), 0),
    )

    pnl_baseline = np.where(
        signals_baseline == 1,
        np.where(outcomes == 1, bet_size, -bet_size),
        np.where(signals_baseline == -1, np.where(outcomes == 0, bet_size, -bet_size), 0),
    )

    # Cumulative P&L
    cumulative_pnl_model = np.cumsum(pnl_model)
    cumulative_pnl_baseline = np.cumsum(pnl_baseline)

    # Calculate statistics
    total_pnl_model = float(cumulative_pnl_model[-1])
    total_pnl_baseline = float(cumulative_pnl_baseline[-1])

    n_trades_model = int(np.sum(signals_model != 0))
    n_trades_baseline = int(np.sum(signals_baseline != 0))

    n_long_model = int(np.sum(signals_model == 1))
    n_short_model = int(np.sum(signals_model == -1))

    win_rate_model = float(np.sum(pnl_model[signals_model != 0] > 0) / max(n_trades_model, 1))
    win_rate_baseline = float(np.sum(pnl_baseline[signals_baseline != 0] > 0) / max(n_trades_baseline, 1))

    sharpe_model = _calculate_sharpe_ratio(pnl_model[signals_model != 0])
    sharpe_baseline = _calculate_sharpe_ratio(pnl_baseline[signals_baseline != 0])

    max_dd_model, dd_start_model, dd_end_model = _calculate_max_drawdown(cumulative_pnl_model)
    max_dd_baseline, _dd_start_baseline, _dd_end_baseline = _calculate_max_drawdown(cumulative_pnl_baseline)

    logger.info("\nModel Strategy:")
    logger.info(f"  Total P&L: {total_pnl_model:+.2f} ({n_trades_model:,} trades)")
    logger.info(f"  Win Rate: {win_rate_model:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe_model:.2f}")
    logger.info(f"  Max Drawdown: {max_dd_model:.2f}")
    logger.info(f"  Long/Short: {n_long_model:,} / {n_short_model:,}")

    logger.info("\nBaseline Strategy:")
    logger.info(f"  Total P&L: {total_pnl_baseline:+.2f} ({n_trades_baseline:,} trades)")
    logger.info(f"  Win Rate: {win_rate_baseline:.1%}")
    logger.info(f"  Sharpe Ratio: {sharpe_baseline:.2f}")
    logger.info(f"  Max Drawdown: {max_dd_baseline:.2f}")

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    apply_plot_style()

    # 3x2 subplot layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Cumulative P&L (full plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, cumulative_pnl_model, linewidth=2, color=COLORS["primary"], label="Model Strategy", alpha=0.9)
    ax1.plot(
        timestamps, cumulative_pnl_baseline, linewidth=2, color=COLORS["danger"], label="Baseline Strategy", alpha=0.9
    )
    ax1.axhline(0, color=COLORS["perfect"], linewidth=1, linestyle="--", alpha=0.5)

    # Highlight drawdown period
    if dd_start_model < dd_end_model:
        ax1.axvspan(
            float(timestamps[dd_start_model]),
            float(timestamps[dd_end_model]),
            alpha=0.1,
            color=COLORS["warning"],
            label="Max DD Period",
        )

    ax1.set_xlabel("Time", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax1.set_ylabel("Cumulative P&L", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax1.set_title(
        f"Equity Curve (Long>{long_threshold:.2f}, Short<{short_threshold:.2f})",
        fontsize=FONT_SIZES["title"] + 2,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=FONT_SIZES["legend"])
    ax1.grid(True, alpha=0.2, linewidth=0.5)

    # 2. Drawdown (model)
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_max_model = np.maximum.accumulate(cumulative_pnl_model)
    drawdown_model = cumulative_pnl_model - cumulative_max_model
    ax2.fill_between(timestamps, 0, drawdown_model, color=COLORS["danger"], alpha=0.3, linewidth=0)
    ax2.plot(timestamps, drawdown_model, color=COLORS["danger"], linewidth=1.5, alpha=0.9)
    ax2.axhline(
        max_dd_model, color=COLORS["warning"], linewidth=1.5, linestyle="--", label=f"Max DD: {max_dd_model:.2f}"
    )
    ax2.set_xlabel("Time", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax2.set_ylabel("Drawdown", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax2.set_title("Model Drawdown", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax2.legend(loc="lower right", fontsize=FONT_SIZES["legend"])
    ax2.grid(True, alpha=0.2, linewidth=0.5)

    # 3. Drawdown (baseline)
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative_max_baseline = np.maximum.accumulate(cumulative_pnl_baseline)
    drawdown_baseline = cumulative_pnl_baseline - cumulative_max_baseline
    ax3.fill_between(timestamps, 0, drawdown_baseline, color=COLORS["danger"], alpha=0.3, linewidth=0)
    ax3.plot(timestamps, drawdown_baseline, color=COLORS["danger"], linewidth=1.5, alpha=0.9)
    ax3.axhline(
        max_dd_baseline, color=COLORS["warning"], linewidth=1.5, linestyle="--", label=f"Max DD: {max_dd_baseline:.2f}"
    )
    ax3.set_xlabel("Time", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax3.set_ylabel("Drawdown", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax3.set_title("Baseline Drawdown", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax3.legend(loc="lower right", fontsize=FONT_SIZES["legend"])
    ax3.grid(True, alpha=0.2, linewidth=0.5)

    # 4. Per-trade P&L distribution (model)
    ax4 = fig.add_subplot(gs[2, 0])
    pnl_trades_model = pnl_model[signals_model != 0]
    ax4.hist(pnl_trades_model, bins=50, alpha=0.7, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    ax4.axvline(0, color=COLORS["perfect"], linewidth=1.5, linestyle="--", alpha=0.7)
    mean_pnl_model = float(np.mean(pnl_trades_model))
    ax4.axvline(
        mean_pnl_model,
        color=COLORS["success"],
        linewidth=2,
        linestyle="--",
        label=f"Mean: {mean_pnl_model:.3f}",
    )
    ax4.set_xlabel("P&L per Trade", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax4.set_ylabel("Frequency", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax4.set_title(f"Model Trade Distribution (n={n_trades_model:,})", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax4.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax4.grid(True, alpha=0.2, linewidth=0.5)

    # 5. Per-trade P&L distribution (baseline)
    ax5 = fig.add_subplot(gs[2, 1])
    pnl_trades_baseline = pnl_baseline[signals_baseline != 0]
    ax5.hist(pnl_trades_baseline, bins=50, alpha=0.7, color=COLORS["danger"], edgecolor="white", linewidth=0.5)
    ax5.axvline(0, color=COLORS["perfect"], linewidth=1.5, linestyle="--", alpha=0.7)
    mean_pnl_baseline = float(np.mean(pnl_trades_baseline))
    ax5.axvline(
        mean_pnl_baseline,
        color=COLORS["success"],
        linewidth=2,
        linestyle="--",
        label=f"Mean: {mean_pnl_baseline:.3f}",
    )
    ax5.set_xlabel("P&L per Trade", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax5.set_ylabel("Frequency", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax5.set_title(
        f"Baseline Trade Distribution (n={n_trades_baseline:,})", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax5.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax5.grid(True, alpha=0.2, linewidth=0.5)

    # Main title
    fig.suptitle(
        "Trading Simulation: Simple Threshold Strategy\n"
        f"Model P&L: {total_pnl_model:+.2f} | Baseline P&L: {total_pnl_baseline:+.2f} | "
        f"Improvement: {total_pnl_model - total_pnl_baseline:+.2f}",
        fontsize=FONT_SIZES["title"] + 4,
        fontweight="bold",
        y=0.98,
    )

    # Save plot
    if output_path is None:
        output_path = str(get_plot_output_path("trading", "equity_curve.png"))

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved equity curve plot to {output_path}")

    # Upload to W&B
    if wandb_log and output_path is not None:
        upload_plot(output_path, "equity_curve")

    plt.close()

    # Return statistics
    return {
        "model": {
            "total_pnl": total_pnl_model,
            "n_trades": n_trades_model,
            "n_long": n_long_model,
            "n_short": n_short_model,
            "win_rate": win_rate_model,
            "sharpe_ratio": sharpe_model,
            "max_drawdown": max_dd_model,
        },
        "baseline": {
            "total_pnl": total_pnl_baseline,
            "n_trades": n_trades_baseline,
            "win_rate": win_rate_baseline,
            "sharpe_ratio": sharpe_baseline,
            "max_drawdown": max_dd_baseline,
        },
        "improvement": {
            "pnl_delta": total_pnl_model - total_pnl_baseline,
            "sharpe_delta": sharpe_model - sharpe_baseline,
        },
        "strategy": {
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "bet_size": bet_size,
        },
    }


def generate_trading_report(
    test_file: str,
    output_dir: str = "results/plots/trading/",
    wandb_log: bool = True,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
) -> dict[str, Any]:
    """
    Generate comprehensive trading simulation report.

    Creates:
    - Equity curve with cumulative P&L
    - Drawdown analysis
    - Per-trade P&L distribution
    - Performance metrics (Sharpe, win rate, etc.)

    Args:
        test_file: Path to test predictions parquet
        output_dir: Output directory for plots
        wandb_log: Whether to upload to W&B
        long_threshold: Threshold for long position
        short_threshold: Threshold for short position

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 80)
    logger.info("GENERATING TRADING REPORT")
    logger.info("=" * 80)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {test_file}...")
    test_df = pl.read_parquet(test_file)
    logger.info(f"Loaded {len(test_df):,} test predictions")

    # Generate equity curve plot
    logger.info("\n[1/1] Generating equity curve...")
    try:
        trading_results = plot_equity_curve(
            test_df=test_df,
            output_path=str(output_dir_path / "equity_curve.png"),
            wandb_log=wandb_log,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
        )

        logger.info("\n" + "=" * 80)
        logger.info("TRADING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model P&L:       {trading_results['model']['total_pnl']:+.2f}")
        logger.info(f"Baseline P&L:    {trading_results['baseline']['total_pnl']:+.2f}")
        logger.info(f"Improvement:     {trading_results['improvement']['pnl_delta']:+.2f}")
        logger.info(f"Model Sharpe:    {trading_results['model']['sharpe_ratio']:.2f}")
        logger.info(f"Baseline Sharpe: {trading_results['baseline']['sharpe_ratio']:.2f}")
        logger.info("=" * 80)

        return trading_results

    except Exception as e:
        logger.error(f"Failed to generate trading plots: {e}")
        return {}


def main() -> None:
    """CLI entry point for trading plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate trading simulation plots")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test predictions parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/trading/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        default=0.55,
        help="Threshold for long position (default: 0.55)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.45,
        help="Threshold for short position (default: 0.45)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_trading_report(
        test_file=args.test_file,
        output_dir=args.output_dir,
        wandb_log=not args.no_wandb,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )


if __name__ == "__main__":
    main()
