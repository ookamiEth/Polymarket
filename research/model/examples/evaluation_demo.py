#!/usr/bin/env python3
"""
Demonstration of Model Evaluation Metrics

This script shows exactly how we evaluate the binary option pricing model.
It uses sample data to demonstrate all key metrics and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def generate_sample_predictions(n_samples: int = 1000, bias: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample predictions and outcomes for demonstration.

    Args:
        n_samples: Number of predictions to generate
        bias: Systematic bias to add (negative = underpricing)

    Returns:
        predictions: Predicted probabilities (0-1)
        outcomes: Actual outcomes (0 or 1)
    """
    # Generate predictions uniformly distributed
    predictions = np.random.uniform(0, 1, n_samples)

    # Generate outcomes based on predictions (with some noise)
    true_probs = predictions + bias
    true_probs = np.clip(true_probs, 0, 1)

    # Add realistic noise
    noise = np.random.normal(0, 0.05, n_samples)
    true_probs = np.clip(true_probs + noise, 0, 1)

    # Generate binary outcomes
    outcomes = np.random.random(n_samples) < true_probs
    outcomes = outcomes.astype(int)

    return predictions, outcomes


def calculate_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Brier Score - the key metric for probabilistic predictions.

    Brier Score = Mean((prediction - outcome)²)
    Range: 0 (perfect) to 1 (worst)
    """
    print("\n" + "="*60)
    print("CALCULATING BRIER SCORE")
    print("="*60)

    # Calculate squared errors
    errors = (predictions - outcomes) ** 2

    # Show some examples
    print("\nExample calculations:")
    for i in range(min(5, len(predictions))):
        print(f"  Prediction: {predictions[i]:.3f}, Outcome: {outcomes[i]}, "
              f"Error²: {errors[i]:.4f}")

    # Calculate mean
    brier_score = np.mean(errors)

    print(f"\nBrier Score = Mean of squared errors = {brier_score:.4f}")

    # Interpretation
    if brier_score < 0.1:
        quality = "EXCELLENT"
    elif brier_score < 0.2:
        quality = "GOOD"
    elif brier_score < 0.25:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    print(f"Quality: {quality}")
    print(f"Benchmark: Random guessing (50%) gives Brier = 0.25")

    return brier_score


def calculate_calibration(predictions: np.ndarray, outcomes: np.ndarray,
                          n_bins: int = 10) -> dict:
    """
    Calculate calibration metrics and prepare data for plotting.
    """
    print("\n" + "="*60)
    print("CALCULATING CALIBRATION")
    print("="*60)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    actual_rates = []
    predicted_rates = []
    counts = []

    print("\nCalibration by bucket:")
    print("| Bucket | Predicted | Actual | Count | Error |")
    print("|--------|-----------|--------|-------|-------|")

    for i in range(n_bins):
        # Get predictions in this bin
        if i == n_bins - 1:  # Last bin includes right edge
            mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])
        else:
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])

        count = mask.sum()
        counts.append(count)

        if count > 0:
            pred_rate = predictions[mask].mean()
            actual_rate = outcomes[mask].mean()
            predicted_rates.append(pred_rate)
            actual_rates.append(actual_rate)

            error = actual_rate - pred_rate
            print(f"| {i+1:^6} | {pred_rate:^9.1%} | {actual_rate:^6.1%} | "
                  f"{count:^5} | {error:+.1%} |")
        else:
            predicted_rates.append(np.nan)
            actual_rates.append(np.nan)

    return {
        'bin_centers': bin_centers,
        'predicted_rates': np.array(predicted_rates),
        'actual_rates': np.array(actual_rates),
        'counts': np.array(counts)
    }


def calculate_ece_mce(cal_data: dict) -> Tuple[float, float]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    print("\n" + "="*60)
    print("ECE AND MCE CALCULATION")
    print("="*60)

    # Remove NaN values
    mask = ~np.isnan(cal_data['actual_rates'])
    predicted = cal_data['predicted_rates'][mask]
    actual = cal_data['actual_rates'][mask]
    counts = cal_data['counts'][mask]

    # Calculate errors for each bin
    errors = np.abs(predicted - actual)

    # ECE: Weighted average of errors
    total_count = counts.sum()
    weights = counts / total_count
    ece = np.sum(weights * errors)

    # MCE: Maximum error
    mce = np.max(errors) if len(errors) > 0 else 0

    print(f"\nECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Interpretation: Average deviation from perfect calibration")
    print(f"  Quality: {'GOOD' if ece < 0.1 else 'NEEDS IMPROVEMENT'}")

    print(f"\nMCE (Maximum Calibration Error): {mce:.4f}")
    print(f"  Interpretation: Worst single bucket's error")
    print(f"  Quality: {'ACCEPTABLE' if mce < 0.2 else 'POOR'}")

    return ece, mce


def calculate_log_loss(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Log Loss (Cross-Entropy).
    Heavily penalizes confident wrong predictions.
    """
    print("\n" + "="*60)
    print("CALCULATING LOG LOSS")
    print("="*60)

    # Clip predictions to avoid log(0)
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)

    # Calculate log loss
    log_loss = -np.mean(
        outcomes * np.log(predictions_clipped) +
        (1 - outcomes) * np.log(1 - predictions_clipped)
    )

    print(f"\nLog Loss: {log_loss:.4f}")

    # Show impact of confident wrong predictions
    print("\nExample penalties for wrong predictions:")
    wrong_predictions = [0.99, 0.90, 0.70, 0.60]
    for pred in wrong_predictions:
        penalty = -np.log(1 - pred)
        print(f"  Predicted {pred:.0%} but wrong → Penalty: {penalty:.2f}")

    return log_loss


def analyze_by_time_buckets(predictions: np.ndarray, outcomes: np.ndarray,
                            time_remaining: np.ndarray) -> None:
    """
    Analyze performance by time remaining to expiry.
    """
    print("\n" + "="*60)
    print("ANALYSIS BY TIME REMAINING")
    print("="*60)

    # Define time buckets (in seconds)
    time_buckets = [
        (0, 60, "<1 min"),
        (60, 300, "1-5 min"),
        (300, 600, "5-10 min"),
        (600, 900, "10-15 min")
    ]

    print("\n| Time Bucket | Samples | Brier Score | Quality |")
    print("|-------------|---------|-------------|---------|")

    for min_t, max_t, label in time_buckets:
        mask = (time_remaining >= min_t) & (time_remaining < max_t)

        if mask.sum() > 0:
            bucket_preds = predictions[mask]
            bucket_outcomes = outcomes[mask]

            # Calculate Brier score for this bucket
            brier = np.mean((bucket_preds - bucket_outcomes) ** 2)

            if brier < 0.1:
                quality = "EXCELLENT"
            elif brier < 0.2:
                quality = "GOOD"
            else:
                quality = "OK"

            print(f"| {label:^11} | {mask.sum():^7} | {brier:^11.4f} | {quality:^7} |")


def analyze_trading_performance(predictions: np.ndarray, outcomes: np.ndarray,
                               threshold: float = 0.05) -> None:
    """
    Analyze trading performance using threshold strategy.
    """
    print("\n" + "="*60)
    print(f"TRADING PERFORMANCE ANALYSIS (Threshold: {threshold:.1%})")
    print("="*60)

    # Generate trading signals
    signals = np.zeros_like(predictions)
    signals[predictions > 0.5 + threshold] = 1  # Buy
    signals[predictions < 0.5 - threshold] = -1  # Sell

    # Calculate PnL for each trade
    pnl = np.zeros_like(predictions)

    # Buy signals
    buy_mask = signals == 1
    pnl[buy_mask] = outcomes[buy_mask] - predictions[buy_mask]

    # Sell signals
    sell_mask = signals == -1
    pnl[sell_mask] = predictions[sell_mask] - outcomes[sell_mask]

    # Get trades only
    trade_mask = signals != 0
    trade_pnl = pnl[trade_mask]

    if len(trade_pnl) > 0:
        # Calculate metrics
        total_trades = len(trade_pnl)
        winning_trades = (trade_pnl > 0).sum()
        total_pnl = trade_pnl.sum()
        avg_pnl = trade_pnl.mean()

        # Sharpe ratio (simplified)
        if trade_pnl.std() > 0:
            sharpe = trade_pnl.mean() / trade_pnl.std()
        else:
            sharpe = 0

        # Max drawdown
        cumsum = trade_pnl.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        drawdown = cumsum - running_max
        max_drawdown = drawdown.min()

        print(f"\n📊 Trading Statistics:")
        print(f"  Total Trades: {total_trades:,}")
        print(f"  Win Rate: {winning_trades/total_trades:.1%}")
        print(f"  Total PnL: {total_pnl:+.2f}")
        print(f"  Average PnL per Trade: {avg_pnl:+.4f}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}")

        # PnL by confidence level
        print(f"\n📈 PnL by Confidence Level:")
        confidence_levels = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]

        for min_conf, max_conf in confidence_levels:
            # Distance from 50%
            distance = np.abs(predictions - 0.5)
            conf_mask = (distance >= min_conf) & (distance < max_conf) & trade_mask

            if conf_mask.sum() > 0:
                conf_pnl = pnl[conf_mask]
                print(f"  {min_conf:.0%}-{max_conf:.0%} from 50%: "
                      f"Avg PnL = {conf_pnl.mean():+.4f}, "
                      f"Trades = {len(conf_pnl):,}")


def plot_calibration(cal_data: dict, save_path: str = None) -> None:
    """
    Create calibration plot.
    """
    plt.figure(figsize=(8, 8), facecolor='white')

    # Remove NaN values
    mask = ~np.isnan(cal_data['actual_rates'])

    # Plot calibration points
    plt.scatter(cal_data['predicted_rates'][mask],
               cal_data['actual_rates'][mask],
               s=cal_data['counts'][mask] * 2,  # Size by sample count
               alpha=0.6, color='blue', edgecolors='black', linewidth=1)

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2,
             label='Perfect Calibration')

    # Labels and formatting
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Actual Win Rate', fontsize=12)
    plt.title('Calibration Plot\nDots should follow the diagonal line', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add sample size note
    total_samples = cal_data['counts'].sum()
    plt.text(0.02, 0.98, f'N = {total_samples:,}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def run_complete_evaluation():
    """
    Run a complete evaluation demonstration.
    """
    print("\n" + "🎯"*30)
    print(" BINARY OPTION MODEL EVALUATION DEMONSTRATION")
    print("🎯"*30)

    # Generate sample data
    print("\n📊 Generating sample predictions and outcomes...")
    np.random.seed(42)  # For reproducibility

    # Generate main dataset with slight underpricing bias (like our real model)
    predictions, outcomes = generate_sample_predictions(n_samples=10000, bias=-0.012)

    # Generate time remaining (uniform from 0 to 900 seconds)
    time_remaining = np.random.uniform(0, 900, len(predictions))

    print(f"Generated {len(predictions):,} predictions")
    print(f"Mean prediction: {predictions.mean():.3f}")
    print(f"Mean outcome: {outcomes.mean():.3f}")
    print(f"Bias: {outcomes.mean() - predictions.mean():+.3f}")

    # 1. Calculate Brier Score
    brier = calculate_brier_score(predictions, outcomes)

    # 2. Calculate Calibration
    cal_data = calculate_calibration(predictions, outcomes)

    # 3. Calculate ECE and MCE
    ece, mce = calculate_ece_mce(cal_data)

    # 4. Calculate Log Loss
    log_loss = calculate_log_loss(predictions, outcomes)

    # 5. Analyze by Time
    analyze_by_time_buckets(predictions, outcomes, time_remaining)

    # 6. Analyze Trading Performance
    analyze_trading_performance(predictions, outcomes, threshold=0.05)

    # 7. Create Calibration Plot
    plot_calibration(cal_data)

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print(f"""
📊 Key Metrics:
  - Brier Score: {brier:.4f} (target < 0.25)
  - ECE: {ece:.4f} (target < 0.10)
  - MCE: {mce:.4f} (target < 0.20)
  - Log Loss: {log_loss:.4f}

🎯 Overall Assessment:
  - Calibration: {'GOOD' if ece < 0.1 else 'NEEDS IMPROVEMENT'}
  - Accuracy: {'EXCELLENT' if brier < 0.2 else 'GOOD' if brier < 0.25 else 'POOR'}
  - Trading Potential: {'YES' if brier < 0.25 else 'NO'}
    """)

    print("\n" + "💡"*30)
    print("\n KEY INSIGHTS:")
    print("  1. Lower Brier Score = Better predictions")
    print("  2. Points on diagonal = Well calibrated")
    print("  3. Profitable trading requires Brier < 0.25")
    print("  4. Time decay improves accuracy")
    print("  5. Small bias can be traded profitably")
    print("\n" + "💡"*30)


def interactive_evaluation():
    """
    Interactive evaluation where user can input their own predictions.
    """
    print("\n" + "🖩"*30)
    print(" INTERACTIVE EVALUATION")
    print("🖩"*30)

    print("\nEnter your predictions and outcomes (comma-separated):")
    print("Example: 0.7,0.3,0.8,0.2,0.6")

    try:
        pred_input = input("Predictions: ")
        predictions = np.array([float(x.strip()) for x in pred_input.split(',')])

        out_input = input("Outcomes (0 or 1): ")
        outcomes = np.array([int(x.strip()) for x in out_input.split(',')])

        if len(predictions) != len(outcomes):
            print("Error: Number of predictions and outcomes must match!")
            return

        # Calculate metrics
        brier = np.mean((predictions - outcomes) ** 2)

        print(f"\n📊 Results:")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Mean Prediction: {predictions.mean():.3f}")
        print(f"  Mean Outcome: {outcomes.mean():.3f}")
        print(f"  Bias: {outcomes.mean() - predictions.mean():+.3f}")

        # Interpretation
        if brier < 0.25:
            print(f"  Assessment: GOOD - Better than random!")
        else:
            print(f"  Assessment: POOR - Worse than random guessing")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_evaluation()
    else:
        run_complete_evaluation()
        print("\n" + "="*60)
        print("Run with --interactive to evaluate your own predictions:")
        print("  python evaluation_demo.py --interactive")
        print("="*60)


if __name__ == "__main__":
    main()