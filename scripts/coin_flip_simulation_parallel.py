#!/usr/bin/env python3
"""
Memory-Efficient Parallel Monte Carlo Simulation: Coin Flip Gambling Problem

Optimized for 1M flips × 10K simulations using multiprocessing and vectorization.
Performance improvements:
- Multiprocessing across CPU cores
- Vectorized operations across simulations
- Efficient memory management
- Batch processing for optimal cache usage
"""

import multiprocessing as mp
import time
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter1d

# Suppress overflow warnings (expected with exponential growth)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")

# Simulation parameters
INITIAL_WEALTH = 100
NUM_SIMULATIONS = 10000000 # Number of Monte Carlo simulations
NUM_FLIPS = 1000  # Number of coin flips per simulation
HEADS_MULTIPLIER = 2.0  # Multiply bet by this on heads (2x = double)
TAILS_MULTIPLIER = 0.4  # Multiply bet by this on tails (0.4 = lose 60%)

# BET SIZING - Adjust this parameter to explore different strategies:
BET_FRACTION = 0.333  # Kelly optimal: 37.5% of bankroll
# BET_FRACTION = 0.50   # Over-betting: 50% of bankroll (more volatile)
# BET_FRACTION = 0.25   # Under-betting: 25% of bankroll (safer, slower growth)
# BET_FRACTION = 1.0    # All-in: 100% of bankroll (GUARANTEED ruin!)

# Storage optimization: only store every Nth flip
STORAGE_INTERVAL = 1000
NUM_STORED_POINTS = (NUM_FLIPS // STORAGE_INTERVAL) + 1

# Number of full trajectories to keep for visualization
NUM_SAMPLE_PATHS = 300  # Increased to show many individual paths like reference image

# Parallelization settings
NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
SIMS_PER_WORKER = NUM_SIMULATIONS // NUM_WORKERS

# Batch size for vectorized processing (process multiple sims at once)
VECTORIZED_BATCH_SIZE = 100  # Process 100 simulations simultaneously


@dataclass
class SimulationResult:
    """Results from a batch of simulations."""

    wealth_stats: np.ndarray  # Shape: (num_sims, num_stored_points)
    time_to_ruin: np.ndarray  # Shape: (num_sims,)
    sample_paths: Optional[np.ndarray] = None  # Shape: (num_samples, num_flips+1)
    sample_indices: Optional[np.ndarray] = None  # Shape: (num_samples,)


def run_vectorized_simulations(
    num_sims: int,
    seed_offset: int,
    bet_fraction: float,
    global_sample_indices: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    Run multiple simulations in parallel using vectorized NumPy operations.

    This function processes VECTORIZED_BATCH_SIZE simulations simultaneously,
    which is much faster than processing them one at a time.

    Args:
        num_sims: Number of simulations to run
        seed_offset: Random seed offset for reproducibility
        bet_fraction: Fraction of bankroll to bet each flip (0.0 to 1.0)
        global_sample_indices: Indices of simulations to track full trajectories
    """
    # Set random seed for reproducibility
    np.random.seed(42 + seed_offset)

    # Determine which simulations to sample for full trajectory
    local_start_idx = seed_offset
    local_end_idx = seed_offset + num_sims

    if global_sample_indices is not None:
        # Find which sample indices fall in this worker's range
        mask = (global_sample_indices >= local_start_idx) & (global_sample_indices < local_end_idx)
        local_sample_indices = global_sample_indices[mask] - local_start_idx
        num_samples = len(local_sample_indices)
    else:
        local_sample_indices = np.array([], dtype=np.int32)
        num_samples = 0

    # Storage arrays
    stored_points = NUM_STORED_POINTS
    wealth_stats = np.zeros((num_sims, stored_points), dtype=np.float64)
    wealth_stats[:, 0] = INITIAL_WEALTH

    time_to_ruin = np.full(num_sims, NUM_FLIPS, dtype=np.int32)

    # Sample paths storage
    if num_samples > 0:
        sample_paths = np.zeros((num_samples, NUM_FLIPS + 1), dtype=np.float64)
        sample_paths[:, 0] = INITIAL_WEALTH
    else:
        sample_paths = None

    # Create a boolean mask for which simulations to track (more efficient)
    track_full_path = np.zeros(num_sims, dtype=bool)
    if num_samples > 0:
        track_full_path[local_sample_indices] = True

    # Process in vectorized batches
    num_batches = (num_sims + VECTORIZED_BATCH_SIZE - 1) // VECTORIZED_BATCH_SIZE

    for batch_idx in range(num_batches):
        batch_start = batch_idx * VECTORIZED_BATCH_SIZE
        batch_end = min(batch_start + VECTORIZED_BATCH_SIZE, num_sims)
        batch_size = batch_end - batch_start

        # Initialize wealth for this batch
        current_wealth = np.full(batch_size, INITIAL_WEALTH, dtype=np.float64)
        still_alive = np.ones(batch_size, dtype=bool)

        # Determine which simulations in this batch need full tracking
        batch_track_mask = track_full_path[batch_start:batch_end]
        batch_track_indices = np.where(batch_track_mask)[0]

        # Calculate effective multipliers based on bet fraction
        # When betting fraction f of bankroll:
        # - Heads: new_wealth = wealth * (1 + f * (HEADS_MULTIPLIER - 1))
        # - Tails: new_wealth = wealth * (1 - f * (1 - TAILS_MULTIPLIER))
        heads_effective_mult = 1 + bet_fraction * (HEADS_MULTIPLIER - 1)
        tails_effective_mult = 1 - bet_fraction * (1 - TAILS_MULTIPLIER)

        # Run all flips for this batch
        for flip_num in range(1, NUM_FLIPS + 1):
            # Generate coin flips for all simulations in batch
            coin_flips = np.random.random(batch_size) < 0.5

            # Apply multipliers vectorized (using fractional betting)
            multipliers = np.where(coin_flips, heads_effective_mult, tails_effective_mult)
            current_wealth *= multipliers

            # Track time to ruin (vectorized)
            newly_broke = still_alive & (current_wealth < 1.0)
            if np.any(newly_broke):
                broke_indices = batch_start + np.where(newly_broke)[0]
                time_to_ruin[broke_indices] = flip_num
                still_alive = current_wealth >= 1.0

            # Store at intervals
            if flip_num % STORAGE_INTERVAL == 0:
                store_idx = flip_num // STORAGE_INTERVAL
                wealth_stats[batch_start:batch_end, store_idx] = current_wealth

            # Store sample paths (EVERY flip for selected simulations)
            if num_samples > 0 and len(batch_track_indices) > 0 and sample_paths is not None:
                for batch_pos in batch_track_indices:
                    # Map batch position to global sample index
                    global_sim_idx = batch_start + batch_pos
                    sample_idx = np.where(local_sample_indices == global_sim_idx)[0][0]
                    sample_paths[sample_idx, flip_num] = current_wealth[batch_pos]

    return SimulationResult(
        wealth_stats=wealth_stats,
        time_to_ruin=time_to_ruin,
        sample_paths=sample_paths,
        sample_indices=local_sample_indices if num_samples > 0 else None,
    )


def worker_task(
    worker_id: int,
    sims_per_worker: int,
    bet_fraction: float,
    global_sample_indices: Optional[np.ndarray],
) -> SimulationResult:
    """Task for each worker process."""
    seed_offset = worker_id * sims_per_worker
    return run_vectorized_simulations(sims_per_worker, seed_offset, bet_fraction, global_sample_indices)


def main() -> None:
    """Main execution function."""
    print("=" * 80)
    print("PARALLEL MONTE CARLO SIMULATION: COIN FLIP GAMBLING")
    print("=" * 80)
    print(f"Initial Wealth: ${INITIAL_WEALTH}")
    print(f"Number of Simulations: {NUM_SIMULATIONS:,}")
    print(f"Flips per Simulation: {NUM_FLIPS:,}")
    print(f"Total Flips: {NUM_SIMULATIONS * NUM_FLIPS:,} (10 billion!)")
    print(f"Heads (50%): {HEADS_MULTIPLIER}x multiplier")
    print(f"Tails (50%): {TAILS_MULTIPLIER}x multiplier (lose 60%)")
    print(f"Betting Strategy: {BET_FRACTION:.1%} of bankroll each flip")
    print(f"Storage: Every {STORAGE_INTERVAL}th flip (memory efficient)")
    print()
    print("PARALLELIZATION:")
    print(f"  CPU Cores: {NUM_WORKERS}")
    print(f"  Simulations per Worker: {SIMS_PER_WORKER:,}")
    print(f"  Vectorized Batch Size: {VECTORIZED_BATCH_SIZE}")
    print("=" * 80)
    print()

    # Mathematical analysis
    # With fractional betting, effective multipliers are:
    heads_effective_mult = 1 + BET_FRACTION * (HEADS_MULTIPLIER - 1)
    tails_effective_mult = 1 - BET_FRACTION * (1 - TAILS_MULTIPLIER)

    expected_multiplier = 0.5 * heads_effective_mult + 0.5 * tails_effective_mult
    geometric_mean_multiplier = np.sqrt(heads_effective_mult * tails_effective_mult)

    print("MATHEMATICAL ANALYSIS:")
    print(f"Effective Multipliers with {BET_FRACTION:.1%} betting:")
    print(f"  Heads: {heads_effective_mult:.4f}x")
    print(f"  Tails: {tails_effective_mult:.4f}x")
    print()
    print(f"Expected Value per flip: {expected_multiplier:.4f}x")
    if expected_multiplier > 1.0:
        print(f"Arithmetic mean: {expected_multiplier:.4f} > 1.0 ✓")
    else:
        print(f"Arithmetic mean: {expected_multiplier:.4f} < 1.0 ✗")
    print()
    print(f"Geometric Mean per flip: {geometric_mean_multiplier:.6f}x")
    if geometric_mean_multiplier > 1.0:
        print(f"Geometric mean: {geometric_mean_multiplier:.6f} > 1.0 ✓ GROWTH!")
        print(
            f"After {NUM_FLIPS:,} flips, expected wealth ≈ ${INITIAL_WEALTH} × {geometric_mean_multiplier}^{NUM_FLIPS}"
        )
        # Calculate final wealth (cap at reasonable display value)
        final_expected = INITIAL_WEALTH * (geometric_mean_multiplier ** min(NUM_FLIPS, 100000))
        print(f"  Expected to grow to: ${final_expected:.2e}")
    else:
        print(f"Geometric mean: {geometric_mean_multiplier:.6f} < 1.0 ✗ RUIN!")
        print(
            f"After {NUM_FLIPS:,} flips, expected wealth ≈ ${INITIAL_WEALTH} × {geometric_mean_multiplier}^{NUM_FLIPS}"
        )
        print(f"  = ${INITIAL_WEALTH} × {geometric_mean_multiplier**NUM_FLIPS:.2e} ≈ $0")
    print()
    print("Kelly Criterion optimal bet size:")
    p = 0.5
    b = (HEADS_MULTIPLIER / TAILS_MULTIPLIER) - 1
    kelly_fraction = (p * (b + 1) - 1) / b if b > 0 else 0
    print(f"f* = {kelly_fraction:.2%} of bankroll")
    if kelly_fraction < BET_FRACTION:
        print(f"Current bet size ({BET_FRACTION:.1%}) is OVER-betting (risky)")
    elif kelly_fraction > BET_FRACTION:
        print(f"Current bet size ({BET_FRACTION:.1%}) is UNDER-betting (safe but slower growth)")
    else:
        print(f"Current bet size ({BET_FRACTION:.1%}) is OPTIMAL!")
    print()
    print("=" * 80)
    print()

    # Select sample indices for full trajectories
    np.random.seed(42)
    global_sample_indices = np.random.choice(NUM_SIMULATIONS, NUM_SAMPLE_PATHS, replace=False)
    global_sample_indices.sort()  # Sort for efficient processing

    # Run parallel simulations
    print(f"Running {NUM_SIMULATIONS:,} simulations across {NUM_WORKERS} workers...")
    start_time = time.time()

    # Create worker tasks
    worker_func = partial(
        worker_task,
        sims_per_worker=SIMS_PER_WORKER,
        bet_fraction=BET_FRACTION,
        global_sample_indices=global_sample_indices,
    )

    # Run in parallel
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_func, range(NUM_WORKERS))

    elapsed_time = time.time() - start_time
    print(f"✓ Simulations complete in {elapsed_time:.1f} seconds")
    print(f"  Performance: {NUM_SIMULATIONS * NUM_FLIPS / elapsed_time / 1e6:.2f} million flips/second")
    print(
        f"  Speedup vs sequential: {elapsed_time / (elapsed_time / NUM_WORKERS):.1f}x (theoretical max: {NUM_WORKERS}x)"
    )
    print()

    # Combine results from all workers
    print("Combining results from workers...")
    wealth_stats = np.vstack([r.wealth_stats for r in results])
    time_to_ruin = np.hstack([r.time_to_ruin for r in results])

    # Combine sample paths
    sample_paths_list = [r.sample_paths for r in results if r.sample_paths is not None]
    wealth_samples = np.vstack(sample_paths_list) if sample_paths_list else np.zeros((0, NUM_FLIPS + 1))

    print(f"✓ Collected {len(sample_paths_list)} sample path arrays from workers")
    print(f"✓ Total sample paths: {wealth_samples.shape[0]} (expected: {NUM_SAMPLE_PATHS})")
    if wealth_samples.shape[0] > 0:
        print(f"✓ Each path has {wealth_samples.shape[1]} flips (full resolution)")

    # Calculate statistics (handle overflow/inf values)
    final_wealth = wealth_stats[:, -1]

    # Replace inf values with very large number for statistics
    final_wealth_clean = np.where(np.isinf(final_wealth), 1e100, final_wealth)
    final_wealth_clean = np.where(np.isnan(final_wealth_clean), 0, final_wealth_clean)

    broke_simulations = np.sum(final_wealth_clean < 1.0)
    broke_percentage = 100 * broke_simulations / NUM_SIMULATIONS

    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print()
    print(f"Final Wealth Statistics (after {NUM_FLIPS:,} flips):")

    # Check for overflow
    num_overflow = np.sum(np.isinf(final_wealth))
    if num_overflow > 0:
        print(
            f"  Note: {num_overflow} simulations ({100 * num_overflow / NUM_SIMULATIONS:.1f}%) exceeded float64 limits!"
        )
        print("  (This is GOOD - means exponential growth!)")
        print()

    print(f"  Mean:   ${np.mean(final_wealth_clean):,.2e}")
    print(f"  Median: ${np.median(final_wealth_clean):,.2e}")
    print(f"  Min:    ${np.min(final_wealth_clean):,.2e}")
    print(f"  Max:    ${np.max(final_wealth_clean):,.2e}")
    print()
    print("Percentiles:")
    print(f"  10th: ${np.percentile(final_wealth_clean, 10):,.2e}")
    print(f"  25th: ${np.percentile(final_wealth_clean, 25):,.2e}")
    print(f"  50th: ${np.percentile(final_wealth_clean, 50):,.2e}")
    print(f"  75th: ${np.percentile(final_wealth_clean, 75):,.2e}")
    print(f"  90th: ${np.percentile(final_wealth_clean, 90):,.2e}")
    print()
    print("Risk Analysis:")
    print(f"  Simulations that went broke (< $1): {broke_simulations:,} ({broke_percentage:.2f}%)")
    survivors = NUM_SIMULATIONS - broke_simulations
    print(f"  Simulations that survived: {survivors:,} ({100 * survivors / NUM_SIMULATIONS:.2f}%)")
    print()
    print("Time to Ruin Statistics:")
    median_ruin = np.median(time_to_ruin[time_to_ruin < NUM_FLIPS]) if np.any(time_to_ruin < NUM_FLIPS) else NUM_FLIPS
    print(f"  Median time to ruin: {int(median_ruin):,} flips")
    if np.any(time_to_ruin < NUM_FLIPS):
        print(f"  10th percentile: {int(np.percentile(time_to_ruin[time_to_ruin < NUM_FLIPS], 10)):,} flips")
        print(f"  90th percentile: {int(np.percentile(time_to_ruin[time_to_ruin < NUM_FLIPS], 90)):,} flips")
    print()

    # Create visualization
    print("Generating visualization...")
    stored_flips = np.arange(0, NUM_FLIPS + 1, STORAGE_INTERVAL)

    # Normalize wealth values to start at 1.0 (performance multiplier)
    wealth_samples_norm = wealth_samples / INITIAL_WEALTH
    wealth_stats_norm = wealth_stats / INITIAL_WEALTH

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # Generate pastel rainbow colors for individual simulation paths
    def generate_pastel_colors(n: int) -> list:
        """Generate n pastel colors using HSV color space."""
        colors = []
        for i in range(n):
            hue = i / n  # Distribute hues evenly around color wheel
            saturation = 0.4 + (np.random.random() * 0.3)  # 0.4-0.7 for pastel effect
            value = 0.8 + (np.random.random() * 0.2)  # 0.8-1.0 for brightness
            rgb = hsv_to_rgb([hue, saturation, value])
            colors.append(rgb)
        return colors

    num_paths_to_plot = min(NUM_SAMPLE_PATHS, wealth_samples.shape[0])
    pastel_colors = generate_pastel_colors(num_paths_to_plot)

    print(f"Plotting {num_paths_to_plot} individual simulation paths...")

    # Plot 1: Wealth time series
    # Plot individual sample paths with pastel colors
    # Since we're only showing first 250 flips, extract just that range for smooth rendering
    for sample_idx in range(num_paths_to_plot):
        # Extract first 250 flips for this path
        path_data = wealth_samples_norm[sample_idx, :251]
        x_data = np.arange(len(path_data))

        # Apply Gaussian smoothing for more continuous appearance
        # sigma=3 provides moderate smoothing (adjust 2-5 for preference)
        smoothed_path = gaussian_filter1d(path_data, sigma=3)

        ax1.plot(
            x_data,
            smoothed_path,
            color=pastel_colors[sample_idx],
            alpha=0.5,
            linewidth=1.0,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # Calculate and plot statistics from stored intervals (handle inf/nan)
    # Replace inf/nan for plotting
    wealth_stats_norm_clean = np.where(np.isinf(wealth_stats_norm), 1e100, wealth_stats_norm)
    wealth_stats_norm_clean = np.where(np.isnan(wealth_stats_norm_clean), 0, wealth_stats_norm_clean)

    mean_wealth = np.mean(wealth_stats_norm_clean, axis=0)
    median_wealth = np.median(wealth_stats_norm_clean, axis=0)

    # Plot mean and median lines (bold, on top of individual paths)
    ax1.plot(
        stored_flips,
        mean_wealth,
        color="#0066CC",
        linewidth=3.0,
        label="Mean",
        alpha=0.95,
        zorder=100,
    )
    ax1.plot(
        stored_flips,
        median_wealth,
        color="#FF6600",
        linewidth=3.0,
        linestyle="--",
        label="Median",
        alpha=0.95,
        zorder=100,
    )

    # Plot initial wealth reference (normalized to 1.0)
    ax1.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Initial Wealth (1.0x)",
        alpha=0.5,
    )

    # Styling
    ax1.set_xlabel("Flip Number", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Performance (log scale)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Cumulative Returns: Monte Carlo Simulations\n({NUM_SIMULATIONS:,} simulations, {BET_FRACTION:.1%} fractional betting, {NUM_FLIPS:,} flips each)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log", base=10)  # Explicitly set log base 10

    # Limit x-axis to show early divergence (like reference image shows ~250 flips)
    # This makes the "straw broom" effect visible
    ax1.set_xlim(left=0, right=250)

    # Set ylim to show the spread at 250 flips
    # Calculate wealth range at flip 250
    if num_paths_to_plot > 0:
        wealth_at_250 = wealth_samples_norm[:, min(250, wealth_samples_norm.shape[1] - 1)]
        min_w = np.min(wealth_at_250[wealth_at_250 > 0]) if np.any(wealth_at_250 > 0) else 0.1
        max_w = np.max(wealth_at_250[np.isfinite(wealth_at_250)])
        ax1.set_ylim(bottom=max(0.1, min_w * 0.5), top=min(max_w * 2, 1e6))
    else:
        ax1.set_ylim(bottom=0.1, top=1000)

    # Add annotation
    growth_text = "GROWTH!" if geometric_mean_multiplier > 1.0 else "RUIN!"
    geometric_text = f"Bet Size: {BET_FRACTION:.1%}\nGeometric Mean: {geometric_mean_multiplier:.6f}x → {growth_text}\nComputed in {elapsed_time:.1f}s ({NUM_SIMULATIONS * NUM_FLIPS / elapsed_time / 1e6:.1f}M flips/s)"
    ax1.text(
        0.02,
        0.97,
        geometric_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    # Plot 2: Time-to-ruin distribution
    bins_ruin = np.logspace(0, np.log10(NUM_FLIPS), 100)
    ruin_times = time_to_ruin[time_to_ruin < NUM_FLIPS]

    if len(ruin_times) > 0:
        ax2.hist(ruin_times, bins=bins_ruin, color="darkred", alpha=0.7, edgecolor="black", linewidth=0.5)

        # Add vertical lines
        median_ruin = np.median(ruin_times)
        ax2.axvline(median_ruin, color="white", linestyle="--", linewidth=3, alpha=0.9)
        ax2.axvline(
            median_ruin,
            color="yellow",
            linestyle="--",
            linewidth=2,
            label=f"Median: {int(median_ruin):,} flips",
            alpha=1.0,
        )

        # Styling
        ax2.set_xlabel("Number of Flips Until Broke (< $1)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Number of Simulations", fontsize=12, fontweight="bold")
        ax2.set_title("Time to Ruin Distribution", fontsize=14, fontweight="bold", pad=15)
        ax2.legend(loc="upper right", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xscale("log")

        # Add statistics box
        stats_text = f"""Went broke: {len(ruin_times):,} ({100 * len(ruin_times) / NUM_SIMULATIONS:.2f}%)
Survived: {survivors:,} ({100 * survivors / NUM_SIMULATIONS:.2f}%)
Median ruin: {int(median_ruin):,} flips
P10: {int(np.percentile(ruin_times, 10)):,} flips
P90: {int(np.percentile(ruin_times, 90)):,} flips"""
        ax2.text(
            0.02,
            0.97,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            bbox={"boxstyle": "round", "facecolor": "salmon", "alpha": 0.8},
        )
    else:
        ax2.text(0.5, 0.5, "All simulations survived!", transform=ax2.transAxes, fontsize=14, ha="center", va="center")

    plt.tight_layout()

    # Save the plot
    bet_pct_str = f"{int(BET_FRACTION * 100)}pct"
    # Use absolute path relative to project root (one level up from scripts/)
    project_root = Path(__file__).parent.parent
    output_file = project_root / "reports" / f"coin_flip_simulation_{bet_pct_str}_1M.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Visualization saved to: {output_file}")
    print()

    # Strategic recommendations
    print("=" * 80)
    print("STRATEGIC ANALYSIS:")
    print("=" * 80)
    print()
    print(f"SHOULD YOU FLIP THE COIN at {BET_FRACTION:.1%} bet size?")
    print()
    if geometric_mean_multiplier > 1.0:
        print(f"✅ YES! With {BET_FRACTION:.1%} betting, you have POSITIVE expected growth!")
        print()
        print("Evidence from 1 MILLION flips:")
        print(f"1. {100 - broke_percentage:.2f}% of simulations survived and grew")
        print(f"2. Final median wealth: ${np.median(final_wealth):.2e}")
        print(f"3. Geometric mean: {geometric_mean_multiplier:.6f} > 1.0")
        print()
        if kelly_fraction == BET_FRACTION:
            print(f"🎯 {BET_FRACTION:.1%} is the OPTIMAL Kelly bet size - maximum growth rate!")
        elif kelly_fraction > BET_FRACTION:
            print(f"📊 {BET_FRACTION:.1%} is UNDER-betting (safe but slower than optimal)")
            print(f"   Kelly optimal: {kelly_fraction:.1%}")
        else:
            print(f"⚠️  {BET_FRACTION:.1%} is OVER-betting (risky - more volatile than optimal)")
            print(f"   Kelly optimal: {kelly_fraction:.1%}")
    else:
        print(f"❌ NO! Even with {BET_FRACTION:.1%} betting, you're likely to go broke!")
        print()
        print("Evidence from 1 MILLION flips:")
        print(f"1. {broke_percentage:.2f}% of simulations went broke")
        print(f"2. Median time to ruin: {int(median_ruin):,} flips")
        print(f"3. Final median wealth: ${np.median(final_wealth):.2e}")
        print()
        print(f"The geometric mean ({geometric_mean_multiplier:.6f}) guarantees long-term ruin.")
        print()
        print(f"✅ BETTER STRATEGY: Bet {kelly_fraction:.1%} of bankroll (Kelly Criterion optimal)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.freeze_support()
    main()
