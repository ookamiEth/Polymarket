#!/usr/bin/env python3
"""
Extract and analyze feature importance from multi-horizon LightGBM models.
Identifies features with low importance that are candidates for pruning.
"""

import lightgbm as lgb
import pandas as pd
from pathlib import Path

def extract_feature_importance(model_path: str) -> pd.DataFrame:
    """Extract feature importance from a LightGBM model."""

    # Load the model
    model = lgb.Booster(model_file=model_path)

    # Get feature importance (gain-based)
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')

    # Get feature names
    feature_names = model.feature_name()

    # Create dataframe
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance_gain,
        'importance_split': importance_split
    })

    # Calculate relative importance
    df['importance_gain_pct'] = 100 * df['importance_gain'] / df['importance_gain'].sum()
    df['importance_split_pct'] = 100 * df['importance_split'] / df['importance_split'].sum()

    return df.sort_values('importance_gain_pct', ascending=False)


def main():
    """Analyze feature importance across all models."""

    model_dir = Path("/home/ubuntu/Polymarket/research/model/results/multi_horizon/models")

    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS FOR V4 PRUNING")
    print("=" * 80)

    # Process each model
    models = {
        'near': model_dir / "lightgbm_near.txt",
        'mid': model_dir / "lightgbm_mid.txt",
        'far': model_dir / "lightgbm_far.txt"
    }

    all_importance = {}

    for name, model_path in models.items():
        print(f"\n{name.upper()} Model ({model_path.name}):")
        print("-" * 40)

        df = extract_feature_importance(str(model_path))
        all_importance[name] = df

        # Show top 10 features
        print("\nTop 10 Features by Gain:")
        for i, row in df.head(10).iterrows():
            print(f"  {row['feature']:30s} {row['importance_gain_pct']:6.2f}%")

        # Show bottom 20 features (pruning candidates)
        print("\nBottom 20 Features by Gain (Pruning Candidates):")
        for i, row in df.tail(20).iterrows():
            print(f"  {row['feature']:30s} {row['importance_gain_pct']:6.2f}%")

        # Count features with <1% importance
        low_importance = df[df['importance_gain_pct'] < 1.0]
        print(f"\nFeatures with <1% importance: {len(low_importance)} / {len(df)}")

        # Count features with zero importance
        zero_importance = df[df['importance_gain'] == 0]
        print(f"Features with ZERO importance: {len(zero_importance)}")
        if len(zero_importance) > 0:
            print("  Zero-importance features:", ', '.join(zero_importance['feature'].tolist()))

    # Find features consistently low across all models
    print("\n" + "=" * 80)
    print("CROSS-MODEL ANALYSIS")
    print("=" * 80)

    # Merge importance scores
    merged = all_importance['near'][['feature', 'importance_gain_pct']].rename(
        columns={'importance_gain_pct': 'near_pct'}
    )
    merged = merged.merge(
        all_importance['mid'][['feature', 'importance_gain_pct']].rename(
            columns={'importance_gain_pct': 'mid_pct'}
        ),
        on='feature'
    )
    merged = merged.merge(
        all_importance['far'][['feature', 'importance_gain_pct']].rename(
            columns={'importance_gain_pct': 'far_pct'}
        ),
        on='feature'
    )

    # Calculate average importance
    merged['avg_importance'] = (merged['near_pct'] + merged['mid_pct'] + merged['far_pct']) / 3
    merged = merged.sort_values('avg_importance', ascending=False)

    # Features consistently low across all models
    low_across_all = merged[
        (merged['near_pct'] < 1.0) &
        (merged['mid_pct'] < 1.0) &
        (merged['far_pct'] < 1.0)
    ]

    print(f"\nFeatures with <1% importance in ALL models: {len(low_across_all)}")
    if len(low_across_all) > 0:
        print("\nConsistently Low-Importance Features (Safe to Prune):")
        for i, row in low_across_all.iterrows():
            print(f"  {row['feature']:30s} Avg: {row['avg_importance']:5.2f}% "
                  f"(Near: {row['near_pct']:5.2f}%, Mid: {row['mid_pct']:5.2f}%, Far: {row['far_pct']:5.2f}%)")

    # Save results
    output_file = "/home/ubuntu/Polymarket/research/model/feature_importance_analysis_v4.csv"
    merged.to_csv(output_file, index=False)
    print(f"\n✅ Saved detailed analysis to: {output_file}")

    # Summary recommendations
    print("\n" + "=" * 80)
    print("PRUNING RECOMMENDATIONS")
    print("=" * 80)

    # Features with average importance < 0.5%
    very_low = merged[merged['avg_importance'] < 0.5]
    print(f"\n1. Features with <0.5% average importance (HIGH confidence for removal): {len(very_low)}")
    if len(very_low) > 0:
        print("   ", ', '.join(very_low['feature'].tolist()[:10]))
        if len(very_low) > 10:
            print(f"   ... and {len(very_low) - 10} more")

    # Features with average importance 0.5-1.0%
    moderate_low = merged[(merged['avg_importance'] >= 0.5) & (merged['avg_importance'] < 1.0)]
    print(f"\n2. Features with 0.5-1.0% average importance (MODERATE confidence): {len(moderate_low)}")
    if len(moderate_low) > 0:
        print("   ", ', '.join(moderate_low['feature'].tolist()[:10]))
        if len(moderate_low) > 10:
            print(f"   ... and {len(moderate_low) - 10} more")

    print("\n" + "=" * 80)
    print(f"TOTAL PRUNING CANDIDATES: {len(very_low) + len(moderate_low)} features")
    print("=" * 80)


if __name__ == "__main__":
    main()