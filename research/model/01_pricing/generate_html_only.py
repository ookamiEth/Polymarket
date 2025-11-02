#!/usr/bin/env python3
"""
Quick script to generate HTML report from existing plot files.
Since all PNG files already exist, this just creates the HTML without re-running visualizations.
"""

from pathlib import Path
from datetime import datetime

def main():
    # Portable path (works regardless of CWD)
    plots_dir = Path(__file__).parent.parent / "results" / "plots"
    output_file = plots_dir / "report.html"

    # Collect all plot files
    plots = {
        "time_series": list((plots_dir / "time_series").glob("*.png")),
        "calibration": list((plots_dir / "calibration").glob("*.png")),
        "predictions": list((plots_dir / "predictions").glob("*.png")),
        "diagnostics": sorted((plots_dir / "diagnostics").glob("*.png")),
        "trading": list((plots_dir / "trading").glob("*.png")),
        "simulation": list((plots_dir / "simulation").glob("*.png")),
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LightGBM Model Visualization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #0a0a0a;
            color: #e0e0e0;
        }}
        h1 {{
            color: #00D4FF;
            border-bottom: 3px solid #00D4FF;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #FF00FF;
            margin-top: 40px;
            border-bottom: 2px solid #FF00FF;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #00FF88;
            margin-top: 30px;
        }}
        .plot-container {{
            margin: 20px 0;
            background-color: #1a1a1a;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 212, 255, 0.1);
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #333;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>🤖 LightGBM Residual Prediction Model - Visualization Report</h1>
    <p style="color: #888; font-size: 14px;">
        Generated: {timestamp}<br>
        Project: BTC 15-minute Options Pricing<br>
        Total Plots: {sum(len(v) for v in plots.values())}
    </p>

    <h2>📈 Time-Series Analysis</h2>
    <p>Temporal analysis of model performance to detect degradation and regime-specific failures.</p>
"""

    # Time-series plots
    for plot_path in plots["time_series"]:
        name = plot_path.stem.replace("_", " ").title()
        rel_path = plot_path.relative_to(plots_dir)
        html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    # Calibration plots
    if plots["calibration"]:
        html += """
    <h2>🎯 Calibration Analysis</h2>
    <p>Model calibration across different market regimes.</p>
"""
        for plot_path in plots["calibration"]:
            name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.relative_to(plots_dir)
            html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    # Prediction distribution
    if plots["predictions"]:
        html += """
    <h2>📊 Prediction Distribution</h2>
    <p>Distribution analysis of model predictions vs baseline.</p>
"""
        for plot_path in plots["predictions"]:
            name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.relative_to(plots_dir)
            html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    # Diagnostics
    if plots["diagnostics"]:
        html += """
    <h2>🔬 Diagnostics & SHAP Analysis</h2>
    <p>Model performance diagnostics, feature importance, and interpretability analysis.</p>
"""
        for plot_path in plots["diagnostics"]:
            name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.relative_to(plots_dir)
            html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    # Trading
    if plots["trading"]:
        html += """
    <h2>💹 Trading Simulation</h2>
    <p>Backtested trading strategy performance and P&L analysis.</p>
"""
        for plot_path in plots["trading"]:
            name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.relative_to(plots_dir)
            html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    # Simulation/Bootstrap
    if plots["simulation"]:
        html += """
    <h2>📊 Uncertainty Quantification</h2>
    <p>Bootstrap confidence intervals for model performance metrics.</p>
"""
        for plot_path in plots["simulation"]:
            name = plot_path.stem.replace("_", " ").title()
            rel_path = plot_path.relative_to(plots_dir)
            html += f"""
    <div class="plot-container">
        <h3>{name}</h3>
        <img src="{rel_path}" alt="{name}">
    </div>
"""

    html += """
    <footer>
        <p>Generated by LightGBM Visualization Suite</p>
        <p>For questions, contact the quant research team</p>
    </footer>
</body>
</html>
"""

    # Write HTML
    with open(output_file, "w") as f:
        f.write(html)

    print(f"✓ Generated complete HTML report: {output_file}")
    print(f"  Total plots included: {sum(len(v) for v in plots.values())}")
    print(f"  - Time-series: {len(plots['time_series'])}")
    print(f"  - Calibration: {len(plots['calibration'])}")
    print(f"  - Predictions: {len(plots['predictions'])}")
    print(f"  - Diagnostics: {len(plots['diagnostics'])}")
    print(f"  - Trading: {len(plots['trading'])}")
    print(f"  - Simulation: {len(plots['simulation'])}")

if __name__ == "__main__":
    main()
