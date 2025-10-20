#!/usr/bin/env python3
"""
Aave V3 Lending Rates Collector
Collects lending and borrowing rate data from Aave V3 on Arbitrum
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import polars as pl
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from query_aave_subgraph import AaveV3SubgraphClient

# Load environment variables from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class AaveLendingRatesCollector:
    """Collector for Aave V3 lending rates data"""

    def __init__(self, output_dir: Path = Path("research/risk_free_rate/data")):
        """
        Initialize the lending rates collector

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = AaveV3SubgraphClient()

    def fetch_market_rates(self, limit: int = 100) -> List[Dict]:
        """
        Fetch current lending and borrowing rates for all markets

        Args:
            limit: Number of markets to fetch

        Returns:
            List of market data with rates
        """
        query = """
        query GetMarketRates($limit: Int!) {
            markets(first: $limit, where: { isActive: true }) {
                id
                name
                isActive
                canBorrowFrom
                canUseAsCollateral
                inputToken {
                    id
                    symbol
                    name
                    decimals
                }
                rates {
                    id
                    rate
                    side
                    type
                }
                totalValueLockedUSD
                totalDepositBalanceUSD
                totalBorrowBalanceUSD
                inputTokenBalance
                inputTokenPriceUSD
                maximumLTV
                liquidationThreshold
                liquidationPenalty
            }
        }
        """

        result = self.client.query(query, {"limit": limit})

        if "errors" in result:
            logger.error(f"GraphQL errors: {result['errors']}")
            return []

        return result.get("data", {}).get("markets", [])

    def parse_market_rates(self, market: Dict) -> Dict:
        """
        Parse market data to extract key rate information

        Args:
            market: Raw market data from GraphQL

        Returns:
            Parsed market data dictionary
        """
        # Extract rates by type
        rates = {}
        for rate in market.get("rates", []):
            key = f"{rate['side'].lower()}_{rate['type'].lower()}"
            rates[key] = float(rate["rate"])

        # Get token info
        token = market.get("inputToken", {})

        # Parse the data
        parsed = {
            "market_id": market["id"],
            "market_name": market["name"],
            "symbol": token.get("symbol", ""),
            "token_address": token.get("id", ""),
            "decimals": int(token.get("decimals", 0)),
            "is_active": market["isActive"],
            "can_borrow": market["canBorrowFrom"],
            "can_use_collateral": market["canUseAsCollateral"],

            # Rates (already in percentage format)
            "lender_variable_apr": rates.get("lender_variable", 0.0),
            "borrower_variable_apr": rates.get("borrower_variable", 0.0),
            "borrower_stable_apr": rates.get("borrower_stable", 0.0),

            # Calculate utilization rate
            "tvl_usd": float(market.get("totalValueLockedUSD", 0)),
            "total_deposits_usd": float(market.get("totalDepositBalanceUSD", 0)),
            "total_borrows_usd": float(market.get("totalBorrowBalanceUSD", 0)),
            "utilization_rate": self._calculate_utilization(
                float(market.get("totalBorrowBalanceUSD", 0)),
                float(market.get("totalDepositBalanceUSD", 0))
            ),

            # Token metrics
            "input_token_balance": float(market.get("inputTokenBalance", 0)) / (10 ** int(token.get("decimals", 0))),
            "token_price_usd": float(market.get("inputTokenPriceUSD", 0)),

            # Risk parameters
            "max_ltv": float(market.get("maximumLTV", 0)),
            "liquidation_threshold": float(market.get("liquidationThreshold", 0)),
            "liquidation_penalty": float(market.get("liquidationPenalty", 0)),

            # Metadata
            "collected_at": datetime.now(timezone.utc),
        }

        return parsed

    def _calculate_utilization(self, total_borrowed: float, total_supplied: float) -> float:
        """Calculate utilization rate as percentage"""
        if total_supplied == 0:
            return 0.0
        return (total_borrowed / total_supplied) * 100

    def collect_current_rates(self) -> pl.DataFrame:
        """
        Collect current lending rates from all active markets

        Returns:
            DataFrame with current rate data
        """
        logger.info("Fetching current market rates from Aave V3 Arbitrum...")

        # Fetch raw data
        markets = self.fetch_market_rates(limit=1000)
        logger.info(f"Fetched {len(markets)} markets")

        if not markets:
            logger.error("No market data received")
            return pl.DataFrame()

        # Parse all markets
        parsed_markets = []
        for market in markets:
            try:
                parsed = self.parse_market_rates(market)
                parsed_markets.append(parsed)
            except Exception as e:
                logger.error(f"Error parsing market {market.get('id', 'unknown')}: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_markets)} markets")

        # Convert to DataFrame
        df = pl.DataFrame(parsed_markets)

        # Sort by TVL descending
        df = df.sort("tvl_usd", descending=True)

        return df

    def filter_stable_assets(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter for stable assets (stablecoins and major assets)

        Args:
            df: DataFrame with market data

        Returns:
            Filtered DataFrame
        """
        # Define stable and major assets
        stable_assets = ["USDC", "USDT", "DAI", "FRAX", "LUSD", "TUSD", "GUSD", "USDP"]
        major_assets = ["WETH", "ETH", "WBTC", "BTC", "stETH", "wstETH"]

        target_assets = stable_assets + major_assets

        # Filter
        df_filtered = df.filter(pl.col("symbol").is_in(target_assets))

        logger.info(f"Filtered to {len(df_filtered)} stable/major assets")

        return df_filtered

    def calculate_risk_free_rate(self, df: pl.DataFrame) -> Dict:
        """
        Calculate composite risk-free rate from stable asset lending rates

        Args:
            df: DataFrame with market data

        Returns:
            Dictionary with risk-free rate metrics
        """
        # Focus on stablecoins for risk-free rate
        stablecoins = ["USDC", "USDT", "DAI", "FRAX", "LUSD"]

        df_stable = df.filter(
            pl.col("symbol").is_in(stablecoins) &
            pl.col("is_active") &
            (pl.col("tvl_usd") > 1000000)  # Minimum $1M TVL
        )

        if len(df_stable) == 0:
            logger.warning("No stable assets found for risk-free rate calculation")
            return {}

        # Calculate weighted average by TVL
        total_tvl = df_stable["tvl_usd"].sum()

        # Weighted average supply rate
        weighted_supply_rate = (
            (df_stable["lender_variable_apr"] * df_stable["tvl_usd"]).sum() / total_tvl
        )

        # Simple average for comparison
        simple_avg_supply_rate = df_stable["lender_variable_apr"].mean()

        # Get individual rates
        individual_rates = {}
        for row in df_stable.iter_rows(named=True):
            individual_rates[row["symbol"]] = {
                "supply_apr": row["lender_variable_apr"],
                "tvl_usd": row["tvl_usd"],
                "utilization": row["utilization_rate"]
            }

        metrics = {
            "weighted_avg_supply_apr": float(weighted_supply_rate),
            "simple_avg_supply_apr": float(simple_avg_supply_rate),
            "total_tvl_usd": float(total_tvl),
            "num_assets": len(df_stable),
            "assets_included": df_stable["symbol"].to_list(),
            "individual_rates": individual_rates,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return metrics

    def save_to_parquet(self, df: pl.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to Parquet file

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        df.write_parquet(
            output_path,
            compression="snappy",
            statistics=True,
            use_pyarrow=True
        )

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved {len(df):,} records to {output_path} ({file_size_mb:.2f} MB)")

        return output_path

    def close(self):
        """Clean up resources"""
        self.client.close()


def main():
    """Main entry point"""
    logger.info("=== Aave V3 Lending Rates Collection ===")

    collector = AaveLendingRatesCollector()

    try:
        # Collect current rates
        df = collector.collect_current_rates()

        if df.is_empty():
            logger.error("No data collected")
            sys.exit(1)

        # Display summary
        logger.info("\n=== Data Summary ===")
        logger.info(f"Total markets: {len(df)}")
        logger.info(f"Active markets: {df.filter(pl.col('is_active')).height}")
        logger.info(f"Total TVL: ${df['tvl_usd'].sum():,.2f}")

        # Show top assets by TVL
        logger.info("\n=== Top 10 Assets by TVL ===")
        top_assets = df.head(10)
        for row in top_assets.iter_rows(named=True):
            logger.info(
                f"  {row['symbol']:10s} - "
                f"Supply APR: {row['lender_variable_apr']:6.2f}% | "
                f"Borrow APR: {row['borrower_variable_apr']:6.2f}% | "
                f"Util: {row['utilization_rate']:5.1f}% | "
                f"TVL: ${row['tvl_usd']:,.0f}"
            )

        # Filter stable assets
        df_stable = collector.filter_stable_assets(df)

        # Calculate risk-free rate
        risk_free_metrics = collector.calculate_risk_free_rate(df)

        if risk_free_metrics:
            logger.info("\n=== Risk-Free Rate Metrics ===")
            logger.info(f"Weighted Average Supply APR: {risk_free_metrics['weighted_avg_supply_apr']:.3f}%")
            logger.info(f"Simple Average Supply APR: {risk_free_metrics['simple_avg_supply_apr']:.3f}%")
            logger.info(f"Based on {risk_free_metrics['num_assets']} stablecoin(s)")
            logger.info(f"Total TVL: ${risk_free_metrics['total_tvl_usd']:,.0f}")

            logger.info("\nIndividual Stablecoin Rates:")
            for asset, rates in risk_free_metrics.get("individual_rates", {}).items():
                logger.info(
                    f"  {asset}: {rates['supply_apr']:.3f}% "
                    f"(TVL: ${rates['tvl_usd']:,.0f}, Util: {rates['utilization']:.1f}%)"
                )

        # Save data
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save all markets
        output_file = f"aave_v3_rates_{timestamp}.parquet"
        collector.save_to_parquet(df, output_file)

        # Save stable assets separately
        if not df_stable.is_empty():
            stable_file = f"aave_v3_stable_rates_{timestamp}.parquet"
            collector.save_to_parquet(df_stable, stable_file)

        # Save risk-free rate metrics
        if risk_free_metrics:
            metrics_file = collector.output_dir / f"risk_free_rate_metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump(risk_free_metrics, f, indent=2)
            logger.info(f"✓ Saved risk-free rate metrics to {metrics_file}")

        logger.info("\n✅ Data collection complete!")

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        collector.close()


if __name__ == "__main__":
    main()