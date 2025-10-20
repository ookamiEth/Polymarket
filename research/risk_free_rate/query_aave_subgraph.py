#!/usr/bin/env python3
"""
Aave V3 Arbitrum Subgraph Query Tool
Queries the Aave V3 lending protocol data from The Graph
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in parent directory (project root)
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class AaveV3SubgraphClient:
    """Client for querying Aave V3 Arbitrum subgraph"""

    # Aave V3 Arbitrum Subgraph ID
    AAVE_V3_ARBITRUM_SUBGRAPH_ID = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"

    def __init__(self, api_key: Optional[str] = None, subgraph_id: Optional[str] = None):
        """
        Initialize the Aave V3 subgraph client

        Args:
            api_key: Graph API key (defaults to env variable)
            subgraph_id: Subgraph deployment ID (defaults to Aave V3 Arbitrum)
        """
        self.api_key = api_key or os.getenv("GRAPH_API_KEY")
        self.subgraph_id = subgraph_id or self.AAVE_V3_ARBITRUM_SUBGRAPH_ID

        if not self.api_key:
            raise ValueError("GRAPH_API_KEY not provided or found in .env")

        self.endpoint = f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/{self.subgraph_id}"
        self.session = httpx.Client(timeout=30.0)

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the subgraph

        Args:
            query: GraphQL query string
            variables: Optional variables for the query

        Returns:
            Dictionary containing the query results

        Raises:
            httpx.HTTPError: If the request fails
        """
        headers = {
            "Content-Type": "application/json",
        }

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = self.session.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Check for GraphQL errors
            if "errors" in result:
                print("GraphQL Errors:", file=sys.stderr)
                for error in result["errors"]:
                    print(f"  - {error.get('message', 'Unknown error')}", file=sys.stderr)

            return result

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
            raise
        except httpx.RequestError as e:
            print(f"Request Error: {e}", file=sys.stderr)
            raise
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}", file=sys.stderr)
            raise

    def get_reserves(self, first: int = 100, skip: int = 0, active_only: bool = True) -> List[Dict]:
        """
        Get all reserves (lending markets) data

        Args:
            first: Number of reserves to fetch
            skip: Number of reserves to skip (pagination)
            active_only: Only return active reserves

        Returns:
            List of reserve data
        """
        where_clause = "where: { isActive: true }" if active_only else ""

        query = f"""
        query GetReserves($first: Int!, $skip: Int!) {{
            reserves(first: $first, skip: $skip, {where_clause}) {{
                id
                underlyingAsset
                symbol
                name
                decimals
                liquidityRate
                variableBorrowRate
                stableBorrowRate
                utilizationRate
                totalLiquidity
                availableLiquidity
                totalCurrentVariableDebt
                totalCurrentStableDebt
                reserveFactor
                baseLTVasCollateral
                reserveLiquidationThreshold
                reserveLiquidationBonus
                borrowingEnabled
                usageAsCollateralEnabled
                stableBorrowRateEnabled
                isActive
                isFrozen
                lastUpdateTimestamp
            }}
        }}
        """

        result = self.query(query, {"first": first, "skip": skip})
        return result.get("data", {}).get("reserves", [])

    def get_reserve_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get data for a specific reserve by symbol

        Args:
            symbol: Asset symbol (e.g., "USDC", "ETH")

        Returns:
            Reserve data or None if not found
        """
        query = """
        query GetReserveBySymbol($symbol: String!) {
            reserves(where: { symbol: $symbol }) {
                id
                underlyingAsset
                symbol
                name
                decimals
                liquidityRate
                variableBorrowRate
                stableBorrowRate
                utilizationRate
                totalLiquidity
                availableLiquidity
                totalCurrentVariableDebt
                totalCurrentStableDebt
                lastUpdateTimestamp
            }
        }
        """

        result = self.query(query, {"symbol": symbol})
        reserves = result.get("data", {}).get("reserves", [])
        return reserves[0] if reserves else None

    def get_historical_rates(
        self, reserve_id: str, start_timestamp: int, end_timestamp: Optional[int] = None, limit: int = 1000
    ) -> List[Dict]:
        """
        Get historical rate data for a reserve

        Args:
            reserve_id: Reserve ID (contract address)
            start_timestamp: Start timestamp (Unix)
            end_timestamp: End timestamp (Unix), defaults to now
            limit: Maximum number of records

        Returns:
            List of historical rate records
        """
        end_timestamp = end_timestamp or int(datetime.now(timezone.utc).timestamp())

        query = """
        query GetHistoricalRates($reserve: String!, $start: Int!, $end: Int!, $limit: Int!) {
            reserveParamsHistoryItems(
                first: $limit,
                where: {
                    reserve: $reserve,
                    timestamp_gte: $start,
                    timestamp_lte: $end
                },
                orderBy: timestamp,
                orderDirection: desc
            ) {
                id
                reserve {
                    symbol
                    decimals
                }
                liquidityRate
                variableBorrowRate
                stableBorrowRate
                utilizationRate
                timestamp
                totalLiquidity
                availableLiquidity
                totalCurrentVariableDebt
                totalCurrentStableDebt
            }
        }
        """

        result = self.query(
            query, {"reserve": reserve_id, "start": start_timestamp, "end": end_timestamp, "limit": limit}
        )
        return result.get("data", {}).get("reserveParamsHistoryItems", [])

    def get_user_position(self, user_address: str) -> Optional[Dict]:
        """
        Get user position data

        Args:
            user_address: Ethereum address of the user

        Returns:
            User position data or None if not found
        """
        query = """
        query GetUserPosition($user: String!) {
            user(id: $user) {
                id
                borrowedReservesCount
                reserves {
                    id
                    reserve {
                        symbol
                        decimals
                    }
                    currentATokenBalance
                    currentVariableDebt
                    currentStableDebt
                    currentTotalDebt
                    liquidityRate
                    stableBorrowRate
                    variableBorrowIndex
                    lastUpdateTimestamp
                    usageAsCollateralEnabledOnUser
                }
            }
        }
        """

        result = self.query(query, {"user": user_address.lower()})
        return result.get("data", {}).get("user")

    def get_protocol_stats(self) -> Dict:
        """
        Get protocol-level statistics

        Returns:
            Protocol statistics
        """
        query = """
        query GetProtocolStats {
            protocols {
                id
                pools {
                    id
                    totalValueLockedUSD
                    totalDepositBalanceUSD
                    totalBorrowBalanceUSD
                    inputTokens {
                        id
                        symbol
                        decimals
                    }
                }
            }
        }
        """

        result = self.query(query)
        protocols = result.get("data", {}).get("protocols", [])
        return protocols[0] if protocols else {}

    def get_metadata(self) -> Dict:
        """
        Get subgraph metadata (block number, etc.)

        Returns:
            Metadata dictionary
        """
        query = """
        query GetMetadata {
            _meta {
                block {
                    number
                    hash
                    timestamp
                }
                deployment
                hasIndexingErrors
            }
        }
        """

        result = self.query(query)
        return result.get("data", {}).get("_meta", {})

    def close(self):
        """Close the HTTP session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Helper functions for data conversion
def ray_to_percentage(ray_value: int) -> float:
    """
    Convert RAY units (10^27) to percentage

    Args:
        ray_value: Value in RAY units

    Returns:
        Percentage value
    """
    if ray_value is None:
        return 0.0
    return float(ray_value) / 10**27 * 100


def format_token_amount(amount: int, decimals: int) -> float:
    """
    Convert raw token amount to human-readable format

    Args:
        amount: Raw token amount
        decimals: Token decimals

    Returns:
        Formatted amount
    """
    if amount is None:
        return 0.0
    return float(amount) / 10**decimals


def apr_to_apy(apr: float, compounds_per_year: int = 365) -> float:
    """
    Convert APR to APY with daily compounding

    Args:
        apr: Annual Percentage Rate
        compounds_per_year: Compounding frequency

    Returns:
        Annual Percentage Yield
    """
    if apr == 0:
        return 0.0
    return (1 + apr / compounds_per_year) ** compounds_per_year - 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Query Aave V3 Arbitrum subgraph for lending/borrowing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get all active reserves
  python query_aave_subgraph.py reserves

  # Get specific asset data
  python query_aave_subgraph.py reserve --symbol USDC

  # Get protocol statistics
  python query_aave_subgraph.py stats

  # Get subgraph metadata
  python query_aave_subgraph.py metadata

  # Custom GraphQL query
  python query_aave_subgraph.py query '{reserves(first: 5) { symbol liquidityRate }}'
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Reserves command
    reserves_parser = subparsers.add_parser("reserves", help="Get all reserves data")
    reserves_parser.add_argument("--limit", type=int, default=100, help="Number of reserves to fetch")
    reserves_parser.add_argument("--all", action="store_true", help="Include inactive reserves")

    # Reserve command
    reserve_parser = subparsers.add_parser("reserve", help="Get specific reserve data")
    reserve_parser.add_argument("--symbol", required=True, help="Asset symbol (e.g., USDC)")

    # Stats command
    subparsers.add_parser("stats", help="Get protocol statistics")

    # Metadata command
    subparsers.add_parser("metadata", help="Get subgraph metadata")

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute custom GraphQL query")
    query_parser.add_argument("graphql", help="GraphQL query string")
    query_parser.add_argument("--variables", help="JSON variables for the query")

    # Output options
    parser.add_argument("--output", "-o", help="Output file path (JSON)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize client
    try:
        client = AaveV3SubgraphClient()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = None

        if args.command == "reserves":
            # Get all reserves
            reserves = client.get_reserves(first=args.limit, active_only=not args.all)

            # Format the data
            formatted_reserves = []
            for reserve in reserves:
                formatted_reserves.append({
                    "symbol": reserve["symbol"],
                    "name": reserve["name"],
                    "supply_rate_apr": ray_to_percentage(int(reserve["liquidityRate"])),
                    "variable_borrow_apr": ray_to_percentage(int(reserve["variableBorrowRate"])),
                    "stable_borrow_apr": ray_to_percentage(int(reserve["stableBorrowRate"])),
                    "utilization_rate": float(reserve.get("utilizationRate", 0)) * 100 if reserve.get("utilizationRate") else 0,
                    "total_supplied": format_token_amount(int(reserve["totalLiquidity"]), int(reserve["decimals"])),
                    "available": format_token_amount(int(reserve["availableLiquidity"]), int(reserve["decimals"])),
                    "is_active": reserve["isActive"],
                    "last_update": datetime.fromtimestamp(int(reserve["lastUpdateTimestamp"]), tz=timezone.utc).isoformat(),
                })

            result = {"reserves": formatted_reserves, "count": len(formatted_reserves)}

        elif args.command == "reserve":
            # Get specific reserve
            reserve = client.get_reserve_by_symbol(args.symbol)
            if reserve:
                result = {
                    "symbol": reserve["symbol"],
                    "name": reserve["name"],
                    "address": reserve["underlyingAsset"],
                    "supply_rate_apr": ray_to_percentage(int(reserve["liquidityRate"])),
                    "variable_borrow_apr": ray_to_percentage(int(reserve["variableBorrowRate"])),
                    "stable_borrow_apr": ray_to_percentage(int(reserve["stableBorrowRate"])),
                    "utilization_rate": float(reserve.get("utilizationRate", 0)) * 100 if reserve.get("utilizationRate") else 0,
                    "total_supplied": format_token_amount(int(reserve["totalLiquidity"]), int(reserve["decimals"])),
                    "available": format_token_amount(int(reserve["availableLiquidity"]), int(reserve["decimals"])),
                    "total_variable_debt": format_token_amount(int(reserve["totalCurrentVariableDebt"]), int(reserve["decimals"])),
                    "total_stable_debt": format_token_amount(int(reserve["totalCurrentStableDebt"]), int(reserve["decimals"])),
                    "last_update": datetime.fromtimestamp(int(reserve["lastUpdateTimestamp"]), tz=timezone.utc).isoformat(),
                }
            else:
                print(f"Reserve with symbol '{args.symbol}' not found", file=sys.stderr)
                sys.exit(1)

        elif args.command == "stats":
            # Get protocol stats
            result = client.get_protocol_stats()

        elif args.command == "metadata":
            # Get metadata
            result = client.get_metadata()

        elif args.command == "query":
            # Custom query
            variables = None
            if args.variables:
                try:
                    variables = json.loads(args.variables)
                except json.JSONDecodeError as e:
                    print(f"Error parsing variables: {e}", file=sys.stderr)
                    sys.exit(1)

            result = client.query(args.graphql, variables)

        # Output results
        if result:
            if args.pretty:
                output = json.dumps(result, indent=2)
            else:
                output = json.dumps(result)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Results saved to {args.output}")
            else:
                print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()