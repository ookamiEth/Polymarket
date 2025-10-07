#!/usr/bin/env python3
"""
API Discovery Script: Finding Polymarket Leaderboard Endpoint

This script tests various potential API endpoints to find how Polymarket
serves leaderboard data for their public leaderboard page.

Date: 2025-10-01
"""

import requests
import json
from datetime import datetime

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_endpoint(url, params=None, description=""):
    """Test a single endpoint and print results"""
    print(f"Testing: {url}")
    if description:
        print(f"Description: {description}")
    if params:
        print(f"Parameters: {json.dumps(params, indent=2)}")

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✓ SUCCESS - Endpoint exists!")
            data = response.json()

            # Print preview of response
            print("\nResponse Preview:")
            json_str = json.dumps(data, indent=2)
            if len(json_str) > 1000:
                print(json_str[:1000] + "\n... [truncated]")
            else:
                print(json_str)

            return True, data
        else:
            print(f"✗ FAILED - HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error}")
            except:
                print(f"Response: {response.text[:200]}")
            return False, None

    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR - {e}")
        return False, None
    finally:
        print("-" * 80)

def main():
    print_section("POLYMARKET LEADERBOARD API DISCOVERY")

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_results": []
    }

    # Known test address from user (@debased)
    test_address = "0x24c8cf69a0e0a17eee21f69d29752bfa32e823e1"

    # =========================================================================
    # Test 1: Direct Leaderboard Endpoints
    # =========================================================================

    print_section("TEST 1: Direct Leaderboard Endpoints")

    leaderboard_urls = [
        ("https://gamma-api.polymarket.com/leaderboard", "Gamma API - direct leaderboard"),
        ("https://gamma-api.polymarket.com/profiles/leaderboard", "Gamma API - profiles leaderboard"),
        ("https://data-api.polymarket.com/leaderboard", "Data API - direct leaderboard"),
        ("https://gamma-api.polymarket.com/users/leaderboard", "Gamma API - users leaderboard"),
        ("https://gamma-api.polymarket.com/leaderboards", "Gamma API - plural leaderboards"),
    ]

    for url, desc in leaderboard_urls:
        success, data = test_endpoint(url, description=desc)
        results["test_results"].append({
            "url": url,
            "description": desc,
            "success": success,
            "has_data": data is not None
        })

    # =========================================================================
    # Test 2: Search API with Profiles
    # =========================================================================

    print_section("TEST 2: Search API with Profile Filtering")

    search_tests = [
        {
            "url": "https://gamma-api.polymarket.com/search",
            "params": {"search_profiles": True, "limit": 10},
            "desc": "Search with profiles enabled"
        },
        {
            "url": "https://gamma-api.polymarket.com/search",
            "params": {"search_profiles": True, "order": "volume", "limit": 10},
            "desc": "Search profiles ordered by volume"
        },
        {
            "url": "https://gamma-api.polymarket.com/search",
            "params": {"q": "", "search_profiles": True, "limit": 10},
            "desc": "Empty search with profiles"
        },
    ]

    for test in search_tests:
        success, data = test_endpoint(
            test["url"],
            params=test["params"],
            description=test["desc"]
        )
        results["test_results"].append({
            "url": test["url"],
            "params": test["params"],
            "description": test["desc"],
            "success": success,
            "has_data": data is not None
        })

    # =========================================================================
    # Test 3: Profile/User Endpoints
    # =========================================================================

    print_section("TEST 3: Individual Profile Endpoints")
    print(f"Test Address: {test_address} (@debased)")

    profile_urls = [
        (f"https://gamma-api.polymarket.com/profiles/{test_address}", "Gamma - profiles/{address}"),
        (f"https://gamma-api.polymarket.com/users/{test_address}", "Gamma - users/{address}"),
        (f"https://gamma-api.polymarket.com/profile/{test_address}", "Gamma - profile/{address}"),
        (f"https://data-api.polymarket.com/profile/{test_address}", "Data API - profile/{address}"),
        (f"https://data-api.polymarket.com/users/{test_address}", "Data API - users/{address}"),
        (f"https://data-api.polymarket.com/users/{test_address}/stats", "Data API - user stats"),
    ]

    for url, desc in profile_urls:
        success, data = test_endpoint(url, description=desc)
        results["test_results"].append({
            "url": url,
            "description": desc,
            "success": success,
            "has_data": data is not None
        })

    # =========================================================================
    # Test 4: Alternative Patterns
    # =========================================================================

    print_section("TEST 4: Alternative Endpoint Patterns")

    alternative_urls = [
        ("https://gamma-api.polymarket.com/stats/leaderboard", "Stats leaderboard"),
        ("https://gamma-api.polymarket.com/rankings", "Rankings"),
        ("https://gamma-api.polymarket.com/top-traders", "Top traders"),
        ("https://data-api.polymarket.com/rankings", "Data API rankings"),
    ]

    for url, desc in alternative_urls:
        success, data = test_endpoint(url, description=desc)
        results["test_results"].append({
            "url": url,
            "description": desc,
            "success": success,
            "has_data": data is not None
        })

    # =========================================================================
    # Summary
    # =========================================================================

    print_section("DISCOVERY SUMMARY")

    successful_endpoints = [r for r in results["test_results"] if r["success"]]

    if successful_endpoints:
        print(f"✓ Found {len(successful_endpoints)} working endpoint(s)!\n")
        for endpoint in successful_endpoints:
            print(f"  • {endpoint['url']}")
            print(f"    {endpoint['description']}\n")
    else:
        print("✗ No leaderboard endpoints found.")
        print("\nNext steps:")
        print("  1. Manually inspect browser network traffic at https://polymarket.com/leaderboard")
        print("  2. Look for XHR/Fetch requests in DevTools Network tab")
        print("  3. Implement fallback: calculate volume from trade history")

    # Save results
    output_file = "/Users/lgierhake/Documents/ETH/BT/top_traders/api_research/discovery_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
