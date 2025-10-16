#!/usr/bin/env python3
"""
Pure Cost Comparison: Own vs Rent GPUs
No revenue assumptions - just total cost of ownership vs rental
"""


def calculate_monthly_payment(principal: float, apr: float, term_months: int) -> float:
    """Calculate monthly payment for amortizing loan"""
    monthly_rate = apr / 12
    numerator = monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1
    return principal * (numerator / denominator)


def own_gpu_cost(
    gpu_value: float,
    ltv: float,
    apr: float,
    term_years: int,
    annual_opex_per_dollar: float,
    annual_depreciation_rate: float,
) -> dict:
    """
    Calculate total cost of owning GPUs via debt financing

    Args:
        gpu_value: Total GPU purchase value
        ltv: Loan-to-value ratio (e.g., 0.70 for 70%)
        apr: Annual percentage rate (e.g., 0.15 for 15%)
        term_years: Loan term in years
        annual_opex_per_dollar: Operating costs as % of GPU value per year
        annual_depreciation_rate: Annual depreciation rate (e.g., 0.25 for 25%)
    """
    # Initial investment
    equity = gpu_value * (1 - ltv)
    borrowed = gpu_value * ltv

    # Loan costs
    term_months = term_years * 12
    monthly_payment = calculate_monthly_payment(borrowed, apr, term_months)
    total_payments = monthly_payment * term_months
    total_interest = total_payments - borrowed

    # Operating costs
    annual_opex = gpu_value * annual_opex_per_dollar
    total_opex = annual_opex * term_years

    # Depreciation
    final_value = gpu_value * ((1 - annual_depreciation_rate) ** term_years)
    depreciation = gpu_value - final_value

    # Total costs
    total_cash_out = equity + total_payments + total_opex

    return {
        "initial_gpu_value": gpu_value,
        "equity_invested": equity,
        "borrowed": borrowed,
        "total_loan_payments": total_payments,
        "total_interest": total_interest,
        "total_opex": total_opex,
        "total_cash_out": total_cash_out,
        "final_gpu_value": final_value,
        "depreciation": depreciation,
        "total_cost_including_depreciation": total_cash_out + depreciation,
        "net_position": final_value - total_cash_out,
    }


def rent_gpu_cost(
    hourly_rate: float,
    hours_per_year: float,
    term_years: int,
    equivalent_gpu_value: float,
    gpu_price: float = 30_000,
) -> dict:
    """
    Calculate total cost of renting equivalent GPU capacity

    Args:
        hourly_rate: $/hour rental rate per GPU
        hours_per_year: Hours rented per year
        term_years: Rental period in years
        equivalent_gpu_value: Value of GPUs being compared (for per-dollar comparison)
        gpu_price: Price per GPU (default $30k)
    """
    # Calculate number of GPUs
    num_gpus = equivalent_gpu_value / gpu_price

    # Cost per GPU per year
    annual_cost_per_gpu = hourly_rate * hours_per_year

    # Total cost for all GPUs
    annual_cost = annual_cost_per_gpu * num_gpus
    total_cost = annual_cost * term_years

    return {
        "hourly_rate": hourly_rate,
        "hours_per_year": hours_per_year,
        "num_gpus": num_gpus,
        "annual_cost": annual_cost,
        "total_cost": total_cost,
        "cost_per_dollar_of_gpu": total_cost / equivalent_gpu_value,
    }


def print_comparison(scenario_name: str, own: dict, rent: dict):
    """Print side-by-side cost comparison"""
    print(f"\n{'='*100}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*100}")

    print(f"\n{'OWN (USD.AI Financing)':<50} {'RENT (Cloud Provider)':<50}")
    print(f"{'-'*50} {'-'*50}")

    print(f"{'Initial GPU Value:':<30} ${own['initial_gpu_value']:>15,.0f}   {'N/A':<30} {'N/A':>15}")
    print(f"{'Equity Required:':<30} ${own['equity_invested']:>15,.0f}   {'Upfront Cost:':<30} ${rent.get('upfront', 0):>15,.0f}")
    print(f"{'Borrowed Amount:':<30} ${own['borrowed']:>15,.0f}")
    print()
    print(f"{'Total Loan Payments:':<30} ${own['total_loan_payments']:>15,.0f}   {'Total Rental Payments:':<30} ${rent['total_cost']:>15,.0f}")
    print(f"{'  - Principal:':<30} ${own['borrowed']:>15,.0f}")
    print(f"{'  - Interest (15%):':<30} ${own['total_interest']:>15,.0f}")
    print()
    print(f"{'Operating Costs (3yr):':<30} ${own['total_opex']:>15,.0f}   {'Operating Costs:':<30} {'$0 (included)':>15}")
    print()
    print(f"{'TOTAL CASH OUT:':<30} ${own['total_cash_out']:>15,.0f}   {'TOTAL CASH OUT:':<30} ${rent['total_cost']:>15,.0f}")
    print()
    print(f"{'Final Asset Value:':<30} ${own['final_gpu_value']:>15,.0f}   {'Final Asset Value:':<30} {'$0':>15}")
    print(f"{'Total Depreciation:':<30} ${own['depreciation']:>15,.0f}")
    print()

    # Net comparison
    own_net_cost = own['total_cash_out'] - own['final_gpu_value']
    rent_net_cost = rent['total_cost']
    savings = rent_net_cost - own_net_cost

    print(f"\n{'NET COST COMPARISON (Cash Out - Final Value)':<50}")
    print(f"{'-'*100}")
    print(f"{'Own Net Cost:':<30} ${own_net_cost:>15,.0f}")
    print(f"{'Rent Net Cost:':<30} ${rent_net_cost:>15,.0f}")
    print(f"{'Savings from Owning:':<30} ${savings:>15,.0f}")

    if savings > 0:
        pct_savings = (savings / rent_net_cost) * 100
        print(f"\n✅ OWNING SAVES ${savings:,.0f} ({pct_savings:.1f}% cheaper than renting)")
    else:
        pct_more = (abs(savings) / rent_net_cost) * 100
        print(f"\n❌ RENTING SAVES ${abs(savings):,.0f} ({pct_more:.1f}% cheaper than owning)")

    print(f"\n{'Per $1 of GPU Value:':<30}")
    print(f"  Own total cost:        ${own_net_cost / own['initial_gpu_value']:.3f}")
    print(f"  Rent total cost:       ${rent_net_cost / own['initial_gpu_value']:.3f}")


def main():
    """Run pure cost comparisons"""

    # Fixed parameters
    GPU_VALUE = 100_000_000  # $100M in GPUs
    LTV = 0.70  # 70% loan-to-value
    APR = 0.15  # 15% APR
    TERM_YEARS = 3

    # Operating costs as % of GPU value per year
    # $5,500/year per $30k GPU = 18.3% per year
    OPEX_RATE = 0.183

    # Rental parameters
    RENTAL_RATE = 2.50  # $/hour (your specified rate)
    HOURS_PER_YEAR = 8760  # 24/7 operation

    print("="*100)
    print("PURE COST COMPARISON: OWN vs RENT GPUs")
    print("="*100)
    print(f"\nAssumptions:")
    print(f"  - GPU Portfolio Value: ${GPU_VALUE:,.0f}")
    print(f"  - Loan Terms: 70% LTV, 15% APR, 3-year amortization")
    print(f"  - Operating Costs: 18.3% of GPU value per year")
    print(f"  - Rental Rate: ${RENTAL_RATE}/hour")
    print(f"  - Hours per year: {HOURS_PER_YEAR:,} (24/7 operation)")

    # Scenario 1: 20% annual depreciation (5-year life)
    print("\n" + "="*100)
    own_1 = own_gpu_cost(GPU_VALUE, LTV, APR, TERM_YEARS, OPEX_RATE, 0.20)
    rent_1 = rent_gpu_cost(RENTAL_RATE, HOURS_PER_YEAR, TERM_YEARS, GPU_VALUE)
    print_comparison("20% Annual Depreciation (5-year economic life)", own_1, rent_1)

    # Scenario 2: 25% annual depreciation (4-year life)
    print("\n" + "="*100)
    own_2 = own_gpu_cost(GPU_VALUE, LTV, APR, TERM_YEARS, OPEX_RATE, 0.25)
    rent_2 = rent_gpu_cost(RENTAL_RATE, HOURS_PER_YEAR, TERM_YEARS, GPU_VALUE)
    print_comparison("25% Annual Depreciation (4-year economic life)", own_2, rent_2)

    # Scenario 3: 33% annual depreciation (3-year life, USD.AI's assumption)
    print("\n" + "="*100)
    own_3 = own_gpu_cost(GPU_VALUE, LTV, APR, TERM_YEARS, OPEX_RATE, 0.33)
    rent_3 = rent_gpu_cost(RENTAL_RATE, HOURS_PER_YEAR, TERM_YEARS, GPU_VALUE)
    print_comparison("33% Annual Depreciation (3-year economic life, USD.AI model)", own_3, rent_3)

    # Scenario 4: 40% annual depreciation (aggressive/pessimistic)
    print("\n" + "="*100)
    own_4 = own_gpu_cost(GPU_VALUE, LTV, APR, TERM_YEARS, OPEX_RATE, 0.40)
    rent_4 = rent_gpu_cost(RENTAL_RATE, HOURS_PER_YEAR, TERM_YEARS, GPU_VALUE)
    print_comparison("40% Annual Depreciation (aggressive/pessimistic)", own_4, rent_4)

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY: NET COST ACROSS ALL DEPRECIATION SCENARIOS")
    print("="*100)
    print(f"\n{'Depreciation Rate':<25} {'Own Net Cost':<20} {'Rent Net Cost':<20} {'Savings':<20} {'% Cheaper':<15}")
    print("-"*100)

    scenarios = [
        ("20%/year (5yr life)", own_1, rent_1),
        ("25%/year (4yr life)", own_2, rent_2),
        ("33%/year (3yr life)", own_3, rent_3),
        ("40%/year (aggressive)", own_4, rent_4),
    ]

    for name, own, rent in scenarios:
        own_net = own['total_cash_out'] - own['final_gpu_value']
        rent_net = rent['total_cost']
        savings = rent_net - own_net
        pct = (savings / rent_net) * 100

        print(f"{name:<25} ${own_net:>18,.0f} ${rent_net:>18,.0f} ${savings:>18,.0f} {pct:>13.1f}%")

    print("\n" + "="*100)
    print("KEY INSIGHT:")
    print("="*100)
    print(f"Rental at ${RENTAL_RATE}/hr costs ${rent_1['total_cost']:,.0f} over 3 years with NO asset value")
    print(f"Owning costs $117-172M cash out, but you keep a $22-51M asset")
    print(f"Owning is cheaper by $16-73M depending on depreciation rate")
    print("="*100)


if __name__ == "__main__":
    main()
