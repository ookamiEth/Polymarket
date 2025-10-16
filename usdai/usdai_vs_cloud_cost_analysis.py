#!/usr/bin/env python3
"""
Cost Comparison: QumulusAI USD.AI Financing vs AWS/GCP Cloud
Analyzes whether it's more cost-effective to own GPUs via debt financing or rent from cloud providers
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LoanTerms:
    """USD.AI loan terms"""

    principal: float  # Amount borrowed
    apr: float  # Annual percentage rate
    term_years: int  # Loan duration
    ltv: float  # Loan-to-value ratio


@dataclass
class GPUEconomics:
    """GPU operational economics"""

    purchase_price: float  # Per GPU
    rental_rate_hour: float  # Revenue per hour
    utilization: float  # Percentage of time rented
    opex_per_gpu_year: float  # Operating costs per year
    depreciation_rate: float  # Annual depreciation rate


def calculate_monthly_payment(principal: float, apr: float, term_months: int) -> float:
    """Calculate monthly payment for amortizing loan"""
    monthly_rate = apr / 12
    if monthly_rate == 0:
        return principal / term_months

    numerator = monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1
    return principal * (numerator / denominator)


def usdai_financing_model(
    gpu_count: int,
    loan_terms: LoanTerms,
    gpu_econ: GPUEconomics,
) -> Dict[str, float]:
    """
    Model QumulusAI's economics with USD.AI financing

    Returns dict with total costs, revenue, profits over loan term
    """
    # Initial investment
    total_gpu_value = gpu_count * gpu_econ.purchase_price
    borrowed_amount = total_gpu_value * loan_terms.ltv
    equity_invested = total_gpu_value * (1 - loan_terms.ltv)

    # Loan payments
    term_months = loan_terms.term_years * 12
    monthly_payment = calculate_monthly_payment(
        borrowed_amount, loan_terms.apr, term_months
    )
    total_loan_payments = monthly_payment * term_months
    total_interest = total_loan_payments - borrowed_amount

    # Revenue over loan term
    hours_per_year = 8760
    annual_revenue_per_gpu = (
        gpu_econ.rental_rate_hour * hours_per_year * gpu_econ.utilization
    )
    annual_revenue_total = annual_revenue_per_gpu * gpu_count
    total_revenue = annual_revenue_total * loan_terms.term_years

    # Operating costs
    annual_opex = gpu_econ.opex_per_gpu_year * gpu_count
    total_opex = annual_opex * loan_terms.term_years

    # Depreciation
    remaining_value_pct = (1 - gpu_econ.depreciation_rate) ** loan_terms.term_years
    gpu_value_at_end = total_gpu_value * remaining_value_pct
    total_depreciation = total_gpu_value - gpu_value_at_end

    # Cash flows
    total_cash_in = total_revenue
    total_cash_out = equity_invested + total_loan_payments + total_opex
    net_cash_flow = total_cash_in - total_cash_out

    # Economic profit (including asset depreciation)
    economic_profit = net_cash_flow - total_depreciation

    # ROI on equity
    roi_on_equity = ((net_cash_flow + gpu_value_at_end) / equity_invested - 1) * 100

    return {
        "total_gpu_value": total_gpu_value,
        "equity_invested": equity_invested,
        "borrowed_amount": borrowed_amount,
        "total_loan_payments": total_loan_payments,
        "total_interest_paid": total_interest,
        "monthly_payment": monthly_payment,
        "total_revenue": total_revenue,
        "annual_revenue": annual_revenue_total,
        "total_opex": total_opex,
        "annual_opex": annual_opex,
        "total_depreciation": total_depreciation,
        "gpu_value_at_end": gpu_value_at_end,
        "net_cash_flow": net_cash_flow,
        "economic_profit": economic_profit,
        "roi_on_equity_pct": roi_on_equity,
        "effective_cost_per_gpu_hour": (total_cash_out - total_cash_in) / (gpu_count * hours_per_year * loan_terms.term_years * gpu_econ.utilization) if total_cash_in < total_cash_out else 0,
    }


def cloud_rental_model(
    gpu_count: int,
    rental_rate_hour: float,
    term_years: int,
    utilization: float,
) -> Dict[str, float]:
    """
    Model costs of renting equivalent GPU capacity from AWS/GCP

    Returns dict with total rental costs
    """
    hours_per_year = 8760
    hours_used_per_year = hours_per_year * utilization
    annual_cost = gpu_count * rental_rate_hour * hours_used_per_year
    total_cost = annual_cost * term_years

    return {
        "annual_cost": annual_cost,
        "total_cost": total_cost,
        "equity_needed": 0,  # No upfront cost
        "effective_rate_per_hour": rental_rate_hour,
    }


def print_comparison(scenario_name: str, own_results: Dict, cloud_results: Dict):
    """Print formatted comparison"""
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")

    print("\n--- USD.AI FINANCING (Own GPUs) ---")
    print(f"Total GPU Purchase Value:    ${own_results['total_gpu_value']:>15,.0f}")
    print(f"Equity Invested (30%):       ${own_results['equity_invested']:>15,.0f}")
    print(f"Borrowed (70%):              ${own_results['borrowed_amount']:>15,.0f}")
    print(f"\nMonthly Loan Payment:        ${own_results['monthly_payment']:>15,.0f}")
    print(f"Total Loan Payments:         ${own_results['total_loan_payments']:>15,.0f}")
    print(f"Total Interest Paid:         ${own_results['total_interest_paid']:>15,.0f}")
    print(f"\nTotal Revenue (3 years):     ${own_results['total_revenue']:>15,.0f}")
    print(f"Total Operating Costs:       ${own_results['total_opex']:>15,.0f}")
    print(f"Total Depreciation:          ${own_results['total_depreciation']:>15,.0f}")
    print(f"GPU Value at End:            ${own_results['gpu_value_at_end']:>15,.0f}")
    print(f"\nNet Cash Flow:               ${own_results['net_cash_flow']:>15,.0f}")
    print(f"Economic Profit:             ${own_results['economic_profit']:>15,.0f}")
    print(f"ROI on Equity:               {own_results['roi_on_equity_pct']:>15.1f}%")

    print("\n--- AWS/GCP CLOUD RENTAL ---")
    print(f"Annual Rental Cost:          ${cloud_results['annual_cost']:>15,.0f}")
    print(f"Total Cost (3 years):        ${cloud_results['total_cost']:>15,.0f}")
    print(f"Upfront Equity Needed:       ${cloud_results['equity_needed']:>15,.0f}")

    print("\n--- COMPARISON ---")
    total_cash_out_own = own_results['equity_invested'] + own_results['total_loan_payments'] + own_results['total_opex']
    savings = cloud_results['total_cost'] - (total_cash_out_own - own_results['total_revenue'])
    print(f"Total Cash Out (Own):        ${total_cash_out_own:>15,.0f}")
    print(f"Total Cash In (Own):         ${own_results['total_revenue']:>15,.0f}")
    print(f"Net Cash (Own):              ${own_results['net_cash_flow']:>15,.0f}")
    print(f"Total Cost (Cloud):          ${cloud_results['total_cost']:>15,.0f}")
    print(f"\nOwning vs Renting Savings:   ${savings:>15,.0f}")

    if savings > 0:
        print(f"✅ OWNING IS CHEAPER by ${savings:,.0f} over 3 years")
    else:
        print(f"❌ RENTING IS CHEAPER by ${abs(savings):,.0f} over 3 years")

    # Break-even analysis
    if own_results['total_revenue'] > 0:
        own_effective_rate = (own_results['total_loan_payments'] + own_results['total_opex']) / (own_results['total_revenue'] / cloud_results['effective_rate_per_hour'])
        print(f"\nEffective hourly cost (Own): ${own_effective_rate:.2f}/hr")
        print(f"Cloud rental rate:           ${cloud_results['effective_rate_per_hour']:.2f}/hr")


def main():
    """Run cost comparison scenarios"""

    # Standard USD.AI loan terms (from docs)
    loan_terms = LoanTerms(
        principal=0,  # Will be calculated based on GPU value
        apr=0.15,  # 15% APR (mid-teens)
        term_years=3,
        ltv=0.70,  # 70% loan-to-value
    )

    # H100 economics
    h100_econ = GPUEconomics(
        purchase_price=30_000,
        rental_rate_hour=1.50,  # User's correction
        utilization=0.80,  # 80% utilization
        opex_per_gpu_year=5_500,  # Datacenter, power, maintenance
        depreciation_rate=0.25,  # 25% per year (moderate estimate)
    )

    # AWS/GCP H100 rental rate (approximate)
    aws_h100_rate = 3.00  # $/hour (typical cloud markup)

    # Scenario 1: $100M deployment (3,333 H100 GPUs)
    gpu_count_1 = 3_333
    own_1 = usdai_financing_model(gpu_count_1, loan_terms, h100_econ)
    cloud_1 = cloud_rental_model(gpu_count_1, aws_h100_rate, loan_terms.term_years, h100_econ.utilization)
    print_comparison("$100M Deployment (3,333 H100 GPUs)", own_1, cloud_1)

    # Scenario 2: Full $500M facility ($714M total deployment)
    gpu_count_2 = 23_800  # $714M / $30k
    own_2 = usdai_financing_model(gpu_count_2, loan_terms, h100_econ)
    cloud_2 = cloud_rental_model(gpu_count_2, aws_h100_rate, loan_terms.term_years, h100_econ.utilization)
    print_comparison("Full $500M Facility ($714M deployment, 23,800 GPUs)", own_2, cloud_2)

    # Scenario 3: Conservative depreciation (33%/year per USD.AI)
    h100_econ_conservative = GPUEconomics(
        purchase_price=30_000,
        rental_rate_hour=1.50,
        utilization=0.80,
        opex_per_gpu_year=5_500,
        depreciation_rate=0.33,  # USD.AI's 3-year full depreciation
    )
    own_3 = usdai_financing_model(gpu_count_1, loan_terms, h100_econ_conservative)
    print_comparison("Conservative Depreciation (33%/year, $100M)", own_3, cloud_1)

    # Scenario 4: High utilization + low opex (QumulusAI's potential advantages)
    h100_econ_optimized = GPUEconomics(
        purchase_price=30_000,
        rental_rate_hour=1.50,
        utilization=0.95,  # 95% utilization (excellent ops)
        opex_per_gpu_year=3_000,  # Own power/datacenter = lower costs
        depreciation_rate=0.25,
    )
    own_4 = usdai_financing_model(gpu_count_1, loan_terms, h100_econ_optimized)
    print_comparison("Optimized Operations (95% util, $3k opex, $100M)", own_4, cloud_1)

    # Scenario 5: What if they could charge premium pricing?
    h100_econ_premium = GPUEconomics(
        purchase_price=30_000,
        rental_rate_hour=2.00,  # Premium pricing
        utilization=0.90,
        opex_per_gpu_year=3_500,
        depreciation_rate=0.25,
    )
    own_5 = usdai_financing_model(gpu_count_1, loan_terms, h100_econ_premium)
    print_comparison("Premium Pricing ($2/hr, 90% util, $100M)", own_5, cloud_1)


if __name__ == "__main__":
    main()
