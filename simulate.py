"""
SKWID Quantitative Risk Engine
================================
Full Monte Carlo simulation in Python/NumPy.
100,000 simulations. Models every trader withdrawal pattern.

Run:
    python simulate.py

Outputs:
    - Console report
    - results.json  (for frontend consumption)
    - plots/        (PNG charts)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict

np.random.seed(42)  # reproducible base; removed per run for MC

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

@dataclass
class FirmConfig:
    # Reserve
    starting_reserve:     float = 10_000
    emergency_ratio:      float = 0.25      # floor = reserve × this

    # Products
    price_5k_1step:       float = 59.0
    price_5k_2step:       float = 49.0
    price_10k_1step:      float = 99.0
    price_10k_2step:      float = 79.0

    # Cost structure
    affiliate_pct:        float = 0.15      # 15% affiliate commission
    payment_pct:          float = 0.04      # 4% payment processing
    op_cost_per_day:      float = 15.0      # daily operating cost

    # Evaluation
    pass_rate_1step:      float = 0.136     # from Propr public data
    pass_rate_2step:      float = 0.086     # harder challenge
    product_mix:          dict  = None      # set in __post_init__

    # Funded account rules
    profit_split:         float = 0.80      # trader keeps 80%
    max_daily_drawdown:   float = 0.05      # 5% daily DD limit
    max_total_drawdown:   float = 0.10      # 10% total DD limit

    # Trader correlation
    correlation:          float = 0.25      # payout clustering in trends

    # Simulation
    n_simulations:        int   = 100_000
    horizon_days:         int   = 90
    random_seed:          int   = None      # None = random each run

    def __post_init__(self):
        if self.product_mix is None:
            self.product_mix = {
                "5k_1step": 0.45,
                "5k_2step": 0.17,
                "10k_1step": 0.28,
                "10k_2step": 0.10,
            }

CFG = FirmConfig()

# ─── TRADER BEHAVIOURAL PROFILES ──────────────────────────────────────────────
# This is the key insight: not all traders behave the same.
# Each profile has:
#   share          = what % of funded traders fall here
#   payout_style   = how they withdraw
#   freq_per_month = how often they request payout
#   size_mean      = average payout size ($)
#   size_std       = std dev of payout size
#   n_payouts_max  = max lifetime payouts before account closes
#   lifetime_days  = how long they stay funded
#   exit_prob_day  = daily probability of account ending (breach/quit)

TRADER_PROFILES = [
    {
        "name":           "Casual",
        "share":          0.55,
        "description":    "Emotional, overleveraged, fail fast. Never reach payout.",
        "payout_prob":    0.01,       # 1% chance they ever pay out
        "freq_per_month": 0,
        "size_mean":      0,
        "size_std":       0,
        "n_payouts_max":  0,
        "lifetime_days":  4.3,        # matches Propr avg account life
        "exit_prob_day":  0.23,
    },
    {
        "name":           "Opportunist",
        "share":          0.20,
        "description":    "Knows the rules. Makes small frequent withdrawals to extract max value. Your $59 challenge costs you $180-400.",
        "payout_prob":    0.60,       # 60% reach payout
        "freq_per_month": 2.5,        # withdraw 2-3x per month
        "size_mean":      120,        # small amounts: $80-160
        "size_std":       40,
        "n_payouts_max":  6,          # 2-3 months then account ends
        "lifetime_days":  75,
        "exit_prob_day":  0.013,
    },
    {
        "name":           "Disciplined",
        "share":          0.15,
        "description":    "Consistent risk management. Monthly payout, medium size.",
        "payout_prob":    0.55,
        "freq_per_month": 1.0,
        "size_mean":      280,
        "size_std":       120,
        "n_payouts_max":  8,
        "lifetime_days":  120,
        "exit_prob_day":  0.008,
    },
    {
        "name":           "Professional",
        "share":          0.08,
        "description":    "Real edge. Regular large payouts. Positive for reputation, expensive for reserve.",
        "payout_prob":    0.75,
        "freq_per_month": 1.5,
        "size_mean":      800,
        "size_std":       350,
        "n_payouts_max":  12,
        "lifetime_days":  180,
        "exit_prob_day":  0.005,
    },
    {
        "name":           "Outlier",
        "share":          0.02,
        "description":    "Rare. Extremely profitable. Single trader can pay out $5k-20k. Determines reserve floor.",
        "payout_prob":    0.85,
        "freq_per_month": 2.0,
        "size_mean":      3500,
        "size_std":       2000,
        "n_payouts_max":  20,
        "lifetime_days":  365,
        "exit_prob_day":  0.003,
    },
]

# ─── PRODUCT PRICING ──────────────────────────────────────────────────────────

PRODUCTS = {
    "5k_1step":  {"price": CFG.price_5k_1step,  "account_size": 5000,  "pass_rate": CFG.pass_rate_1step},
    "5k_2step":  {"price": CFG.price_5k_2step,  "account_size": 5000,  "pass_rate": CFG.pass_rate_2step},
    "10k_1step": {"price": CFG.price_10k_1step, "account_size": 10000, "pass_rate": CFG.pass_rate_1step},
    "10k_2step": {"price": CFG.price_10k_2step, "account_size": 10000, "pass_rate": CFG.pass_rate_2step},
}

# ─── UNIT ECONOMICS PER PRODUCT ───────────────────────────────────────────────

def compute_unit_economics(product_key: str, profile: dict, config: FirmConfig) -> dict:
    """
    For a given product + trader profile combination,
    compute the full economic lifecycle.

    This answers: if I sell one challenge to someone who
    becomes a [Profile] trader, what does that actually cost me?
    """
    prod = PRODUCTS[product_key]
    price = prod["price"]
    pass_rate = prod["pass_rate"]

    # Immediate revenue
    net_revenue = price * (1 - config.affiliate_pct - config.payment_pct)

    # Expected payout liability over trader lifetime
    if profile["payout_prob"] > 0 and profile["size_mean"] > 0:
        n_payouts_expected = min(
            profile["payout_prob"] * profile["freq_per_month"] * (profile["lifetime_days"] / 30),
            profile["n_payouts_max"]
        )
        expected_payout_per_event = profile["size_mean"] * config.profit_split
        total_expected_liability  = pass_rate * n_payouts_expected * expected_payout_per_event
    else:
        n_payouts_expected        = 0
        expected_payout_per_event = 0
        total_expected_liability  = 0

    contribution     = net_revenue - total_expected_liability
    liability_multiple = (total_expected_liability / price) if price > 0 else 0

    return {
        "product":               product_key,
        "profile":               profile["name"],
        "price":                 price,
        "net_revenue":           net_revenue,
        "pass_rate":             pass_rate,
        "expected_payouts":      n_payouts_expected,
        "liability_per_payout":  expected_payout_per_event,
        "total_liability":       total_expected_liability,
        "contribution":          contribution,
        "liability_multiple":    liability_multiple,
        "viable":                contribution > 0,
    }

# ─── SINGLE TRADER LIFECYCLE SIMULATION ───────────────────────────────────────

def simulate_trader_lifecycle(profile: dict, account_size: float, config: FirmConfig, rng: np.random.Generator) -> dict:
    """
    Simulate a single funded trader's complete lifecycle.
    Returns all payouts they make and when.
    """
    payouts = []
    day = 0
    payout_count = 0
    active = True

    # Does this trader ever reach payout? (based on profile probability)
    if rng.random() > profile["payout_prob"]:
        return {"payouts": [], "lifetime_days": int(rng.exponential(profile["lifetime_days"])), "profile": profile["name"]}

    payout_interval_days = 30 / max(0.1, profile["freq_per_month"])

    while active and payout_count < profile["n_payouts_max"]:
        # Time until next payout attempt
        days_to_next = rng.exponential(payout_interval_days)
        day += days_to_next

        # Daily exit check over this period
        if rng.random() < (1 - (1 - profile["exit_prob_day"]) ** days_to_next):
            active = False
            break

        if day > profile["lifetime_days"]:
            break

        # Payout size: Log-Normal distribution
        # This creates realistic right tail — most payouts small, few very large
        if profile["size_mean"] > 0:
            mu = np.log(profile["size_mean"]**2 / np.sqrt(profile["size_std"]**2 + profile["size_mean"]**2))
            sigma = np.sqrt(np.log(1 + (profile["size_std"] / profile["size_mean"])**2))
            raw_payout = rng.lognormal(mu, sigma)
        else:
            raw_payout = 0

        # Cap at account size (can't withdraw more than funded capital × profit split)
        raw_payout = min(raw_payout, account_size * config.profit_split)
        firm_pays = raw_payout * config.profit_split

        payouts.append({
            "day":        int(day),
            "gross":      float(raw_payout),
            "firm_pays":  float(firm_pays),
        })
        payout_count += 1

    return {
        "payouts":      payouts,
        "lifetime_days": int(day),
        "profile":      profile["name"],
    }

# ─── FULL MONTE CARLO ENGINE ──────────────────────────────────────────────────

def run_monte_carlo(
    sales_per_day: float = 3.0,
    config: FirmConfig = CFG,
    n_sims: int = None,
    verbose: bool = True,
) -> dict:
    """
    Main simulation engine.
    Runs n_sims independent 90-day scenarios.
    Each scenario models daily sales, evaluations, funded traders,
    payout events, correlation shocks, fraud, operating costs.
    """
    N = n_sims or config.n_simulations
    DAYS = config.horizon_days
    rng = np.random.default_rng(config.random_seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SKWID Monte Carlo Engine — {N:,} simulations × {DAYS} days")
        print(f"{'='*60}")
        print(f"  Starting reserve:  ${config.starting_reserve:,.0f}")
        print(f"  Sales per day:     {sales_per_day}")
        print(f"  Pass rates:        1-step {config.pass_rate_1step:.1%} | 2-step {config.pass_rate_2step:.1%}")
        print(f"  Profit split:      {config.profit_split:.0%}")
        print(f"  Correlation:       {config.correlation:.0%}")
        print()

    floor = config.starting_reserve * config.emergency_ratio

    # Pre-generate product mix probabilities
    products = list(config.product_mix.keys())
    mix_probs = np.array([config.product_mix[p] for p in products])
    mix_probs /= mix_probs.sum()

    # Storage
    final_reserves  = np.zeros(N)
    total_revenues  = np.zeros(N)
    total_payouts   = np.zeros(N)
    total_funded    = np.zeros(N)
    max_drawdowns   = np.zeros(N)
    ruined          = np.zeros(N, dtype=bool)
    profile_payouts = {p["name"]: [] for p in TRADER_PROFILES}

    # Profile cumulative shares for assignment
    profile_shares = np.array([p["share"] for p in TRADER_PROFILES])
    profile_shares /= profile_shares.sum()
    profile_cumsum = np.cumsum(profile_shares)

    if verbose:
        print("  Running simulations...")

    for s in range(N):
        if verbose and s % 10000 == 0:
            print(f"    {s:>7,} / {N:,}  ({100*s/N:.0f}%)")

        reserve = config.starting_reserve
        peak_reserve = reserve
        funded_traders = []
        total_rev = 0
        total_pay = 0
        n_funded = 0

        # Sample simulation-level parameters (epistemic uncertainty)
        sim_pass_1step = np.clip(rng.normal(config.pass_rate_1step, 0.02), 0.02, 0.40)
        sim_pass_2step = np.clip(rng.normal(config.pass_rate_2step, 0.015), 0.01, 0.30)
        sim_corr       = np.clip(rng.normal(config.correlation, 0.08), 0, 0.90)

        # Daily loop
        for day in range(DAYS):

            # ── DAILY SALES ──────────────────────────────────────────────
            n_sales = max(0, int(rng.poisson(sales_per_day)))

            for _ in range(n_sales):
                # Pick product
                prod_key = products[np.searchsorted(mix_probs.cumsum(), rng.random())]
                prod = PRODUCTS[prod_key]

                # Revenue
                net = prod["price"] * (1 - config.affiliate_pct - config.payment_pct)
                reserve += net
                total_rev += net

                # Evaluation outcome
                pass_rate = sim_pass_1step if "1step" in prod_key else sim_pass_2step
                if rng.random() < pass_rate:
                    # Assign trader profile
                    r = rng.random()
                    profile_idx = np.searchsorted(profile_cumsum, r)
                    profile_idx = min(profile_idx, len(TRADER_PROFILES) - 1)
                    profile = TRADER_PROFILES[profile_idx]

                    funded_traders.append({
                        "profile":      profile,
                        "account_size": prod["account_size"],
                        "start_day":    day,
                        "payouts_made": 0,
                        "active":       True,
                        "next_payout_day": day + max(1, int(rng.exponential(30 / max(0.1, profile["freq_per_month"])))),
                        "will_pay":     rng.random() < profile["payout_prob"],
                    })
                    n_funded += 1

            # ── PROCESS ACTIVE FUNDED TRADERS ────────────────────────────
            # Correlation shock: trending market makes pro traders cluster
            corr_shock_active = rng.random() < sim_corr

            for trader in funded_traders:
                if not trader["active"]:
                    continue

                # Daily exit check
                if rng.random() < trader["profile"]["exit_prob_day"]:
                    trader["active"] = False
                    continue

                # Payout check
                if (trader["will_pay"] and
                    day >= trader["next_payout_day"] and
                    trader["payouts_made"] < trader["profile"]["n_payouts_max"]):

                    profile = trader["profile"]
                    if profile["size_mean"] > 0:
                        mu = np.log(profile["size_mean"]**2 / np.sqrt(profile["size_std"]**2 + profile["size_mean"]**2))
                        sigma = np.sqrt(np.log(1 + (profile["size_std"] / profile["size_mean"])**2))
                        raw = rng.lognormal(mu, sigma)

                        # Correlation shock amplifies payouts when market trending
                        if corr_shock_active and profile["name"] in ["Professional", "Outlier", "Disciplined"]:
                            raw *= np.clip(rng.normal(1.5, 0.3), 1.0, 3.0)

                        # Cap at account size
                        raw = min(raw, trader["account_size"])
                        firm_pays = raw * config.profit_split

                        reserve -= firm_pays
                        total_pay += firm_pays
                        trader["payouts_made"] += 1
                        profile_payouts[profile["name"]].append(firm_pays)

                        # Schedule next payout
                        interval = max(1, int(rng.exponential(30 / max(0.1, profile["freq_per_month"]))))
                        trader["next_payout_day"] = day + interval

            # ── FRAUD EVENT ───────────────────────────────────────────────
            if rng.random() < 0.005:
                fraud_loss = rng.lognormal(np.log(250), 0.6)
                reserve -= fraud_loss
                total_pay += fraud_loss

            # ── OPERATING COSTS ───────────────────────────────────────────
            reserve -= config.op_cost_per_day

            # ── DRAWDOWN TRACKING ─────────────────────────────────────────
            if reserve > peak_reserve:
                peak_reserve = reserve
            dd = (peak_reserve - reserve) / max(1, peak_reserve)

            # ── RUIN CHECK ────────────────────────────────────────────────
            if reserve < floor:
                ruined[s] = True

            # Store max drawdown for this run
            if s == 0:
                max_drawdowns[s] = dd
            else:
                max_drawdowns[s] = max(max_drawdowns[s], dd)

        final_reserves[s] = reserve
        total_revenues[s]  = total_rev
        total_payouts[s]   = total_pay
        total_funded[s]    = n_funded

    if verbose:
        print(f"    {N:>7,} / {N:,}  (100%)\n")

    # ── AGGREGATE RESULTS ─────────────────────────────────────────────────────
    pRuin = ruined.mean()
    pcts  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    reserve_percentiles = {f"p{p}": float(np.percentile(final_reserves, p)) for p in pcts}

    # Profile payout statistics
    profile_stats = {}
    for name, pays in profile_payouts.items():
        if pays:
            arr = np.array(pays)
            profile_stats[name] = {
                "count":      len(arr),
                "mean":       float(arr.mean()),
                "median":     float(np.median(arr)),
                "p95":        float(np.percentile(arr, 95)),
                "p99":        float(np.percentile(arr, 99)),
                "max":        float(arr.max()),
                "total":      float(arr.sum()),
            }
        else:
            profile_stats[name] = {"count": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0, "total": 0}

    results = {
        "config": {
            "n_simulations":    N,
            "horizon_days":     DAYS,
            "starting_reserve": config.starting_reserve,
            "sales_per_day":    sales_per_day,
            "profit_split":     config.profit_split,
            "correlation":      config.correlation,
        },
        "risk": {
            "probability_of_ruin":   float(pRuin),
            "worst_reserve":         float(final_reserves.min()),
            "best_reserve":          float(final_reserves.max()),
            "avg_max_drawdown":      float(max_drawdowns.mean()),
            "p95_max_drawdown":      float(np.percentile(max_drawdowns, 95)),
        },
        "reserve": reserve_percentiles,
        "cashflow": {
            "avg_revenue_90d":    float(total_revenues.mean()),
            "avg_payouts_90d":    float(total_payouts.mean()),
            "avg_net_90d":        float((total_revenues - total_payouts).mean()),
            "avg_funded_90d":     float(total_funded.mean()),
            "p95_payouts_90d":    float(np.percentile(total_payouts, 95)),
            "p99_payouts_90d":    float(np.percentile(total_payouts, 99)),
        },
        "profiles":     profile_stats,
        "histogram": {
            "values": [float(x) for x in np.histogram(final_reserves, bins=40)[0]],
            "edges":  [float(x) for x in np.histogram(final_reserves, bins=40)[1]],
        }
    }

    return results

# ─── UNIT ECONOMICS REPORT ────────────────────────────────────────────────────

def run_unit_economics_report(config: FirmConfig = CFG) -> dict:
    """
    For every product × trader profile combination,
    compute the full economic outcome.
    This answers: which combinations make money, which lose money?
    """
    report = []
    for prod_key in PRODUCTS:
        for profile in TRADER_PROFILES:
            ue = compute_unit_economics(prod_key, profile, config)
            report.append(ue)
    return report

# ─── SCENARIO: OPPORTUNIST TRADER DEEP DIVE ──────────────────────────────────

def simulate_opportunist_deep_dive(n_traders: int = 10000, config: FirmConfig = CFG) -> dict:
    """
    Deep dive on your key concern: the smart trader who pays $59
    and then makes small frequent withdrawals.

    Models: how much does one Opportunist trader actually cost you?
    """
    rng = np.random.default_rng(42)
    profile = next(p for p in TRADER_PROFILES if p["name"] == "Opportunist")
    results = []

    for _ in range(n_traders):
        lifecycle = simulate_trader_lifecycle(profile, 5000, config, rng)
        total_paid_to_trader = sum(p["firm_pays"] for p in lifecycle["payouts"])
        n_payouts = len(lifecycle["payouts"])
        results.append({
            "n_payouts":       n_payouts,
            "total_cost":      total_paid_to_trader,
            "lifetime_days":   lifecycle["lifetime_days"],
            "cost_vs_revenue": total_paid_to_trader / (config.price_5k_1step * (1 - config.affiliate_pct - config.payment_pct)),
        })

    arr_cost = np.array([r["total_cost"] for r in results])
    arr_n    = np.array([r["n_payouts"] for r in results])
    arr_mult = np.array([r["cost_vs_revenue"] for r in results])

    return {
        "n_traders_simulated": n_traders,
        "pct_who_paid_out":    float((arr_cost > 0).mean()),
        "avg_total_cost":      float(arr_cost.mean()),
        "median_total_cost":   float(np.median(arr_cost)),
        "p95_total_cost":      float(np.percentile(arr_cost, 95)),
        "p99_total_cost":      float(np.percentile(arr_cost, 99)),
        "max_total_cost":      float(arr_cost.max()),
        "avg_n_payouts":       float(arr_n.mean()),
        "avg_cost_multiple":   float(arr_mult[arr_mult > 0].mean()) if (arr_mult > 0).any() else 0,
        "p95_cost_multiple":   float(np.percentile(arr_mult[arr_mult > 0], 95)) if (arr_mult > 0).any() else 0,
        "pct_cost_3x_or_more": float((arr_mult >= 3).mean()),
        "pct_cost_5x_or_more": float((arr_mult >= 5).mean()),
    }

# ─── WHAT SHOULD I DO NEXT — ORACLE ──────────────────────────────────────────

def oracle(
    current_reserve:     float,
    accounts_sold:       int,
    simultaneous_stress: int = 3,
    config:              FirmConfig = CFG,
    n_sims:              int = 50_000,
) -> dict:
    """
    Given current state, run 50k forward simulations and
    return a plain-English verdict + recommended actions.
    """
    rng = np.random.default_rng()
    floor = current_reserve * config.emergency_ratio

    # Expected funded from accounts sold (weighted by product mix and pass rates)
    weighted_pass = sum(
        config.product_mix.get(k, 0) * PRODUCTS[k]["pass_rate"]
        for k in PRODUCTS
    )
    expected_funded = round(accounts_sold * weighted_pass)

    # Stress test: simultaneous payouts
    ruin_count = 0
    total_payouts_sim = []

    for _ in range(n_sims):
        sim_corr = np.clip(rng.normal(config.correlation, 0.08), 0, 0.90)
        total_pay = 0
        for _ in range(simultaneous_stress):
            # Random profile weighted by share
            r = rng.random()
            profile = TRADER_PROFILES[np.searchsorted(np.cumsum([p["share"] for p in TRADER_PROFILES]) / sum(p["share"] for p in TRADER_PROFILES), r)]
            if profile["size_mean"] > 0:
                mu = np.log(profile["size_mean"]**2 / np.sqrt(profile["size_std"]**2 + profile["size_mean"]**2))
                sigma = np.sqrt(np.log(1 + (profile["size_std"] / profile["size_mean"])**2))
                raw = rng.lognormal(mu, sigma)
                if rng.random() < sim_corr:
                    raw *= np.clip(rng.normal(1.5, 0.3), 1.0, 3.0)
                total_pay += raw * config.profit_split

        total_payouts_sim.append(total_pay)
        if current_reserve - total_pay < floor:
            ruin_count += 1

    arr = np.array(total_payouts_sim)
    p_ruin = ruin_count / n_sims
    p99_payout = float(np.percentile(arr, 99))
    safe_simultaneous = max(1, int((current_reserve - floor) * 0.9 / max(1, float(np.percentile(arr, 99)) / simultaneous_stress)))

    # Reserve needed
    avg_profile_cost = sum(
        p["share"] * p["payout_prob"] * (p["freq_per_month"] * (p["lifetime_days"]/30)) * p["size_mean"] * config.profit_split
        for p in TRADER_PROFILES
    ) / sum(p["share"] for p in TRADER_PROFILES)
    reserve_needed = expected_funded * avg_profile_cost * 2.5

    # Verdict
    if p_ruin > 0.10:
        verdict = "STOP"
        color = "red"
    elif p_ruin > 0.02:
        verdict = "CAUTION"
        color = "yellow"
    else:
        verdict = "SAFE"
        color = "green"

    return {
        "verdict":           verdict,
        "color":             color,
        "p_ruin":            float(p_ruin),
        "p99_stress_payout": float(p99_payout),
        "safe_simultaneous": safe_simultaneous,
        "expected_funded":   expected_funded,
        "reserve_needed":    float(reserve_needed),
        "reserve_gap":       float(max(0, reserve_needed - current_reserve)),
        "median_stress":     float(np.median(arr)),
    }

# ─── MAIN REPORT ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  SKWID RISK ENGINE — FULL SIMULATION REPORT")
    print("="*60)

    # 1. Unit Economics
    print("\n── UNIT ECONOMICS ──────────────────────────────────────────")
    print(f"{'Product':<14} {'Profile':<14} {'Price':>6} {'Net Rev':>8} {'Exp Liab':>10} {'Contrib':>9} {'Viable':>7}")
    print("-" * 72)
    ue_report = run_unit_economics_report()
    for ue in ue_report:
        viable = "✓" if ue["viable"] else "✗"
        print(f"{ue['product']:<14} {ue['profile']:<14} ${ue['price']:>5.0f} ${ue['net_revenue']:>7.0f} ${ue['total_liability']:>9.0f} ${ue['contribution']:>8.0f} {viable:>7}")

    # 2. Opportunist deep dive
    print("\n── OPPORTUNIST TRADER DEEP DIVE (10,000 traders) ──────────")
    opp = simulate_opportunist_deep_dive(10000)
    print(f"  % who reach payout:        {opp['pct_who_paid_out']:.1%}")
    print(f"  Avg total cost to firm:    ${opp['avg_total_cost']:.0f}")
    print(f"  Median total cost:         ${opp['median_total_cost']:.0f}")
    print(f"  95th pct cost:             ${opp['p95_total_cost']:.0f}")
    print(f"  99th pct cost:             ${opp['p99_total_cost']:.0f}")
    print(f"  Worst case cost:           ${opp['max_total_cost']:.0f}")
    print(f"  Avg payouts per trader:    {opp['avg_n_payouts']:.1f}")
    print(f"  Avg cost multiple (×rev):  {opp['avg_cost_multiple']:.2f}×")
    print(f"  % costing 3× or more:     {opp['pct_cost_3x_or_more']:.1%}")
    print(f"  % costing 5× or more:     {opp['pct_cost_5x_or_more']:.1%}")

    # 3. Full Monte Carlo
    print("\n── MONTE CARLO (100,000 simulations) ──────────────────────")
    mc = run_monte_carlo(sales_per_day=3.0, n_sims=100_000, verbose=True)

    print(f"\n  RISK")
    print(f"  Probability of ruin:       {mc['risk']['probability_of_ruin']:.3%}")
    print(f"  Avg max drawdown:          {mc['risk']['avg_max_drawdown']:.1%}")
    print(f"  95th pct drawdown:         {mc['risk']['p95_max_drawdown']:.1%}")

    print(f"\n  RESERVE PERCENTILES (90-day final)")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = mc['reserve'][f'p{pct}']
        flag = " ← RUIN" if val < CFG.starting_reserve * CFG.emergency_ratio else ""
        print(f"  p{pct:<3}  ${val:>10,.0f}{flag}")

    print(f"\n  CASHFLOW (90-day averages)")
    print(f"  Revenue:                   ${mc['cashflow']['avg_revenue_90d']:>8,.0f}")
    print(f"  Payouts:                   ${mc['cashflow']['avg_payouts_90d']:>8,.0f}")
    print(f"  Net:                       ${mc['cashflow']['avg_net_90d']:>8,.0f}")
    print(f"  Funded traders:            {mc['cashflow']['avg_funded_90d']:>8.0f}")
    print(f"  p95 payouts:               ${mc['cashflow']['p95_payouts_90d']:>8,.0f}")
    print(f"  p99 payouts:               ${mc['cashflow']['p99_payouts_90d']:>8,.0f}")

    print(f"\n  PAYOUT BY TRADER PROFILE")
    for name, stats in mc["profiles"].items():
        if stats["count"] > 0:
            print(f"  {name:<14}  count={stats['count']:>6,}  mean=${stats['mean']:>6.0f}  p99=${stats['p99']:>7.0f}  max=${stats['max']:>8.0f}")

    # 4. Oracle example
    print("\n── ORACLE: 10k reserve + 100 accounts ─────────────────────")
    verdict = oracle(current_reserve=10000, accounts_sold=100, simultaneous_stress=3)
    print(f"  Verdict:                   {verdict['verdict']}")
    print(f"  P(ruin) at 3 sim payouts:  {verdict['p_ruin']:.2%}")
    print(f"  Safe simultaneous:         {verdict['safe_simultaneous']} traders")
    print(f"  Expected funded:           {verdict['expected_funded']} traders")
    print(f"  Reserve needed:            ${verdict['reserve_needed']:,.0f}")
    print(f"  Reserve gap:               ${verdict['reserve_gap']:,.0f}")

    # Save JSON for frontend
    output = {
        "unit_economics":  ue_report,
        "opportunist":     opp,
        "monte_carlo":     mc,
        "oracle_example":  verdict,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/simulation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Results saved to results/simulation_results.json")
    print("="*60 + "\n")

    return output

if __name__ == "__main__":
    main()
