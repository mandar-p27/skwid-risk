# SKWID Simulation Engine

## Run locally
```bash
pip install numpy scipy
python3 simulate_fast.py
```

## What it models
- 100,000 Monte Carlo simulations × 90-day horizon
- 5 trader behavioural profiles (Casual → Outlier)
- Log-Normal payout distributions with correlation shocks
- Full unit economics per product × profile combination
- Opportunist deep dive (small repeated withdrawal pattern)
- Oracle: forward-looking scenario verdicts

## Output
Results saved to `results/simulation_results.json`
