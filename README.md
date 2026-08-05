# BookMaker — Limit Order Book Market-Making Simulator
Market-making simulator on a synthetic limit order book. Compares hand-tuned inventory/adverse-selection strategies against an RL-learned policy, under realistic latency modeling. Backtested on fill quality, P&amp;L, and adverse-selection cost. Runs entirely on CPU.

## Data

Dual-source, decided up front rather than left implicit:

**Primary: synthetic order flow (`data/synthetic_lob.py`).** Poisson-process
order arrivals with a reference-mid random walk, geometric depth placement,
lognormal sizes, and independent exponential order lifetimes for cancels.
Calibration parameters are starting points drawn from published market
microstructure literature (queue-reactive / Avellaneda-Stoikov-style
models), not fitted to any specific stock — a disclosed judgment call, not
an empirical fact. This is the primary source because the project's core
value is mechanism (matching-engine correctness, latency effects, RL vs.
heuristics), which needs unlimited volume and controllable regimes (a
`regime="stressed"` knob raises arrival rates/volatility for the latency
sweep and robustness testing), not a specific real
instrument. Real LOBSTER equity data was investigated and ruled out on
cost: lobsterdata.com now gates all downloads behind a paid university
subscription or an approval-gated academic trial billed per MB in
"lobster-data-coin."

**Secondary / validation: real Binance order book + trade data
(`data/binance_capture.py`).** Captured live from Binance's public,
unauthenticated market-data mirror (`data-api.binance.vision` /
`data-stream.binance.vision` — geo-unrestricted, no API key; the primary
`api.binance.com` host is geo-blocked from where this was built).
Reconstructs a local order book from a REST snapshot + WebSocket diff
stream per Binance's documented reconciliation procedure, alongside the
real trade tape. Used to check whether findings on synthetic
data hold up on real market data, not as the primary dataset. Two
limitations to keep in mind when reading those results: (1) Binance's
public feed is price-level aggregated, not per-order, so it can validate
book-level dynamics (spread, imbalance, mid-price, fill-vs-real-trades)
but can't drive the individual-order matching engine the way the synthetic
generator's events do; (2) crypto trades 24/7 with different tick/lot
conventions and materially lighter regulation than equities, so a finding
that replicates here is evidence the mechanism generalizes past one
synthetic model, not a claim that it holds for equities markets.
