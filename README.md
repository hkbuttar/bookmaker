# BookMaker — Limit Order Book Market-Making Simulator

Market-making simulator on a synthetic limit order book. Compares hand-tuned inventory/adverse-selection strategies against an RL-learned policy, under realistic latency modeling. Backtested on fill quality, P&amp;L, and adverse-selection cost, then validated against real Binance order book data. Runs entirely on CPU.

**Live demo:** [bookmaker-frontend-oeed.onrender.com/frontend](https://bookmaker-frontend-oeed.onrender.com/frontend) — hosted on Render's free tier, so the first load after a period of inactivity can take ~50s to spin back up.

## Motivation

Market making is a natural setting for comparing hand-tuned heuristics against learned policies: the reward signal (P&L) is unambiguous, the state is well-defined (the book), and the decision is simple to express (where to quote, how much) while still being genuinely hard to do well — a market maker has to balance capturing spread against inventory risk and adverse selection, under real latency between deciding on a quote and that quote landing in the book. This project builds the full stack needed to ask that comparison honestly: a price-time-priority matching engine (not a fill-probability approximation), three hand-tuned strategies of increasing sophistication, a latency model that separates *decision* time from *arrival* time, and a DQN policy trained and evaluated through the exact same execution path as the heuristics — so "RL vs. hand-tuned" is a fair fight, not an artifact of the two sides being tested differently. The honest answer turned out to be more interesting than a clean win either way; see [Results](#results).

## Architecture

```
data/         synthetic order-flow generator + Binance capture/reconciliation
lob/          matching engine, order book, feature extraction, latency model
strategies/   naive / inventory-aware / adverse-selection-aware quoting logic
risk/         kill-switch (absolute drawdown) + inventory-limit wrapper
backtest/     replay harness, portfolio/fill accounting, metrics, Binance harness
rl/           Gymnasium env, DQN training, evaluation, diagnostics
backend/      FastAPI API + SQLAlchemy models + one-time results-population script
frontend/     Bokeh dashboard (reads the database directly, not through the API)
notebooks/    pre-executed research notebook — the full comparison, honestly reported
tests/        201 tests across every layer above
```

Data flows one direction: a background event stream (`data/`) drives the matching engine (`lob/`); strategies (`strategies/`, optionally `rl/`) observe book state and return quotes; `backtest/` replays the whole thing with a configurable latency model and produces fills, P&L, and metrics. `backend/` and `frontend/` are a thin persistence + visualization layer on top of that — neither reimplements any simulation logic.

## Data

Dual-source, decided up front rather than left implicit:

**Primary: synthetic order flow (`data/synthetic_lob.py`).** Poisson-process order arrivals with a reference-mid random walk, geometric depth placement, lognormal sizes, and independent exponential order lifetimes for cancels. Calibration parameters are starting points drawn from published market microstructure literature (queue-reactive / Avellaneda-Stoikov-style models), not fitted to any specific stock — a disclosed judgment call, not an empirical fact. This is the primary source because the project's core value is mechanism (matching-engine correctness, latency effects, RL vs. heuristics), which needs unlimited volume and controllable regimes (a `regime="stressed"` knob raises arrival rates/volatility for the latency sweep and robustness testing), not a specific real instrument. Real LOBSTER equity data was investigated and ruled out on cost: lobsterdata.com now gates all downloads behind a paid university subscription or an approval-gated academic trial billed per MB in "lobster-data-coin."

**Secondary / validation: real Binance order book + trade data (`data/binance_capture.py`).** Captured live from Binance's public, unauthenticated market-data mirror (`data-api.binance.vision` / `data-stream.binance.vision` — geo-unrestricted, no API key; the primary `api.binance.com` host is geo-blocked from where this was built). Reconstructs a local order book from a REST snapshot + WebSocket diff stream per Binance's documented reconciliation procedure, alongside the real trade tape. Used to check whether findings on synthetic data hold up on real market data, not as the primary dataset. Two limitations to keep in mind when reading those results: (1) Binance's public feed is price-level aggregated, not per-order, so it can validate book-level dynamics (spread, imbalance, mid-price, fill-vs-real-trades) but can't drive the individual-order matching engine the way the synthetic generator's events do; (2) crypto trades 24/7 with different tick/lot conventions and materially lighter regulation than equities, so a finding that replicates here is evidence the mechanism generalizes past one synthetic model, not a claim that it holds for equities markets.

## Methodology

**Matching engine** (`lob/`): price-time priority, `SortedDict` bid/ask sides with FIFO queues per price level — real order-level matching, not a fill-probability model. Decision time and arrival time are separate from the start: a strategy decides a quote at one timestamp, and (under a latency model) that quote only reaches the book at a later one, which is what makes latency effects and RL/heuristic comparisons under delay meaningful rather than approximated after the fact.

**Strategies** (`strategies/`), in increasing sophistication: `naive` quotes a fixed half-spread around mid; `inventory_aware` shifts its reservation price against current inventory (a simplified Avellaneda-Stoikov-style penalty); `adverse_selection_aware` additionally tracks an EMA of order-book imbalance and widens or pulls its quotes when persistent (not instantaneous) imbalance signals informed flow. All three round-trip through the same `risk/` layer — an absolute-dollar-drawdown kill switch (not fractional, since this starts every session at $0 equity rather than a funded account) and an inventory-limit clip.

**Latency** (`lob/latency.py`): a lognormal delay model with four presets (0/5/20/50ms), applied to the gap between a strategy's decision and that decision's arrival in the book — every strategy, hand-tuned or RL, is measured under the identical mechanism.

**RL** (`rl/`): a Gymnasium environment wrapping the same matching engine and latency model, decision cadence fixed at 1s (not per-event, so a trained policy's action rate is comparable to the hand-tuned strategies') and a discrete `{1,2,3}-tick offset × {5,10,20}-unit size` action table (narrowed from a wider grid after diagnosing that finer offsets were essentially unreachable at this order book's calibrated depth). Trained with DQN (`stable-baselines3`) against a Huber-style (capped-quadratic, not pure-quadratic) inventory penalty, with an inventory-penalty curriculum that ramps in over the first 30% of training — a plain quadratic penalty from step one collapsed training to "never quote" (see [Results](#results)). `rl/diagnostics.py` flags training-curve concerns automatically, but is explicitly documented (and, in the notebook, demonstrated) as necessary and not sufficient — it both missed a real failure mode and false-flagged a working curriculum run, which is why every RL claim in this project is backed by held-out evaluation, not training curves alone.

**Backtesting** (`backtest/`): replays a background event stream through the matching engine and a strategy's decisions in real time, producing fills, a portfolio history, and summary metrics (fill count, P&L, within-session Sharpe, adverse-selection cost via markout). A separate, deliberately simpler harness (`backtest/binance_backtest.py`) handles the real Binance data, since its price-level-aggregated feed
can't drive the same order-level matching engine.

## Results

Full methodology, tables, and charts are in the pre-executed `notebooks/research.ipynb` — this is a condensed, honest summary, not a sales pitch:

- **RL beats the hand-tuned baselines on the primary comparison**, after a disclosed, evidence-based configuration choice (an optional reward bonus for landing fills helps one trained variant and hurts the other — confirmed via a controlled ablation, not asserted). The better-configured policy(`rl_latency_naive`) gets 43 fills / $11.50 P&L on the headline held-out session vs. the naive baseline's 36 fills / $7.20, and averaged over ten held-out sessions: 53.5 mean fills / $11.85 mean P&L vs. 24.3 / $2.63, with zero zero-fill sessions.
- **The weaker RL variant (`rl_latency_aware`) still fails outright on the specific headline session** (0 fills at every latency preset) despite averaging respectably across the other nine — disclosed directly rather than swapped out for a better-looking seed. RL's real, but session-sensitive, not uniformly reliable.
- **Latency (0-50ms) has little effect on P&L for any strategy at this session length** — consistent across both the original calibration sweep and the final RL comparison.
- **On real Binance data, all three hand-tuned strategies lose money** (-$149 to -$173 over 15 minutes) with markout costs two orders of magnitude worse than on synthetic data — not because the strategies are wrong, but because a $0.02 half-spread calibrated for a ~$100 synthetic instrument provides essentially no buffer against real BTC price momentum at its actual ~$64k scale. Parameters don't transfer across instruments without recalibration; that's a finding, not a bug.
- **Strategy ranking flips between synthetic and real data**: `inventory_aware` is worst on the calm synthetic session (its inventory clamping forgoes fills naive happily takes) but *best* on real BTC data — its risk discipline pays off exactly where real volatility rewards it.
- **RL policies evaluated directly on Binance data didn't break**, because the action table's price offsets are tick-relative rather than price-relative — but their risk *perception* almost certainly didn't transfer (inventory normalization was calibrated for the synthetic instrument's scale), which this project's evidence can flag but can't fully confirm without instrumenting the network's internals.

## Limitations

- Synthetic calibration (arrival rates, volatility) is a disclosed, illustrative choice, not fitted to any real instrument.
- The Binance validation layer is price-level aggregated and simulates fills under an optimistic always-at-front-of-queue assumption — read its results directionally, not as precise real-world P&L.
- RL was trained on a single fixed 10-minute synthetic session (600s) for CPU-time reasons; the curriculum's warmup schedule (30% of training, linear ramp) is a disclosed, un-tuned choice.
- The Binance sub-window robustness check splits one continuous 15-minute capture rather than sampling independent market realizations — within-capture consistency, not true robustness.
- No transaction cost or exchange fee modeling anywhere in this project.
- Binance strategy parameters and RL observation normalization were not recalibrated for BTC's price/volatility scale — several Binance findings reflect that mismatch as much as the strategies themselves.

## Future work

- Recalibrate strategy parameters and RL observation normalization for BTC's actual scale, to separate "doesn't transfer without recalibration" from "doesn't work" on real data.
- Sweep the RL curriculum's warmup schedule and train across multiple synthetic sessions/seeds, rather than the current single fixed session.
- Add transaction costs / exchange fees, which would compress every strategy's edge and could change the synthetic-vs-real ranking further.
- Extend the Binance capture beyond one continuous 15-minute window to get genuinely independent real-data sessions, closing the gap with the synthetic robustness check.

## Running it locally

```bash
git clone <this repo> && cd bookmaker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # full dev env: sim + RL training + notebook + tests

pytest                                  # 201 tests across every layer
jupyter notebook notebooks/research.ipynb  # pre-executed and committed; open to read, or re-execute in place to reproduce it fresh

python3 -m backend.populate             # trains both RL policies + runs the comparison sweep (a few minutes) -- writes to ./bookmaker.db (SQLite) by default
uvicorn backend.main:app --reload       # optional: REST API at localhost:8000/docs
bokeh serve frontend --show             # dashboard at localhost:5006/frontend
```

`DATABASE_URL` (unset defaults to local SQLite) points every piece — `backend.populate`, the API, and the dashboard — at the same store; set it once to a Postgres URL to run everything against a shared database instead.

## Deployment

Two independent, stateless web services sharing one Postgres database -- the dashboard (`frontend/`) reads the database directly rather than through the backend API, so the two services never call each other and share nothing but `DATABASE_URL`. RL training is a one-time offline job, not a deployed service.

**One command via Render's Blueprint** (`render.yaml`, repo root): push this repo to GitHub, then in the Render dashboard choose *New > Blueprint* and point it at the repo. It provisions a free Postgres instance and both services (`bookmaker-backend`, a FastAPI/uvicorn app; `bookmaker-frontend`, `bokeh serve`), wiring `DATABASE_URL` between them automatically. Both build from `requirements-deploy.txt`, a lean subset of `requirements.txt` -- neither service imports `rl/`, so torch/gymnasium/stable-baselines3 (large, slow to install) are left out of the deploy image; that split is enforced by `backend/populate.py` being the only module that imports `rl.*`.

**Populate the database once, after the first deploy** -- the comparison table, RL training curves, and order-book-depth demo runs are all precomputed (a full sweep takes minutes; training longer), not something either service computes on request:
```bash
DATABASE_URL=<external connection string from the Render Postgres dashboard> python3 -m backend.populate
```
Run this from a machine with the full `requirements.txt` installed (it needs torch/stable-baselines3 to train the RL policies) -- it's a one-time local/CI step against the deployed database, not something that runs inside either Render service. Re-run it any time to refresh the numbers; it adds fresh rows rather than erroring, but doesn't deduplicate, so point it at a fresh database (or manually clear the tables) for a clean rebuild.

**Known trade-off:** both services and the database default to Render's free plan in `render.yaml` -- free web services spin down after periods of inactivity (a cold start delays the first request) and free Postgres instances expire after a fixed window. Fine for a portfolio deployment; bump `plan:` to a paid tier for anything that needs to stay warm or persist indefinitely.
