"""Binance public order book capture (secondary / validation data source).

Data source decision (see README "Data" section and synthetic_lob.py's
docstring for the primary-source rationale): this module captures real
order book and trade data from Binance's public market-data API, used to
validate that the hand-tuned strategy comparisons and their findings hold
up on real data, not just the synthetic generator's regimes. It is a
validation layer, not the primary source, and its findings are scoped to
crypto microstructure specifically -- see the granularity/market-structure
limitations below before treating results here as generalizing to equities.

Endpoint choice: `data-api.binance.vision` / `data-stream.binance.vision`
are Binance's own read-only public market-data mirrors of api.binance.com
-- same data, no API key, no geo-restriction, intended for exactly this
kind of public data consumption. (api.binance.com itself 451s from this
environment's network location; the mirror is the documented workaround,
not a scrape.)

Two disclosed limitations relative to LOBSTER/the synthetic generator:

1. Granularity. Binance's public diff-depth stream is aggregated by price
   level (total resting quantity at each price), not per-order -- exchanges
   don't publish individual order IDs on public feeds. That means this
   data can validate book-level dynamics a strategy conditions on (spread,
   imbalance, mid-price, depth) and, via the trade stream, whether a
   hypothetical resting quote would have been executed -- but it cannot
   feed a price-time-priority matching engine at the individual-order
   level the way LOBSTER or the synthetic generator's per-order events do.
2. Market structure. Crypto trades 24/7 with different tick/lot
   conventions and materially lighter regulation than the NASDAQ
   microstructure LOBSTER/the synthetic generator model. A finding that
   replicates here is evidence the mechanism generalizes past one
   synthetic model, not a claim that it holds for equities.

Local book reconstruction follows Binance's documented procedure
(buffer diff events -> fetch a REST snapshot -> discard diffs at or before
the snapshot's lastUpdateId -> verify the first applied diff bridges the
snapshot -> apply the rest in order), implemented in `LocalOrderBook`
below, which is unit-tested without any network dependency.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import threading
import time

import pandas as pd
import requests
import websocket

REST_BASE = "https://data-api.binance.vision"
WS_BASE = "wss://data-stream.binance.vision"


class SequenceGapError(RuntimeError):
    """Raised when a diff event doesn't bridge from the previous update."""


class LocalOrderBook:
    """Maintains bids/asks from a REST snapshot + a stream of diff events,
    per Binance's documented local-book-management algorithm. Pure logic,
    no I/O -- the network layer feeds it snapshot/diff payloads.
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int | None = None

    def apply_snapshot(self, snapshot: dict) -> None:
        self.bids = {float(p): float(q) for p, q in snapshot["bids"]}
        self.asks = {float(p): float(q) for p, q in snapshot["asks"]}
        self.last_update_id = snapshot["lastUpdateId"]

    def should_discard(self, event: dict) -> bool:
        """True if this diff is entirely covered by the current snapshot."""
        return event["u"] <= self.last_update_id

    def is_first_valid_event(self, event: dict) -> bool:
        """Binance's bridging check: the first diff applied after a snapshot
        must straddle lastUpdateId+1 (U <= lastUpdateId+1 <= u)."""
        return event["U"] <= self.last_update_id + 1 <= event["u"]

    def apply_diff(self, event: dict) -> None:
        if self.last_update_id is None:
            raise RuntimeError("apply_snapshot must be called before apply_diff")
        if self.should_discard(event):
            return
        if self.last_update_id is not None and event["U"] > self.last_update_id + 1:
            raise SequenceGapError(
                f"Gap detected: expected U <= {self.last_update_id + 1}, got {event['U']}"
            )

        for price_str, qty_str in event["b"]:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price_str, qty_str in event["a"]:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty
        self.last_update_id = event["u"]

    def top_levels(self, n: int) -> dict:
        bid_prices = sorted(self.bids, reverse=True)[:n]
        ask_prices = sorted(self.asks)[:n]
        row = {}
        for i in range(n):
            row[f"bid_price_{i + 1}"] = bid_prices[i] if i < len(bid_prices) else float("nan")
            row[f"bid_size_{i + 1}"] = self.bids[bid_prices[i]] if i < len(bid_prices) else 0.0
            row[f"ask_price_{i + 1}"] = ask_prices[i] if i < len(ask_prices) else float("nan")
            row[f"ask_size_{i + 1}"] = self.asks[ask_prices[i]] if i < len(ask_prices) else 0.0
        return row


@dataclasses.dataclass
class CaptureResult:
    book_snapshots: pd.DataFrame
    trades: pd.DataFrame


def _fetch_rest_snapshot(symbol: str, limit: int = 1000) -> dict:
    resp = requests.get(
        f"{REST_BASE}/api/v3/depth", params={"symbol": symbol.upper(), "limit": limit}, timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def capture_session(
    symbol: str,
    duration_seconds: float,
    depth_levels: int = 10,
    diff_speed: str = "100ms",
    out_dir: pathlib.Path | str | None = None,
) -> CaptureResult:
    """Capture `duration_seconds` of real order book + trade data for
    `symbol` from Binance's public feeds, reconciled into a local book.

    Writes `book_snapshots.csv` and `trades.csv` under `out_dir` if given
    (both header-included, unlike LOBSTER's raw CSVs, since this is our
    own captured format rather than a fixed third-party schema).
    """
    symbol_lower = symbol.lower()
    stream_names = f"{symbol_lower}@depth@{diff_speed}/{symbol_lower}@trade"
    ws_url = f"{WS_BASE}/stream?streams={stream_names}"

    diff_buffer: list[dict] = []
    book_rows: list[dict] = []
    trade_rows: list[dict] = []
    book = LocalOrderBook()
    state = {"snapshot_applied": False, "started_at": None}
    lock = threading.Lock()

    def handle_depth_event(event: dict) -> None:
        with lock:
            if not state["snapshot_applied"]:
                diff_buffer.append(event)
                return
            if book.should_discard(event):
                return
            book.apply_diff(event)
            row = book.top_levels(depth_levels)
            row["time"] = event["E"] / 1000.0
            book_rows.append(row)

    def handle_trade_event(event: dict) -> None:
        trade_rows.append(
            {
                "time": event["E"] / 1000.0,
                "price": float(event["p"]),
                "size": float(event["q"]),
                "is_buyer_maker": event["m"],
            }
        )

    def on_message(_ws, message: str) -> None:
        payload = json.loads(message)
        data = payload.get("data", payload)
        if data.get("e") == "depthUpdate":
            handle_depth_event(data)
        elif data.get("e") == "trade":
            handle_trade_event(data)

    def on_error(_ws, error) -> None:
        raise RuntimeError(f"Binance websocket error: {error}")

    ws_app = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
    thread = threading.Thread(target=ws_app.run_forever, kwargs={"ping_interval": 20}, daemon=True)
    thread.start()
    state["started_at"] = time.monotonic()

    # Let diffs buffer briefly before pulling the snapshot, matching
    # Binance's documented ordering (open the stream first, then snapshot).
    time.sleep(1.0)
    snapshot = _fetch_rest_snapshot(symbol, limit=1000)

    with lock:
        book.apply_snapshot(snapshot)
        bridging = [e for e in diff_buffer if not book.should_discard(e)]
        if bridging and not book.is_first_valid_event(bridging[0]):
            raise SequenceGapError(
                "First buffered diff doesn't bridge the REST snapshot; "
                "retry the capture (this is expected to be rare)."
            )
        for e in bridging:
            book.apply_diff(e)
        state["snapshot_applied"] = True

    remaining = duration_seconds - (time.monotonic() - state["started_at"])
    if remaining > 0:
        time.sleep(remaining)
    ws_app.close()
    thread.join(timeout=5)

    book_df = pd.DataFrame(book_rows)
    trades_df = pd.DataFrame(trade_rows)

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        book_df.to_csv(out_dir / f"{symbol.upper()}_book_snapshots.csv", index=False)
        trades_df.to_csv(out_dir / f"{symbol.upper()}_trades.csv", index=False)

    return CaptureResult(book_snapshots=book_df, trades=trades_df)


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    out_dir = pathlib.Path(__file__).parent / "binance_sample"

    result = capture_session(symbol, duration, out_dir=out_dir)
    print(f"Captured {len(result.book_snapshots)} book rows, {len(result.trades)} trades")
    print(f"Written to {out_dir}")
