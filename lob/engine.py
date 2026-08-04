"""Event loop driving an OrderBook from a stream of decision events.

This is where the decision-time vs. arrival-time separation lives, by
design from the start rather than retrofitted for Step 8: every event in
the input stream has a `time` (when a strategy *decided* to act), and a
`latency_model` maps that to an `arrival_time` (when the order actually
reaches the book). The book only ever sees arrival order.

With the default zero-latency model, arrival_time == decision_time and the
loop reduces to "process events in the order they were decided" -- but the
sort-by-arrival-time step below still runs. That matters even at zero
latency: it's what keeps this code path identical to the one Step 8 will
exercise with a real stochastic model, so introducing latency later means
swapping the model, not rewriting the loop.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import pandas as pd

from lob.models import EventType, Fill, Side
from lob.order_book import OrderBook

LatencyModel = Callable[[float], float]


def zero_latency(decision_time: float) -> float:
    return decision_time


@dataclasses.dataclass
class ReplayResult:
    fills: list[Fill]
    unknown_cancels: int  # cancels that targeted an already-gone order_id


class MatchingEngine:
    def __init__(self, tick_size: float = 0.01, latency_model: LatencyModel | None = None) -> None:
        self.book = OrderBook(tick_size=tick_size)
        self.latency_model = latency_model or zero_latency

    def replay(self, events: pd.DataFrame) -> ReplayResult:
        """Replay a decision-event stream (order_id, time, type, side, price,
        size -- the schema data/synthetic_lob.py produces) through the book.

        `time` is treated as decision_time. Events are re-sorted by the
        arrival_time the latency model assigns them (stable, decision-time
        then order_id as tiebreaks) before processing, so a later decision
        with lower latency can legitimately reach the book before an
        earlier one with higher latency.
        """
        records = events.to_dict("records")
        for rec in records:
            rec["arrival_time"] = self.latency_model(rec["time"])
        records.sort(key=lambda r: (r["arrival_time"], r["time"], r["order_id"]))

        fills: list[Fill] = []
        unknown_cancels = 0

        for rec in records:
            event_type = EventType(rec["type"])
            decision_time = rec["time"]
            arrival_time = rec["arrival_time"]
            order_id = int(rec["order_id"])

            if event_type is EventType.CANCEL:
                found = self.book.cancel_order(order_id, arrival_time)
                if not found:
                    unknown_cancels += 1
                continue

            side = Side(rec["side"])
            size = int(rec["size"])

            if event_type is EventType.LIMIT:
                fills.extend(
                    self.book.submit_limit_order(
                        order_id, side, float(rec["price"]), size, decision_time, arrival_time
                    )
                )
            elif event_type is EventType.MARKET:
                fills.extend(
                    self.book.submit_market_order(order_id, side, size, decision_time, arrival_time)
                )

        return ReplayResult(fills=fills, unknown_cancels=unknown_cancels)
