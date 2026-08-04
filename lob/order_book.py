"""Price-time priority limit order book.

Resting orders are kept in `SortedDict[price -> OrderedDict[order_id -> Order]]`
per side: the SortedDict gives O(log n) best-price lookup/insertion, and the
per-level OrderedDict gives O(1) FIFO time-priority (oldest order is always
first) plus O(1) cancel-by-order_id. This is the standard structure for a
research-scale (not co-located-HFT-scale) matching engine and is deliberately
plain Python/dataclasses rather than numba-JIT'd: correctness is what a
subtly wrong matching engine would silently cost every downstream result, so
this trades raw throughput for something easy to read and easy to prove
correct against the tests in tests/test_order_book.py. If replay throughput
ever becomes the actual bottleneck (profile before assuming it will), the
matching loop here is the place to rewrite with primitive arrays under numba
-- nothing above this module would need to change.

This module only knows about `arrival_time`: it has no concept of when a
strategy *decided* to submit or cancel an order. That separation is enforced
one layer up, in `lob/engine.py`, which is what lets Step 8 slot in a
stochastic latency model later without touching matching logic at all.
"""

from __future__ import annotations

from collections import OrderedDict

from sortedcontainers import SortedDict

from lob.models import Fill, Order, Side


class OrderBook:
    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size
        # Both sides stored ascending by price. Best bid = max price =
        # bids.peekitem(-1); best ask = min price = asks.peekitem(0).
        self.bids: SortedDict[float, OrderedDict[int, Order]] = SortedDict()
        self.asks: SortedDict[float, OrderedDict[int, Order]] = SortedDict()
        self._order_index: dict[int, Order] = {}

    def best_bid(self) -> float | None:
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def _book_side(self, side: Side) -> SortedDict:
        return self.bids if side == Side.BUY else self.asks

    def _opposite_side(self, side: Side) -> SortedDict:
        return self.asks if side == Side.BUY else self.bids

    def _rest_order(self, order: Order) -> None:
        level = self._book_side(order.side).setdefault(order.price, OrderedDict())
        level[order.order_id] = order
        self._order_index[order.order_id] = order

    def _match_against(
        self,
        opposite: SortedDict,
        remaining: int,
        order_id: int,
        taker_side: Side,
        arrival_time: float,
        price_limit: float | None,
        best_price_fn,
    ) -> tuple[int, list[Fill]]:
        """Walk `opposite` from the best price outward, filling FIFO within
        each level, until `remaining` hits 0, the book is exhausted, or (for
        a limit order) `price_limit` is no longer marketable.
        """
        fills: list[Fill] = []
        maker_side = Side.SELL if taker_side == Side.BUY else Side.BUY

        while remaining > 0 and opposite:
            level_price = best_price_fn(opposite)
            if price_limit is not None:
                crosses = level_price <= price_limit if taker_side == Side.BUY else level_price >= price_limit
                if not crosses:
                    break

            level = opposite[level_price]
            while remaining > 0 and level:
                maker_id, maker_order = next(iter(level.items()))
                trade_size = min(remaining, maker_order.size)
                fills.append(
                    Fill(
                        time=arrival_time,
                        price=level_price,
                        size=trade_size,
                        maker_order_id=maker_id,
                        taker_order_id=order_id,
                        maker_side=maker_side,
                        taker_side=taker_side,
                    )
                )
                maker_order.size -= trade_size
                remaining -= trade_size
                if maker_order.size == 0:
                    del level[maker_id]
                    del self._order_index[maker_id]

            if not level:
                del opposite[level_price]

        return remaining, fills

    def submit_limit_order(
        self,
        order_id: int,
        side: Side,
        price: float,
        size: int,
        decision_time: float,
        arrival_time: float,
    ) -> list[Fill]:
        opposite = self._opposite_side(side)
        best_price_fn = (lambda s: s.peekitem(0)[0]) if side == Side.BUY else (lambda s: s.peekitem(-1)[0])

        remaining, fills = self._match_against(
            opposite, size, order_id, side, arrival_time, price_limit=price, best_price_fn=best_price_fn
        )

        if remaining > 0:
            self._rest_order(Order(order_id, side, price, remaining, decision_time, arrival_time))

        return fills

    def submit_market_order(
        self,
        order_id: int,
        side: Side,
        size: int,
        decision_time: float,
        arrival_time: float,
    ) -> list[Fill]:
        opposite = self._opposite_side(side)
        best_price_fn = (lambda s: s.peekitem(0)[0]) if side == Side.BUY else (lambda s: s.peekitem(-1)[0])

        # Market orders never rest: any unfilled remainder after the book
        # is exhausted is simply dropped (returned fills reflect what
        # actually executed; callers can infer the unfilled amount as
        # size - sum(f.size for f in fills)).
        _, fills = self._match_against(
            opposite, size, order_id, side, arrival_time, price_limit=None, best_price_fn=best_price_fn
        )
        return fills

    def cancel_order(self, order_id: int, arrival_time: float) -> bool:
        """No-op (returns False) if the order doesn't exist -- already
        filled or already canceled. The event stream (see data/synthetic_lob.py)
        can and does emit cancels for orders it doesn't know were already
        filled, since the generator has no matching logic of its own.
        """
        order = self._order_index.get(order_id)
        if order is None:
            return False
        del self._order_index[order_id]
        book_side = self._book_side(order.side)
        level = book_side[order.price]
        del level[order.order_id]
        if not level:
            del book_side[order.price]
        return True

    def has_order(self, order_id: int) -> bool:
        """Whether `order_id` is currently resting -- used by callers (e.g.
        backtest.market_maker_sim) that need to know if their own quote is
        still live after a submission that may have partially or fully
        filled immediately, without reaching into book internals.
        """
        return order_id in self._order_index

    def depth(self, n_levels: int = 5) -> dict[str, list[tuple[float, int]]]:
        """Aggregate resting size per price level, best-first, up to
        `n_levels` per side.

        Uses SortedDict's key-view slicing to grab only the top `n_levels`
        prices directly, rather than materializing every resting price
        level on the book -- this runs once per replayed event (Step 3
        records book state after every event), so it needs to stay
        O(n_levels), not O(all resting levels).
        """
        bid_prices = list(reversed(self.bids.keys()[-n_levels:])) if self.bids else []
        ask_prices = list(self.asks.keys()[:n_levels]) if self.asks else []
        bid_levels = [(p, sum(o.size for o in self.bids[p].values())) for p in bid_prices]
        ask_levels = [(p, sum(o.size for o in self.asks[p].values())) for p in ask_prices]
        return {"bids": bid_levels, "asks": ask_levels}

    def top_levels(self, n_levels: int) -> dict[str, float]:
        """Flat {bid_price_i, bid_size_i, ask_price_i, ask_size_i} snapshot,
        best-first, padded with NaN price / 0 size for levels the book
        doesn't have. Deliberately the same shape as
        data/binance_capture.py's LocalOrderBook.top_levels, so book state
        from either data source can feed the same replay/feature code
        (Step 11 compares strategies across both).
        """
        d = self.depth(n_levels)
        row: dict[str, float] = {}
        for i in range(n_levels):
            bid_price, bid_size = d["bids"][i] if i < len(d["bids"]) else (float("nan"), 0)
            ask_price, ask_size = d["asks"][i] if i < len(d["asks"]) else (float("nan"), 0)
            row[f"bid_price_{i + 1}"] = bid_price
            row[f"bid_size_{i + 1}"] = bid_size
            row[f"ask_price_{i + 1}"] = ask_price
            row[f"ask_size_{i + 1}"] = ask_size
        return row
