"""Core order/fill data types shared by the order book and the event loop."""

from __future__ import annotations

import dataclasses
import enum


class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class EventType(enum.Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


@dataclasses.dataclass
class Order:
    """A resting limit order sitting in the book. `size` is remaining
    (unfilled) size and mutates in place as partial fills happen -- the
    same Order instance stays at its FIFO queue position for the rest of
    its resting life, which is exactly the time-priority guarantee.
    """

    order_id: int
    side: Side
    price: float
    size: int
    decision_time: float
    arrival_time: float


@dataclasses.dataclass
class Fill:
    """One trade produced by matching an incoming order against a resting
    (maker) order. `time` is the arrival_time at which the match happened,
    not the maker's or taker's decision_time -- fills only ever happen at
    the moment the book actually processes an event.
    """

    time: float
    price: float
    size: int
    maker_order_id: int
    taker_order_id: int
    maker_side: Side
    taker_side: Side
