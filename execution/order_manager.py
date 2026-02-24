"""
Order manager — Alpaca paper + live execution.

In paper mode (default): all orders are logged, not sent.
In live mode (ALPACA_BASE_URL != paper-api): real orders sent via Alpaca REST.

Each strategy submits orders independently.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    symbol: str
    side: str          # "buy" or "sell"
    qty: float
    notional: float    # USD value
    success: bool
    paper: bool
    order_id: Optional[str] = None
    message: str = ""

    def __str__(self) -> str:
        mode = "[PAPER]" if self.paper else "[LIVE]"
        status = "OK" if self.success else "FAIL"
        return (f"{mode} {status} {self.side.upper()} {self.symbol}  "
                f"qty={self.qty:.4f}  notional=${self.notional:.2f}  {self.message}")


class OrderManager:
    def __init__(self, paper: bool = True):
        self.paper = paper or (settings.ALPACA_BASE_URL and "paper" in settings.ALPACA_BASE_URL)
        self._client = None
        if not self.paper:
            self._client = self._init_client()

    def _init_client(self):
        try:
            from alpaca.trading.client import TradingClient
            return TradingClient(
                settings.ALPACA_API_KEY,
                settings.ALPACA_SECRET_KEY,
                paper=False,
            )
        except Exception as exc:
            logger.error("Alpaca client init failed: %s", exc)
            return None

    def place_market_order(self, symbol: str, side: str, notional: float) -> OrderResult:
        """
        Place a market order for a given USD notional.
        side: "buy" or "sell"
        """
        if self.paper:
            logger.info("[PAPER] %s %s $%.2f", side.upper(), symbol, notional)
            return OrderResult(
                symbol=symbol, side=side, qty=0, notional=notional,
                success=True, paper=True, order_id=f"paper_{symbol}_{side}",
            )

        if self._client is None:
            return OrderResult(symbol=symbol, side=side, qty=0, notional=notional,
                               success=False, paper=False, message="Client not initialized")

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            return OrderResult(
                symbol=symbol, side=side, qty=float(order.qty or 0),
                notional=notional, success=True, paper=False,
                order_id=str(order.id),
            )
        except Exception as exc:
            logger.error("Order failed %s %s: %s", side, symbol, exc)
            return OrderResult(symbol=symbol, side=side, qty=0, notional=notional,
                               success=False, paper=False, message=str(exc))

    def get_positions(self) -> dict[str, float]:
        """Return current positions as {ticker: market_value}."""
        if self.paper:
            return {}
        if self._client is None:
            return {}
        try:
            positions = self._client.get_all_positions()
            return {p.symbol: float(p.market_value) for p in positions}
        except Exception as exc:
            logger.error("Get positions failed: %s", exc)
            return {}

    def close_all(self) -> int:
        """Close all open positions. Returns count closed."""
        if self.paper:
            logger.info("[PAPER] close_all() called")
            return 0
        if self._client is None:
            return 0
        try:
            self._client.close_all_positions(cancel_orders=True)
            logger.info("All positions closed")
            return 1
        except Exception as exc:
            logger.error("close_all failed: %s", exc)
            return 0
