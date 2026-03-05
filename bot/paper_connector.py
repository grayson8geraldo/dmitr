"""
Paper Trading Exchange Connector.
Uses real market data but simulates orders with a virtual balance.
No API keys required for market data (public endpoints).
"""

import asyncio
import time
import uuid
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from config import ExchangeConfig


class PaperExchangeConnector:
    """
    Paper trading connector: real market data, virtual execution.
    - Fetches live OHLCV, tickers, funding rates from exchange.
    - Simulates order fills at current market price.
    - Tracks virtual balance and positions internally.
    """

    def __init__(self, config: ExchangeConfig, initial_balance: float = 100.0):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None

        # Virtual state
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: list[dict] = []
        self._order_history: list[dict] = []
        self._trade_log: list[dict] = []

    async def connect(self):
        """Connect to exchange for public data only (no auth needed)."""
        exchange_class = getattr(ccxt, self.config.name, None)
        if exchange_class is None:
            raise ValueError(f"Exchange {self.config.name} not supported")

        options = {
            "defaultType": "swap",
            "adjustForTimeDifference": True,
        }

        if self.config.name == "bybit":
            options["defaultSettle"] = "USDT"

        # No API keys — public data only
        self.exchange = exchange_class({
            "options": options,
            "enableRateLimit": True,
        })

        await self.exchange.load_markets()
        logger.info(
            f"[PAPER] Connected to {self.config.name} (public data) | "
            f"Markets: {len(self.exchange.markets)} | "
            f"Virtual balance: ${self._balance:.2f}"
        )

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
        self._print_paper_summary()

    def _print_paper_summary(self):
        """Print paper trading session summary."""
        pnl = self._balance - self._initial_balance
        pnl_pct = (pnl / self._initial_balance) * 100 if self._initial_balance > 0 else 0
        wins = sum(1 for t in self._trade_log if t["pnl"] > 0)
        losses = sum(1 for t in self._trade_log if t["pnl"] <= 0)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

        logger.info("=" * 60)
        logger.info("  PAPER TRADING SESSION SUMMARY")
        logger.info(f"  Start balance: ${self._initial_balance:.2f}")
        logger.info(f"  Final balance: ${self._balance:.2f}")
        logger.info(f"  Total PnL:     ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"  Trades:        {total} (W:{wins} / L:{losses})")
        logger.info(f"  Win rate:      {win_rate:.0f}%")
        logger.info("=" * 60)

    # ─── Account (Virtual) ─────────────────────────────────────────

    async def setup_trading_mode(self, symbol: str):
        """No-op for paper trading — just log it."""
        logger.debug(f"[PAPER] Trading mode set for {symbol} (simulated)")

    async def get_balance(self) -> float:
        """Return virtual total balance (free + in positions margin)."""
        margin_in_use = sum(
            p["margin"] for p in self._positions if p["status"] == "open"
        )
        return self._balance + margin_in_use

    async def get_free_balance(self) -> float:
        """Return virtual free balance."""
        return self._balance

    # ─── Market Data (Real) ────────────────────────────────────────

    def _swap_symbol(self, symbol: str) -> str:
        """Convert spot symbol to linear swap symbol for Bybit futures."""
        if self.config.name == "bybit" and ":USDT" not in symbol:
            swap = f"{symbol}:USDT"
            if swap in self.exchange.markets:
                return swap
        return symbol

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "5m", limit: int = 300
    ) -> pd.DataFrame:
        """Fetch real OHLCV data from exchange with rate limit retry."""
        swap = self._swap_symbol(symbol)
        for attempt in range(3):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(swap, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
            except ccxt.RateLimitExceeded:
                wait = 2 ** (attempt + 1)
                logger.debug(f"Rate limited on {symbol}, retry in {wait}s")
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
                return pd.DataFrame()
        logger.warning(f"Rate limit persists for {symbol}, skipping")
        return pd.DataFrame()

    async def get_ticker(self, symbol: str) -> dict:
        """Fetch real ticker data."""
        try:
            ticker = await self.exchange.fetch_ticker(self._swap_symbol(symbol))
            return {
                "price": float(ticker.get("last", 0)),
                "bid": float(ticker.get("bid", 0)),
                "ask": float(ticker.get("ask", 0)),
                "volume_24h": float(ticker.get("quoteVolume", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {"price": 0, "bid": 0, "ask": 0, "volume_24h": 0}

    async def get_funding_rate(self, symbol: str) -> float:
        """Fetch real funding rate."""
        try:
            funding = await self.exchange.fetch_funding_rate(self._swap_symbol(symbol))
            return float(funding.get("fundingRate", 0))
        except Exception as e:
            logger.debug(f"Failed to fetch funding rate for {symbol}: {e}")
            return 0.0

    # ─── Order Execution (Simulated) ───────────────────────────────

    async def open_long(
        self,
        symbol: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[dict]:
        """Simulate opening a long position."""
        ticker = await self.get_ticker(symbol)
        price = ticker["ask"] if ticker["ask"] > 0 else ticker["price"]
        if price <= 0:
            logger.error(f"[PAPER] Cannot open LONG {symbol}: no price data")
            return None

        position_usd = amount * price
        margin = position_usd / self.config.default_leverage

        if margin > self._balance:
            logger.warning(
                f"[PAPER] Insufficient balance for LONG {symbol}: "
                f"need ${margin:.2f}, have ${self._balance:.2f}"
            )
            return None

        self._balance -= margin

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        position = {
            "id": order_id,
            "symbol": symbol,
            "side": "long",
            "amount": amount,
            "entry_price": price,
            "position_usd": position_usd,
            "margin": margin,
            "leverage": self.config.default_leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "opened_at": time.time(),
        }
        self._positions.append(position)

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "amount": amount,
            "price": price,
            "status": "closed",
            "filled": amount,
        }
        self._order_history.append(order)

        logger.info(
            f"[PAPER] LONG opened | {symbol} | "
            f"Price: {price:.4f} | Amount: {amount} | "
            f"Size: ${position_usd:.2f} | Margin: ${margin:.2f} | "
            f"SL: {stop_loss:.4f} | TP: {take_profit:.4f}"
        )
        return order

    async def open_short(
        self,
        symbol: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[dict]:
        """Simulate opening a short position."""
        ticker = await self.get_ticker(symbol)
        price = ticker["bid"] if ticker["bid"] > 0 else ticker["price"]
        if price <= 0:
            logger.error(f"[PAPER] Cannot open SHORT {symbol}: no price data")
            return None

        position_usd = amount * price
        margin = position_usd / self.config.default_leverage

        if margin > self._balance:
            logger.warning(
                f"[PAPER] Insufficient balance for SHORT {symbol}: "
                f"need ${margin:.2f}, have ${self._balance:.2f}"
            )
            return None

        self._balance -= margin

        order_id = f"paper_{uuid.uuid4().hex[:12]}"
        position = {
            "id": order_id,
            "symbol": symbol,
            "side": "short",
            "amount": amount,
            "entry_price": price,
            "position_usd": position_usd,
            "margin": margin,
            "leverage": self.config.default_leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "opened_at": time.time(),
        }
        self._positions.append(position)

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": "sell",
            "type": "market",
            "amount": amount,
            "price": price,
            "status": "closed",
            "filled": amount,
        }
        self._order_history.append(order)

        logger.info(
            f"[PAPER] SHORT opened | {symbol} | "
            f"Price: {price:.4f} | Amount: {amount} | "
            f"Size: ${position_usd:.2f} | Margin: ${margin:.2f} | "
            f"SL: {stop_loss:.4f} | TP: {take_profit:.4f}"
        )
        return order

    async def close_position(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Optional[dict]:
        """Simulate closing a position: calculate PnL at current price."""
        ticker = await self.get_ticker(symbol)
        if side == "long":
            price = ticker["bid"] if ticker["bid"] > 0 else ticker["price"]
        else:
            price = ticker["ask"] if ticker["ask"] > 0 else ticker["price"]

        if price <= 0:
            logger.error(f"[PAPER] Cannot close {side} {symbol}: no price data")
            return None

        # Find matching open position
        pos = None
        for p in self._positions:
            if p["symbol"] == symbol and p["side"] == side and p["status"] == "open":
                pos = p
                break

        if pos is None:
            logger.warning(f"[PAPER] No open {side} position for {symbol}")
            return None

        # Calculate PnL
        if side == "long":
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * pos["position_usd"]
        else:
            pnl = (pos["entry_price"] - price) / pos["entry_price"] * pos["position_usd"]

        # Return margin + PnL
        self._balance += pos["margin"] + pnl
        pos["status"] = "closed"
        pos["exit_price"] = price
        pos["pnl"] = pnl
        pos["closed_at"] = time.time()

        self._trade_log.append({
            "symbol": symbol,
            "side": side,
            "entry": pos["entry_price"],
            "exit": price,
            "size": pos["position_usd"],
            "pnl": pnl,
            "duration_s": pos["closed_at"] - pos["opened_at"],
        })

        order = {
            "id": f"paper_{uuid.uuid4().hex[:12]}",
            "symbol": symbol,
            "side": "sell" if side == "long" else "buy",
            "type": "market",
            "amount": amount,
            "price": price,
            "status": "closed",
            "filled": amount,
        }
        self._order_history.append(order)

        pnl_pct = (pnl / pos["position_usd"]) * 100
        logger.info(
            f"[PAPER] {side.upper()} closed | {symbol} | "
            f"Entry: {pos['entry_price']:.4f} -> Exit: {price:.4f} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"Balance: ${self._balance:.2f}"
        )
        return order

    async def get_open_positions(self) -> list[dict]:
        """Return virtual open positions in exchange format."""
        result = []
        for pos in self._positions:
            if pos["status"] != "open":
                continue

            ticker = await self.get_ticker(pos["symbol"])
            current_price = ticker["price"]

            if pos["side"] == "long":
                unrealized_pnl = (
                    (current_price - pos["entry_price"])
                    / pos["entry_price"]
                    * pos["position_usd"]
                )
            else:
                unrealized_pnl = (
                    (pos["entry_price"] - current_price)
                    / pos["entry_price"]
                    * pos["position_usd"]
                )

            result.append({
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["amount"],
                "entry_price": pos["entry_price"],
                "unrealized_pnl": unrealized_pnl,
                "leverage": pos["leverage"],
                "margin_mode": "cross",
            })
        return result

    # ─── Utility ───────────────────────────────────────────────────

    async def get_min_order_amount(self, symbol: str) -> float:
        """Get minimum order amount for a symbol."""
        market = self.exchange.market(self._swap_symbol(symbol))
        if market and "limits" in market:
            return float(market["limits"]["amount"]["min"] or 0)
        return 0.0

    async def calculate_amount(
        self, symbol: str, position_size_usd: float, price: float
    ) -> float:
        """Calculate order amount in base currency from USD size."""
        if price <= 0:
            return 0.0

        amount = position_size_usd / price
        min_amount = await self.get_min_order_amount(symbol)
        if amount < min_amount:
            logger.warning(
                f"[PAPER] Amount {amount} below minimum {min_amount} for {symbol}"
            )
            return 0.0

        swap_sym = self._swap_symbol(symbol)
        market = self.exchange.market(swap_sym)
        if market:
            precision = market.get("precision", {}).get("amount", 8)
            if isinstance(precision, int):
                amount = round(amount, precision)
            else:
                amount = float(
                    self.exchange.amount_to_precision(swap_sym, amount)
                )

        return amount

    def get_trade_log(self) -> list[dict]:
        """Return full trade history for analysis."""
        return self._trade_log.copy()
