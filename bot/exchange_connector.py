"""
Exchange Connector Module.
Handles all communication with the exchange via CCXT.
Supports Bybit and Binance futures.
"""

import asyncio
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from config import ExchangeConfig


class ExchangeConnector:
    """
    Async exchange connector using CCXT.
    Configures cross-margin, hedge mode, and handles order execution.
    """

    def __init__(
        self,
        config: ExchangeConfig,
        api_key: str,
        api_secret: str,
    ):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.api_key = api_key
        self.api_secret = api_secret

    async def connect(self):
        """Initialize exchange connection with proper settings."""
        exchange_class = getattr(ccxt, self.config.name, None)
        if exchange_class is None:
            raise ValueError(f"Exchange {self.config.name} not supported by CCXT")

        options = {
            "defaultType": "swap",  # Futures/perpetual
            "adjustForTimeDifference": True,
        }

        if self.config.name == "bybit":
            options["defaultSettle"] = "USDT"

        self.exchange = exchange_class({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "options": options,
            "enableRateLimit": True,
        })

        if self.config.testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("TESTNET mode enabled")

        await self.exchange.load_markets()
        logger.info(
            f"Connected to {self.config.name} | "
            f"Markets loaded: {len(self.exchange.markets)}"
        )

    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()

    # ─── Account & Configuration ───────────────────────────────────────

    async def setup_trading_mode(self, symbol: str):
        """
        Configure cross-margin and hedge mode for a symbol.
        Must be done before first trade.
        """
        try:
            # Set leverage
            await self.exchange.set_leverage(
                self.config.default_leverage, symbol
            )
            logger.debug(
                f"Leverage set to {self.config.default_leverage}x for {symbol}"
            )
        except Exception as e:
            logger.debug(f"Set leverage for {symbol}: {e}")

        try:
            # Set margin mode to cross
            await self.exchange.set_margin_mode(
                "cross", symbol
            )
            logger.debug(f"Cross margin set for {symbol}")
        except Exception as e:
            logger.debug(f"Set margin mode for {symbol}: {e}")

        try:
            # Set position mode to hedge (both sides)
            if self.config.name == "bybit":
                await self.exchange.set_position_mode(True, symbol)
            elif self.config.name == "binance":
                await self.exchange.fapiPrivate_post_positionside_dual(
                    {"dualSidePosition": "true"}
                )
            logger.debug(f"Hedge mode set for {symbol}")
        except Exception as e:
            logger.debug(f"Set position mode for {symbol}: {e}")

    async def get_balance(self) -> float:
        """Get total USDT balance."""
        try:
            balance = await self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total = float(usdt.get("total", 0))
            return total
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    async def get_free_balance(self) -> float:
        """Get available (free) USDT balance."""
        try:
            balance = await self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            free = float(usdt.get("free", 0))
            return free
        except Exception as e:
            logger.error(f"Failed to fetch free balance: {e}")
            return 0.0

    # ─── Market Data ───────────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 300,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles as DataFrame."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
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
        """Get current funding rate for a symbol."""
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0))
        except Exception as e:
            logger.debug(f"Failed to fetch funding rate for {symbol}: {e}")
            return 0.0

    # ─── Order Execution ───────────────────────────────────────────────

    async def open_long(
        self,
        symbol: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[dict]:
        """
        Open a long position with SL/TP.
        Uses market order for immediate fill.
        """
        try:
            params = {
                "stopLoss": {"triggerPrice": stop_loss, "type": "market"},
                "takeProfit": {"triggerPrice": take_profit, "type": "market"},
                "positionSide": "long",
            }

            order = await self.exchange.create_market_buy_order(
                symbol, amount, params=params
            )

            logger.info(
                f"LONG order placed | {symbol} | "
                f"Amount: {amount} | "
                f"SL: {stop_loss} | TP: {take_profit} | "
                f"Order ID: {order['id']}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to open LONG {symbol}: {e}")
            return None

    async def open_short(
        self,
        symbol: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> Optional[dict]:
        """
        Open a short position with SL/TP.
        Uses market order for immediate fill.
        """
        try:
            params = {
                "stopLoss": {"triggerPrice": stop_loss, "type": "market"},
                "takeProfit": {"triggerPrice": take_profit, "type": "market"},
                "positionSide": "short",
            }

            order = await self.exchange.create_market_sell_order(
                symbol, amount, params=params
            )

            logger.info(
                f"SHORT order placed | {symbol} | "
                f"Amount: {amount} | "
                f"SL: {stop_loss} | TP: {take_profit} | "
                f"Order ID: {order['id']}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to open SHORT {symbol}: {e}")
            return None

    async def close_position(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Optional[dict]:
        """Close a position by placing an opposite market order."""
        try:
            params = {"positionSide": side}
            if side == "long":
                order = await self.exchange.create_market_sell_order(
                    symbol, amount, params=params
                )
            else:
                order = await self.exchange.create_market_buy_order(
                    symbol, amount, params=params
                )

            logger.info(
                f"Position closed | {symbol} {side} | "
                f"Amount: {amount} | Order ID: {order['id']}"
            )
            return order

        except Exception as e:
            logger.error(f"Failed to close {side} {symbol}: {e}")
            return None

    async def get_open_positions(self) -> list[dict]:
        """Get all open positions from exchange."""
        try:
            positions = await self.exchange.fetch_positions()
            open_positions = []
            for pos in positions:
                size = float(pos.get("contracts", 0))
                if size > 0:
                    open_positions.append({
                        "symbol": pos["symbol"],
                        "side": pos["side"],
                        "size": size,
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(
                            pos.get("unrealizedPnl", 0)
                        ),
                        "leverage": int(pos.get("leverage", 1)),
                        "margin_mode": pos.get("marginMode", ""),
                    })
            return open_positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    # ─── Utility ───────────────────────────────────────────────────────

    async def get_min_order_amount(self, symbol: str) -> float:
        """Get minimum order amount for a symbol."""
        market = self.exchange.market(symbol)
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
                f"Calculated amount {amount} below minimum {min_amount} "
                f"for {symbol}"
            )
            return 0.0

        # Round to market precision
        market = self.exchange.market(symbol)
        if market:
            precision = market.get("precision", {}).get("amount", 8)
            if isinstance(precision, int):
                amount = round(amount, precision)
            else:
                amount = float(
                    self.exchange.amount_to_precision(symbol, amount)
                )

        return amount
