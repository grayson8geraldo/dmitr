"""
Market Analysis Module.
Analyzes liquidations, Open Interest, CVD, and volatility
to find optimal entry points per the strategy.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
from loguru import logger


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class MarketState:
    """Current market state for a symbol."""
    symbol: str
    price: float
    price_change_1h_pct: float
    price_change_24h_pct: float
    volume_24h_usd: float
    volatility_pct: float

    # Open Interest data
    open_interest_usd: float
    oi_change_pct: float  # OI change over analysis period

    # Liquidation data
    long_liquidations_usd: float   # Recent long liquidations
    short_liquidations_usd: float  # Recent short liquidations

    # CVD (Cumulative Volume Delta)
    cvd_trend: float  # Positive = buying pressure, Negative = selling pressure

    # Funding rate
    funding_rate: float

    # Derived signal
    signal: Signal = Signal.NONE
    signal_strength: float = 0.0  # 0 to 1
    signal_reason: str = ""


class MarketAnalyzer:
    """
    Analyzes market conditions to find liquidation-based entry points.

    Strategy:
    - SHORT: After pump, when short liquidations cascade, OI starts declining,
             CVD shows selling pressure.
    - LONG: After dump, when long liquidations cascade, price finds support,
            CVD shows buying pressure.
    """

    def __init__(self, strategy_config, coinglass_api_key: str = ""):
        self.config = strategy_config
        self.coinglass_api_key = coinglass_api_key
        self.coinglass_base_url = "https://open-api-v3.coinglass.com/api"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.coinglass_api_key:
                headers["CG-API-KEY"] = self.coinglass_api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─── CoinGlass Data Fetching ───────────────────────────────────────

    async def fetch_liquidation_data(self, symbol: str) -> dict:
        """Fetch recent liquidation data from CoinGlass."""
        try:
            session = await self._get_session()
            coin = symbol.split("/")[0]
            url = f"{self.coinglass_base_url}/futures/liquidation/detail"
            params = {"symbol": coin, "timeType": "1"}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        return data["data"]
        except Exception as e:
            logger.debug(f"CoinGlass liquidation fetch failed for {symbol}: {e}")

        return {"longLiquidationUsd": 0, "shortLiquidationUsd": 0}

    async def fetch_open_interest(self, symbol: str) -> dict:
        """Fetch open interest data from CoinGlass."""
        try:
            session = await self._get_session()
            coin = symbol.split("/")[0]
            url = f"{self.coinglass_base_url}/futures/openInterest/ohlc-history"
            params = {"symbol": coin, "timeType": "5m", "limit": 30}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        return data["data"]
        except Exception as e:
            logger.debug(f"CoinGlass OI fetch failed for {symbol}: {e}")

        return {}

    async def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        try:
            session = await self._get_session()
            coin = symbol.split("/")[0]
            url = f"{self.coinglass_base_url}/futures/funding/current"
            params = {"symbol": coin}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and data.get("data"):
                        rates = data["data"]
                        if rates:
                            return float(rates[0].get("rate", 0))
        except Exception as e:
            logger.debug(f"CoinGlass funding fetch failed for {symbol}: {e}")

        return 0.0

    # ─── Exchange OHLCV Analysis ───────────────────────────────────────

    def analyze_ohlcv(self, df: pd.DataFrame) -> dict:
        """
        Analyze OHLCV data for volatility, trend, and CVD.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if df.empty or len(df) < 10:
            return {
                "volatility": 0,
                "price_change_1h": 0,
                "price_change_24h": 0,
                "cvd_trend": 0,
                "volume_24h": 0,
            }

        close = df["close"].values
        volume = df["volume"].values
        open_prices = df["open"].values
        high = df["high"].values
        low = df["low"].values

        # Volatility: average true range as % of price
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        volatility = atr / close[-1] if close[-1] > 0 else 0

        # Price changes
        price_change_1h = 0
        if len(close) >= 12:  # 12 x 5min = 1 hour
            price_change_1h = (close[-1] - close[-12]) / close[-12]

        price_change_24h = 0
        if len(close) >= 288:  # 288 x 5min = 24 hours
            price_change_24h = (close[-1] - close[-288]) / close[-288]
        elif len(close) >= 2:
            price_change_24h = (close[-1] - close[0]) / close[0]

        # CVD approximation using volume delta
        # Positive candle (close > open) = buying volume
        # Negative candle (close < open) = selling volume
        deltas = []
        for i in range(len(close)):
            if close[i] >= open_prices[i]:
                deltas.append(volume[i])
            else:
                deltas.append(-volume[i])

        cvd = np.cumsum(deltas)
        # CVD trend: compare last N periods
        n = min(self.config.cvd_reversal_periods, len(cvd))
        if n >= 2:
            cvd_recent = cvd[-n:]
            cvd_trend = (cvd_recent[-1] - cvd_recent[0]) / (
                abs(cvd_recent[0]) + 1e-10
            )
        else:
            cvd_trend = 0

        volume_24h = float(np.sum(volume) * close[-1])

        return {
            "volatility": float(volatility),
            "price_change_1h": float(price_change_1h),
            "price_change_24h": float(price_change_24h),
            "cvd_trend": float(cvd_trend),
            "volume_24h": volume_24h,
            "current_price": float(close[-1]),
        }

    # ─── Signal Generation ─────────────────────────────────────────────

    async def analyze_symbol(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
    ) -> MarketState:
        """
        Full analysis of a symbol. Returns MarketState with signal.
        """
        # Analyze price data
        ohlcv_analysis = self.analyze_ohlcv(ohlcv_df)

        # Fetch on-chain/derivatives data in parallel
        liq_data, oi_data, funding = await asyncio.gather(
            self.fetch_liquidation_data(symbol),
            self.fetch_open_interest(symbol),
            self.fetch_funding_rate(symbol),
        )

        # Parse liquidation data
        long_liqs = float(liq_data.get("longLiquidationUsd", 0))
        short_liqs = float(liq_data.get("shortLiquidationUsd", 0))

        # Parse OI change
        oi_change_pct = 0.0
        if isinstance(oi_data, list) and len(oi_data) >= 2:
            oi_start = float(oi_data[0].get("o", 0))
            oi_end = float(oi_data[-1].get("c", 0))
            if oi_start > 0:
                oi_change_pct = (oi_end - oi_start) / oi_start

        state = MarketState(
            symbol=symbol,
            price=ohlcv_analysis["current_price"],
            price_change_1h_pct=ohlcv_analysis["price_change_1h"],
            price_change_24h_pct=ohlcv_analysis["price_change_24h"],
            volume_24h_usd=ohlcv_analysis["volume_24h"],
            volatility_pct=ohlcv_analysis["volatility"],
            open_interest_usd=0,
            oi_change_pct=oi_change_pct,
            long_liquidations_usd=long_liqs,
            short_liquidations_usd=short_liqs,
            cvd_trend=ohlcv_analysis["cvd_trend"],
            funding_rate=funding,
        )

        # Generate signal
        state = self._generate_signal(state)

        return state

    def _generate_signal(self, state: MarketState) -> MarketState:
        """
        Generate trading signal based on liquidation strategy.

        SHORT signal conditions:
        1. Recent short liquidations (pumped, shorts got rekt)
        2. OI starting to decline (smart money exiting)
        3. CVD turning negative (selling pressure)
        4. Don't short green pumping candles!

        LONG signal conditions:
        1. Recent long liquidations (dumped, longs got rekt)
        2. CVD turning positive (buying pressure emerging)
        3. Don't catch falling knives - wait for first bounce
        """
        short_score = 0.0
        long_score = 0.0
        reasons_short = []
        reasons_long = []

        # ── Short Signal Evaluation ──

        # 1. Short liquidations happened (shorts got rekt = price pumped)
        if state.short_liquidations_usd > self.config.min_liquidation_volume_usd:
            short_score += 0.3
            reasons_short.append(
                f"Short liqs: ${state.short_liquidations_usd:,.0f}"
            )

        # 2. OI declining (smart money leaving after pump)
        if state.oi_change_pct < self.config.oi_decline_threshold_pct:
            short_score += 0.25
            reasons_short.append(f"OI declining: {state.oi_change_pct:.2%}")

        # 3. CVD turning negative (selling starting)
        if state.cvd_trend < -self.config.cvd_divergence_threshold:
            short_score += 0.25
            reasons_short.append(f"CVD bearish: {state.cvd_trend:.2f}")

        # 4. Price recently pumped (confirms highs were taken)
        if state.price_change_1h_pct > 0.02:
            short_score += 0.1
            reasons_short.append(
                f"Recent pump: {state.price_change_1h_pct:.2%}"
            )

        # 5. PENALTY: Don't short active green candles (pump still going)
        if state.cvd_trend > 0.5 and state.price_change_1h_pct > 0.05:
            short_score -= 0.5
            reasons_short.append("SKIP: Active pump in progress")

        # ── Long Signal Evaluation ──

        # 1. Long liquidations happened (longs got rekt = price dumped)
        if state.long_liquidations_usd > self.config.min_liquidation_volume_usd:
            long_score += 0.3
            reasons_long.append(
                f"Long liqs: ${state.long_liquidations_usd:,.0f}"
            )

        # 2. CVD turning positive (buying pressure emerging)
        if state.cvd_trend > self.config.cvd_divergence_threshold:
            long_score += 0.25
            reasons_long.append(f"CVD bullish: {state.cvd_trend:.2f}")

        # 3. Price recently dumped significantly
        if state.price_change_1h_pct < -0.03:
            long_score += 0.15
            reasons_long.append(
                f"Recent dump: {state.price_change_1h_pct:.2%}"
            )

        # 4. First bounce detected (CVD reversal after dump)
        if (
            state.price_change_1h_pct < -0.02
            and state.cvd_trend > 0
        ):
            long_score += 0.2
            reasons_long.append("Bounce detected after dump")

        # 5. PENALTY: Don't catch falling knives
        if state.cvd_trend < -0.5 and state.price_change_1h_pct < -0.05:
            long_score -= 0.5
            reasons_long.append("SKIP: Active dump in progress")

        # ── Volume & Volatility Filters ──
        if state.volume_24h_usd < self.config.min_24h_volume_usd:
            short_score *= 0.3
            long_score *= 0.3

        if state.volatility_pct < self.config.min_volatility_pct:
            short_score *= 0.5
            long_score *= 0.5

        # ── Apply short bias (alts fall most of the time) ──
        short_score *= self.config.short_bias + 0.3  # boost shorts
        long_score *= (1 - self.config.short_bias) + 0.3  # reduce longs

        # ── Final Signal Decision ──
        min_threshold = 0.4

        if short_score > long_score and short_score >= min_threshold:
            state.signal = Signal.SHORT
            state.signal_strength = min(1.0, short_score)
            state.signal_reason = " | ".join(reasons_short)
        elif long_score > short_score and long_score >= min_threshold:
            state.signal = Signal.LONG
            state.signal_strength = min(1.0, long_score)
            state.signal_reason = " | ".join(reasons_long)
        else:
            state.signal = Signal.NONE
            state.signal_strength = 0
            state.signal_reason = "No clear signal"

        if state.signal != Signal.NONE:
            logger.info(
                f"Signal: {state.signal.value.upper()} {state.symbol} | "
                f"Strength: {state.signal_strength:.2f} | "
                f"{state.signal_reason}"
            )

        return state
