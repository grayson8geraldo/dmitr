"""
Market Analysis Module.
Uses free OHLCV-based indicators: RSI, Bollinger Bands, volume spikes,
wick analysis (liquidation proxy), EMA trend, and CVD.
No paid APIs required — all data from exchange public endpoints.
"""

from dataclasses import dataclass
from enum import Enum

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

    # Technical indicators
    rsi: float                   # RSI(14)
    bb_position: float           # Price position in Bollinger Bands (0=lower, 1=upper)
    ema_trend: float             # EMA9/EMA21 ratio (>1 = bullish)
    volume_spike: float          # Current volume / avg volume ratio
    wick_ratio_upper: float      # Upper wick / candle range (liquidation proxy)
    wick_ratio_lower: float      # Lower wick / candle range (liquidation proxy)

    # CVD (Cumulative Volume Delta)
    cvd_trend: float             # Positive = buying pressure, Negative = selling

    # Funding rate (free from exchange)
    funding_rate: float

    # Derived signal
    signal: Signal = Signal.NONE
    signal_strength: float = 0.0
    signal_reason: str = ""


class MarketAnalyzer:
    """
    Analyzes market conditions using free indicators.

    Strategy:
    - SHORT: RSI overbought + upper wick spikes (stop hunts) +
             volume spike + CVD divergence + BB upper band rejection
    - LONG: RSI oversold + lower wick spikes (stop hunts) +
            volume spike + CVD reversal + BB lower band bounce
    """

    def __init__(self, strategy_config):
        self.config = strategy_config

    async def close(self):
        """No resources to clean up."""
        pass

    # ─── Technical Indicators ─────────────────────────────────────────

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _calc_bollinger(close: np.ndarray, period: int = 20, num_std: float = 2.0):
        """Calculate Bollinger Bands position (0 = at lower, 1 = at upper)."""
        if len(close) < period:
            return 0.5
        window = close[-period:]
        sma = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.5
        upper = sma + num_std * std
        lower = sma - num_std * std
        band_width = upper - lower
        if band_width == 0:
            return 0.5
        position = (close[-1] - lower) / band_width
        return float(np.clip(position, 0, 1))

    @staticmethod
    def _calc_ema(data: np.ndarray, period: int) -> float:
        """Calculate EMA using numpy."""
        if len(data) < period:
            return float(data[-1])
        multiplier = 2 / (period + 1)
        ema = data[0]
        for val in data[1:]:
            ema = (val - ema) * multiplier + ema
        return float(ema)

    @staticmethod
    def _calc_wick_ratios(
        open_p: np.ndarray, high: np.ndarray,
        low: np.ndarray, close: np.ndarray, lookback: int = 5
    ) -> tuple[float, float]:
        """
        Calculate average wick ratios over recent candles.
        Large wicks = stop hunts / liquidation cascades.
        Returns (upper_wick_ratio, lower_wick_ratio).
        """
        n = min(lookback, len(open_p))
        upper_ratios = []
        lower_ratios = []

        for i in range(-n, 0):
            candle_range = high[i] - low[i]
            if candle_range <= 0:
                continue
            body_top = max(open_p[i], close[i])
            body_bottom = min(open_p[i], close[i])
            upper_wick = high[i] - body_top
            lower_wick = body_bottom - low[i]
            upper_ratios.append(upper_wick / candle_range)
            lower_ratios.append(lower_wick / candle_range)

        avg_upper = float(np.mean(upper_ratios)) if upper_ratios else 0
        avg_lower = float(np.mean(lower_ratios)) if lower_ratios else 0
        return avg_upper, avg_lower

    def analyze_ohlcv(self, df: pd.DataFrame) -> dict:
        """
        Full OHLCV analysis with all free indicators.
        """
        if df.empty or len(df) < 20:
            return {
                "volatility": 0, "price_change_1h": 0,
                "price_change_24h": 0, "cvd_trend": 0,
                "volume_24h": 0, "current_price": 0,
                "rsi": 50, "bb_position": 0.5,
                "ema_trend": 1.0, "volume_spike": 1.0,
                "wick_ratio_upper": 0, "wick_ratio_lower": 0,
            }

        close = df["close"].values
        volume = df["volume"].values
        open_prices = df["open"].values
        high = df["high"].values
        low = df["low"].values

        # ── Volatility (ATR-based) ──
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        volatility = atr / close[-1] if close[-1] > 0 else 0

        # ── Price changes ──
        price_change_1h = 0
        if len(close) >= 12:
            price_change_1h = (close[-1] - close[-12]) / close[-12]

        price_change_24h = 0
        if len(close) >= 288:
            price_change_24h = (close[-1] - close[-288]) / close[-288]
        elif len(close) >= 2:
            price_change_24h = (close[-1] - close[0]) / close[0]

        # ── RSI ──
        rsi = self._calc_rsi(close)

        # ── Bollinger Bands position ──
        bb_position = self._calc_bollinger(close)

        # ── EMA trend (EMA9 / EMA21) ──
        ema9 = self._calc_ema(close, 9)
        ema21 = self._calc_ema(close, 21)
        ema_trend = ema9 / ema21 if ema21 > 0 else 1.0

        # ── Volume spike (current vs average) ──
        avg_vol = np.mean(volume[-50:]) if len(volume) >= 50 else np.mean(volume)
        recent_vol = np.mean(volume[-3:]) if len(volume) >= 3 else volume[-1]
        volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # ── Wick ratios (liquidation proxy) ──
        wick_upper, wick_lower = self._calc_wick_ratios(
            open_prices, high, low, close
        )

        # ── CVD approximation ──
        deltas = []
        for i in range(len(close)):
            if close[i] >= open_prices[i]:
                deltas.append(volume[i])
            else:
                deltas.append(-volume[i])

        cvd = np.cumsum(deltas)
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
            "rsi": float(rsi),
            "bb_position": float(bb_position),
            "ema_trend": float(ema_trend),
            "volume_spike": float(volume_spike),
            "wick_ratio_upper": float(wick_upper),
            "wick_ratio_lower": float(wick_lower),
        }

    # ─── Signal Generation ─────────────────────────────────────────────

    async def analyze_symbol(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        funding_rate: float = 0.0,
    ) -> MarketState:
        """
        Full analysis of a symbol. Returns MarketState with signal.
        funding_rate should be passed from exchange connector (free).
        """
        analysis = self.analyze_ohlcv(ohlcv_df)

        state = MarketState(
            symbol=symbol,
            price=analysis["current_price"],
            price_change_1h_pct=analysis["price_change_1h"],
            price_change_24h_pct=analysis["price_change_24h"],
            volume_24h_usd=analysis["volume_24h"],
            volatility_pct=analysis["volatility"],
            rsi=analysis["rsi"],
            bb_position=analysis["bb_position"],
            ema_trend=analysis["ema_trend"],
            volume_spike=analysis["volume_spike"],
            wick_ratio_upper=analysis["wick_ratio_upper"],
            wick_ratio_lower=analysis["wick_ratio_lower"],
            cvd_trend=analysis["cvd_trend"],
            funding_rate=funding_rate,
        )

        state = self._generate_signal(state)
        return state

    def _generate_signal(self, state: MarketState) -> MarketState:
        """
        Generate trading signal using free indicators.

        SHORT signal (after pump exhaustion):
        1. RSI overbought (>70) — momentum exhaustion
        2. Large upper wicks — stop hunts / short liquidations happened
        3. Volume spike — cascade activity
        4. CVD turning negative — sellers stepping in
        5. BB near upper band — mean reversion expected
        6. EMA bearish cross starting

        LONG signal (after dump exhaustion):
        1. RSI oversold (<30) — momentum exhaustion
        2. Large lower wicks — long liquidation cascades happened
        3. Volume spike — cascade activity
        4. CVD turning positive — buyers stepping in
        5. BB near lower band — mean reversion expected
        6. EMA bullish cross starting
        """
        short_score = 0.0
        long_score = 0.0
        reasons_short = []
        reasons_long = []

        # ── SHORT Signal ──

        # 1. RSI overbought
        if state.rsi > 70:
            score = min(0.3, (state.rsi - 70) / 30 * 0.3)
            short_score += score
            reasons_short.append(f"RSI overbought: {state.rsi:.0f}")
        elif state.rsi > 60:
            short_score += 0.1
            reasons_short.append(f"RSI elevated: {state.rsi:.0f}")

        # 2. Upper wick spikes (liquidation proxy)
        if state.wick_ratio_upper > 0.4:
            short_score += 0.25
            reasons_short.append(
                f"Upper wicks (stop hunts): {state.wick_ratio_upper:.0%}"
            )
        elif state.wick_ratio_upper > 0.25:
            short_score += 0.1

        # 3. Volume spike
        if state.volume_spike > 2.0:
            short_score += 0.15
            reasons_short.append(f"Volume spike: {state.volume_spike:.1f}x")

        # 4. CVD turning negative
        if state.cvd_trend < -self.config.cvd_divergence_threshold:
            short_score += 0.2
            reasons_short.append(f"CVD bearish: {state.cvd_trend:.2f}")

        # 5. BB near upper band
        if state.bb_position > 0.85:
            short_score += 0.15
            reasons_short.append(f"BB upper rejection: {state.bb_position:.0%}")

        # 6. EMA bearish (9 crossing below 21)
        if state.ema_trend < 0.999:
            short_score += 0.1
            reasons_short.append(f"EMA bearish: {state.ema_trend:.4f}")

        # 7. Price recently pumped
        if state.price_change_1h_pct > 0.02:
            short_score += 0.1
            reasons_short.append(
                f"Recent pump: {state.price_change_1h_pct:.2%}"
            )

        # PENALTY: Don't short during active pump with strong buying
        if state.cvd_trend > 0.5 and state.price_change_1h_pct > 0.05:
            short_score -= 0.5
            reasons_short.append("SKIP: Active pump in progress")

        # PENALTY: RSI not even elevated — no exhaustion signal
        if state.rsi < 45:
            short_score *= 0.5

        # ── LONG Signal ──

        # 1. RSI oversold
        if state.rsi < 30:
            score = min(0.3, (30 - state.rsi) / 30 * 0.3)
            long_score += score
            reasons_long.append(f"RSI oversold: {state.rsi:.0f}")
        elif state.rsi < 40:
            long_score += 0.1
            reasons_long.append(f"RSI low: {state.rsi:.0f}")

        # 2. Lower wick spikes (liquidation proxy)
        if state.wick_ratio_lower > 0.4:
            long_score += 0.25
            reasons_long.append(
                f"Lower wicks (stop hunts): {state.wick_ratio_lower:.0%}"
            )
        elif state.wick_ratio_lower > 0.25:
            long_score += 0.1

        # 3. Volume spike
        if state.volume_spike > 2.0:
            long_score += 0.15
            reasons_long.append(f"Volume spike: {state.volume_spike:.1f}x")

        # 4. CVD turning positive
        if state.cvd_trend > self.config.cvd_divergence_threshold:
            long_score += 0.2
            reasons_long.append(f"CVD bullish: {state.cvd_trend:.2f}")

        # 5. BB near lower band
        if state.bb_position < 0.15:
            long_score += 0.15
            reasons_long.append(f"BB lower bounce: {state.bb_position:.0%}")

        # 6. EMA bullish (9 crossing above 21)
        if state.ema_trend > 1.001:
            long_score += 0.1
            reasons_long.append(f"EMA bullish: {state.ema_trend:.4f}")

        # 7. Price recently dumped
        if state.price_change_1h_pct < -0.02:
            long_score += 0.1
            reasons_long.append(
                f"Recent dump: {state.price_change_1h_pct:.2%}"
            )

        # 8. Bounce detected (CVD reversal after dump)
        if state.price_change_1h_pct < -0.02 and state.cvd_trend > 0:
            long_score += 0.15
            reasons_long.append("Bounce detected after dump")

        # PENALTY: Don't catch falling knives
        if state.cvd_trend < -0.5 and state.price_change_1h_pct < -0.05:
            long_score -= 0.5
            reasons_long.append("SKIP: Active dump in progress")

        # PENALTY: RSI not even low — no exhaustion signal
        if state.rsi > 55:
            long_score *= 0.5

        # ── Volume & Volatility Filters ──
        if state.volume_24h_usd < self.config.min_24h_volume_usd:
            short_score *= 0.6
            long_score *= 0.6

        if state.volatility_pct < self.config.min_volatility_pct:
            short_score *= 0.7
            long_score *= 0.7

        # ── Short bias (alts fall most of the time) ──
        short_score *= self.config.short_bias + 0.3
        long_score *= (1 - self.config.short_bias) + 0.3

        # ── Final Signal Decision ──
        min_threshold = 0.3

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
