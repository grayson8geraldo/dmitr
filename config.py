"""
Trading bot configuration.
All strategy parameters from the deposit acceleration strategy.
"""

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    """Risk management settings."""
    # Deposit and position sizing
    initial_deposit: float = 100.0
    target_deposit: float = 1000.0
    max_position_pct: float = 0.10       # 10% of deposit per trade
    min_position_pct: float = 0.05       # 5% of deposit per trade
    leverage: int = 10                    # 10x leverage

    # Risk-reward
    risk_reward_ratio: float = 3.0       # 1:3 RR
    max_risk_per_trade_pct: float = 0.005  # 0.5% of deposit risked per trade
    max_profit_per_trade_pct: float = 0.015  # 1.5% of deposit target profit

    # Stop loss / Take profit (price movement %)
    take_profit_pct: float = 0.05        # 5% price move for TP
    stop_loss_pct: float = 0.0167        # ~1.67% price move for SL (1:3 RR)

    # Position limits
    max_open_positions: int = 3
    max_daily_loss_pct: float = 0.03     # 3% max daily loss -> stop trading

    # Averaging rules
    averaging_distance_pct: float = 0.12  # 10-15% price move before averaging
    max_averages: int = 1                 # Max 1 averaging per position

    # Hedging
    hedge_loss_threshold_pct: float = 0.08  # Hedge if position loses 8%+
    funding_rate_hedge_threshold: float = -0.02  # Hedge if funding < -2%


@dataclass
class StrategyConfig:
    """Strategy parameters for quick scalping."""
    # Timeframes
    analysis_timeframe: str = "5m"
    trend_timeframe: str = "1h"

    # CVD (Cumulative Volume Delta)
    cvd_reversal_periods: int = 5         # Periods to confirm CVD reversal
    cvd_divergence_threshold: float = 0.3  # Min divergence strength

    # Coin selection
    preferred_coins: list = field(default_factory=lambda: [
        # Top-tier (highest volume)
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
        # Large caps
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT",
        "DOT/USDT", "POL/USDT", "NEAR/USDT", "SUI/USDT",
        "APT/USDT", "FIL/USDT", "LTC/USDT", "BCH/USDT",
        # Mid caps (volatile, good for scalping)
        "PEPE/USDT", "WIF/USDT", "FLOKI/USDT", "BONK/USDT",
        "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT",
        "SEI/USDT", "JUP/USDT", "STX/USDT", "IMX/USDT",
        "RENDER/USDT", "FET/USDT", "RUNE/USDT", "AAVE/USDT",
        "ENA/USDT", "WLD/USDT", "ORDI/USDT", "PENDLE/USDT",
    ])

    # Filters
    min_24h_volume_usd: float = 50_000_000  # Min daily volume
    min_volatility_pct: float = 0.03         # Min 3% daily volatility

    # Short bias (90-95% of alts fall 90% of time)
    short_bias: float = 0.7   # 70% preference for shorts


@dataclass
class ExchangeConfig:
    """Exchange connection settings."""
    name: str = "bybit"
    margin_mode: str = "cross"       # Cross-margin mandatory
    position_mode: str = "hedge"     # Hedge mode mandatory (both long+short)
    default_leverage: int = 10
    testnet: bool = True             # Start with testnet!
    paper_trading: bool = False      # Paper trading: real data, virtual balance


@dataclass
class BotConfig:
    """Main bot configuration."""
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    # Bot operation
    scan_interval_seconds: int = 30   # Scan market every 30s
    log_level: str = "INFO"
