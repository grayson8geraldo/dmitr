"""
Risk Management Module.
Controls position sizing, daily loss limits, and risk-reward enforcement.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from loguru import logger

from config import RiskConfig


@dataclass
class TradeRisk:
    """Risk parameters calculated for a specific trade."""
    position_size_usd: float    # Total position size in USD
    margin_required: float      # Actual margin used from deposit
    stop_loss_pct: float        # SL distance in %
    take_profit_pct: float      # TP distance in %
    max_loss_usd: float         # Max loss in USD
    expected_profit_usd: float  # Expected profit in USD
    risk_reward: float          # Actual RR ratio


class RiskManager:
    """
    Enforces all risk management rules:
    - Position sizing: 5-10% of deposit
    - Risk per trade: 0.5% of deposit
    - Risk-Reward: 1:3
    - Max daily loss: 3%
    - Max open positions
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.current_date: date = date.today()
        self.open_positions_count: int = 0
        self.current_balance: float = config.initial_deposit
        self.trade_history: list = []

    def reset_daily_stats(self):
        """Reset daily tracking at the start of a new day."""
        today = date.today()
        if today != self.current_date:
            logger.info(
                f"New day: resetting daily stats. "
                f"Previous day PnL: ${self.daily_pnl:.2f}"
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.current_date = today

    def update_balance(self, new_balance: float):
        """Update current balance from exchange."""
        self.current_balance = new_balance

    def can_open_trade(self) -> tuple[bool, str]:
        """Check if a new trade is allowed under risk rules."""
        self.reset_daily_stats()

        # Check daily loss limit
        max_daily_loss = self.current_balance * self.config.max_daily_loss_pct
        if self.daily_pnl <= -max_daily_loss:
            return False, (
                f"Daily loss limit reached: ${self.daily_pnl:.2f} "
                f"(max: -${max_daily_loss:.2f})"
            )

        # Check max open positions
        if self.open_positions_count >= self.config.max_open_positions:
            return False, (
                f"Max open positions reached: {self.open_positions_count}"
                f"/{self.config.max_open_positions}"
            )

        # Check minimum balance
        min_position = self.current_balance * self.config.min_position_pct
        if min_position < 1.0:
            return False, f"Balance too low for trading: ${self.current_balance:.2f}"

        return True, "OK"

    def calculate_trade_risk(self, side: str) -> TradeRisk:
        """
        Calculate position size and risk parameters for a new trade.

        Uses fixed USD position sizing (not percentage slider!).
        At 10x leverage, $50 position uses only $5 of margin.
        """
        # Position size: 5-10% of current balance
        # Use adaptive sizing: smaller when losing, larger when winning
        if self.daily_pnl >= 0:
            position_pct = self.config.max_position_pct  # 10%
        else:
            position_pct = self.config.min_position_pct  # 5%

        position_size_usd = self.current_balance * position_pct
        margin_required = position_size_usd / self.config.leverage

        # Use config SL/TP directly (1.67% / 5% = 1:3 RR)
        stop_loss_pct = self.config.stop_loss_pct
        take_profit_pct = self.config.take_profit_pct

        # Verify actual risk doesn't exceed max risk per trade
        max_loss_usd = self.current_balance * self.config.max_risk_per_trade_pct
        actual_loss_usd = position_size_usd * stop_loss_pct

        # If loss would exceed risk budget, reduce position size
        if actual_loss_usd > max_loss_usd:
            position_size_usd = max_loss_usd / stop_loss_pct
            margin_required = position_size_usd / self.config.leverage
            actual_loss_usd = max_loss_usd

        expected_profit_usd = position_size_usd * take_profit_pct
        actual_rr = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0

        trade_risk = TradeRisk(
            position_size_usd=round(position_size_usd, 2),
            margin_required=round(margin_required, 2),
            stop_loss_pct=round(stop_loss_pct, 4),
            take_profit_pct=round(take_profit_pct, 4),
            max_loss_usd=round(max_loss_usd, 2),
            expected_profit_usd=round(expected_profit_usd, 2),
            risk_reward=round(actual_rr, 2),
        )

        logger.info(
            f"Trade risk calculated | Side: {side} | "
            f"Size: ${trade_risk.position_size_usd} | "
            f"Margin: ${trade_risk.margin_required} | "
            f"SL: {trade_risk.stop_loss_pct:.2%} | "
            f"TP: {trade_risk.take_profit_pct:.2%} | "
            f"RR: 1:{trade_risk.risk_reward}"
        )

        return trade_risk

    def record_trade_result(self, pnl: float, symbol: str):
        """Record a closed trade's PnL."""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.current_balance += pnl

        self.trade_history.append({
            "symbol": symbol,
            "pnl": pnl,
            "balance_after": self.current_balance,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(
            f"Trade closed | {symbol} | PnL: ${pnl:.2f} | "
            f"Daily PnL: ${self.daily_pnl:.2f} | "
            f"Balance: ${self.current_balance:.2f}"
        )

    def on_position_opened(self):
        """Track position count increase."""
        self.open_positions_count += 1

    def on_position_closed(self):
        """Track position count decrease."""
        self.open_positions_count = max(0, self.open_positions_count - 1)

    def should_stop_trading(self) -> bool:
        """Check if bot should stop trading for the day."""
        max_daily_loss = self.current_balance * self.config.max_daily_loss_pct
        if self.daily_pnl <= -max_daily_loss:
            logger.warning(
                f"STOP TRADING: Daily loss limit hit "
                f"(${self.daily_pnl:.2f} / -${max_daily_loss:.2f})"
            )
            return True
        return False

    def can_average_position(self, current_loss_pct: float) -> bool:
        """
        Check if averaging is allowed.
        Only average at significant levels (10-15% price movement).
        """
        return abs(current_loss_pct) >= self.config.averaging_distance_pct

    def should_hedge(
        self, current_loss_pct: float, funding_rate: float
    ) -> bool:
        """
        Check if position should be hedged.
        Hedge when loss exceeds threshold AND funding is strongly negative.
        """
        loss_exceeded = abs(current_loss_pct) >= self.config.hedge_loss_threshold_pct
        funding_dangerous = funding_rate <= self.config.funding_rate_hedge_threshold
        return loss_exceeded and funding_dangerous

    def get_progress_report(self) -> dict:
        """Return progress towards the target."""
        progress_pct = (
            (self.current_balance - self.config.initial_deposit)
            / (self.config.target_deposit - self.config.initial_deposit)
            * 100
        )
        return {
            "current_balance": self.current_balance,
            "initial_deposit": self.config.initial_deposit,
            "target": self.config.target_deposit,
            "progress_pct": round(max(0, progress_pct), 2),
            "total_pnl": round(
                self.current_balance - self.config.initial_deposit, 2
            ),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "total_trades": len(self.trade_history),
        }
