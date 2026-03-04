"""
Position Management Module.
Handles open positions, averaging, hedging, TP/SL management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from loguru import logger

from config import RiskConfig


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    OPEN = "open"
    HEDGED = "hedged"
    CLOSED = "closed"


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: PositionSide
    entry_price: float
    amount: float              # Amount in base currency (coins)
    position_size_usd: float   # Total position notional
    margin_used: float         # Actual margin from deposit
    leverage: int

    stop_loss_price: float
    take_profit_price: float

    status: PositionStatus = PositionStatus.OPEN
    average_count: int = 0     # How many times averaged
    hedge_position: Optional["Position"] = None

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pnl: float = 0.0

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized PnL as % of position."""
        # This needs current price to be meaningful
        return 0.0

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL given current price."""
        if self.side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) / self.entry_price
        else:
            pnl = (self.entry_price - current_price) / self.entry_price
        return pnl * self.position_size_usd

    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized PnL as % of entry."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


class PositionManager:
    """
    Manages all open positions with averaging and hedging logic.

    Averaging rules:
    - Wait for 10-15% price movement against position
    - Average with same amount of coins
    - Only average once per position

    Hedging rules:
    - Hedge when loss exceeds threshold AND funding is dangerous
    - Open opposite position with same coin amount
    - Close hedge on resistance/support, use profits to average main position
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: dict[str, Position] = {}  # symbol -> Position

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        amount: float,
        position_size_usd: float,
        leverage: int,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Position:
        """Register a new position."""
        margin = position_size_usd / leverage

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            position_size_usd=position_size_usd,
            margin_used=margin,
            leverage=leverage,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        self.positions[symbol] = position

        logger.info(
            f"Position opened | {side.value.upper()} {symbol} | "
            f"Entry: {entry_price} | Amount: {amount} | "
            f"Size: ${position_size_usd} | Margin: ${margin:.2f} | "
            f"SL: {stop_loss_price} | TP: {take_profit_price}"
        )

        return position

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized PnL."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return 0.0

        position = self.positions[symbol]
        pnl = position.calculate_pnl(exit_price)
        position.pnl = pnl
        position.status = PositionStatus.CLOSED

        # If hedged, close hedge too
        if position.hedge_position:
            hedge_pnl = position.hedge_position.calculate_pnl(exit_price)
            pnl += hedge_pnl
            logger.info(f"Hedge closed for {symbol} | Hedge PnL: ${hedge_pnl:.2f}")

        del self.positions[symbol]

        logger.info(
            f"Position closed | {symbol} | "
            f"Exit: {exit_price} | PnL: ${pnl:.2f}"
        )

        return pnl

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if current price hit stop loss."""
        position = self.positions.get(symbol)
        if not position:
            return False

        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss_price
        else:
            return current_price >= position.stop_loss_price

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if current price hit take profit."""
        position = self.positions.get(symbol)
        if not position:
            return False

        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit_price
        else:
            return current_price <= position.take_profit_price

    def calculate_average_entry(
        self, symbol: str, new_price: float, new_amount: float
    ) -> dict:
        """
        Calculate new average entry price after averaging.
        Must average with same coin amount as current position.

        Returns dict with new entry price and total amount.
        """
        position = self.positions.get(symbol)
        if not position:
            return {}

        total_amount = position.amount + new_amount
        new_entry = (
            (position.entry_price * position.amount)
            + (new_price * new_amount)
        ) / total_amount

        return {
            "new_entry_price": new_entry,
            "total_amount": total_amount,
            "new_position_size_usd": total_amount * new_entry,
        }

    def apply_averaging(
        self, symbol: str, avg_price: float, avg_amount: float
    ) -> Optional[Position]:
        """
        Apply averaging to an existing position.
        Strictly same coin amount, at a significant level.
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        if position.average_count >= self.config.max_averages:
            logger.warning(
                f"Cannot average {symbol}: max averages reached "
                f"({position.average_count}/{self.config.max_averages})"
            )
            return None

        calc = self.calculate_average_entry(symbol, avg_price, avg_amount)
        if not calc:
            return None

        old_entry = position.entry_price
        position.entry_price = calc["new_entry_price"]
        position.amount = calc["total_amount"]
        position.position_size_usd = calc["new_position_size_usd"]
        position.margin_used = position.position_size_usd / position.leverage
        position.average_count += 1

        # Update SL/TP relative to new entry
        sl_distance = abs(old_entry - position.stop_loss_price) / old_entry
        tp_distance = abs(old_entry - position.take_profit_price) / old_entry

        if position.side == PositionSide.LONG:
            position.stop_loss_price = position.entry_price * (1 - sl_distance)
            position.take_profit_price = position.entry_price * (1 + tp_distance)
        else:
            position.stop_loss_price = position.entry_price * (1 + sl_distance)
            position.take_profit_price = position.entry_price * (1 - tp_distance)

        logger.info(
            f"Position averaged | {symbol} | "
            f"Old entry: {old_entry} -> New entry: {position.entry_price:.6f} | "
            f"Total amount: {position.amount} | "
            f"New SL: {position.stop_loss_price:.6f} | "
            f"New TP: {position.take_profit_price:.6f}"
        )

        return position

    def apply_hedge(self, symbol: str, hedge_price: float) -> Optional[Position]:
        """
        Open a hedge position (opposite side, same coin amount).
        Freezes PnL and neutralizes funding rate impact.
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        if position.status == PositionStatus.HEDGED:
            logger.warning(f"Position {symbol} already hedged")
            return None

        hedge_side = (
            PositionSide.LONG
            if position.side == PositionSide.SHORT
            else PositionSide.SHORT
        )

        hedge = Position(
            symbol=symbol,
            side=hedge_side,
            entry_price=hedge_price,
            amount=position.amount,  # Same coin amount
            position_size_usd=position.amount * hedge_price,
            margin_used=(position.amount * hedge_price) / position.leverage,
            leverage=position.leverage,
            stop_loss_price=0,  # Managed manually
            take_profit_price=0,
        )

        position.hedge_position = hedge
        position.status = PositionStatus.HEDGED

        logger.info(
            f"Hedge opened | {hedge_side.value.upper()} {symbol} | "
            f"Price: {hedge_price} | Amount: {position.amount} | "
            f"Main position PnL now frozen"
        )

        return hedge

    def close_hedge(self, symbol: str, exit_price: float) -> float:
        """
        Close the hedge position and return its PnL.
        Use hedge profits to average the main position.
        """
        position = self.positions.get(symbol)
        if not position or not position.hedge_position:
            return 0.0

        hedge_pnl = position.hedge_position.calculate_pnl(exit_price)
        position.hedge_position = None
        position.status = PositionStatus.OPEN

        logger.info(
            f"Hedge closed | {symbol} | "
            f"Hedge PnL: ${hedge_pnl:.2f} | "
            f"Main position unfrozen"
        )

        return hedge_pnl

    def get_all_positions_summary(self, current_prices: dict) -> list[dict]:
        """Get summary of all open positions."""
        summary = []
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            pnl = pos.calculate_pnl(price)
            pnl_pct = pos.calculate_pnl_pct(price)

            summary.append({
                "symbol": symbol,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "current_price": price,
                "amount": pos.amount,
                "size_usd": pos.position_size_usd,
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct * 100, 2),
                "status": pos.status.value,
                "hedged": pos.hedge_position is not None,
                "averaged": pos.average_count,
            })

        return summary
