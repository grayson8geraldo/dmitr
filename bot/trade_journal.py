"""
Trade Journal Module.
Logs every trade to JSON file, tracks running statistics,
and provides real-time performance reports.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from loguru import logger


@dataclass
class TradeRecord:
    """Single trade record."""
    id: int
    symbol: str
    side: str                    # long / short
    entry_price: float
    exit_price: float
    amount: float
    position_size_usd: float
    pnl_usd: float
    pnl_pct: float
    reason_open: str             # signal reason
    reason_close: str            # TP / SL / manual
    signal_strength: float
    entry_time: str
    exit_time: str
    duration_seconds: float
    balance_before: float
    balance_after: float


class TradeJournal:
    """
    Persistent trade journal with real-time statistics.
    Saves all trades to a JSON file and computes running stats.
    """

    def __init__(self, journal_path: str = "paper_trades.json"):
        self.journal_path = journal_path
        self.trades: list[TradeRecord] = []
        self._next_id = 1

        # Running stats
        self._peak_balance = 0.0
        self._max_drawdown = 0.0
        self._max_drawdown_pct = 0.0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._best_streak = 0
        self._worst_streak = 0

        # Load existing journal if present
        self._load()

    def _load(self):
        """Load existing journal from file."""
        if not os.path.exists(self.journal_path):
            return
        try:
            with open(self.journal_path, "r") as f:
                data = json.load(f)
            self.trades = [TradeRecord(**t) for t in data.get("trades", [])]
            if self.trades:
                self._next_id = max(t.id for t in self.trades) + 1
                # Restore streaks
                for t in self.trades:
                    self._update_streaks(t.pnl_usd)
                # Restore peak balance
                if self.trades:
                    self._peak_balance = max(t.balance_after for t in self.trades)
            logger.info(
                f"[JOURNAL] Loaded {len(self.trades)} trades from {self.journal_path}"
            )
        except Exception as e:
            logger.warning(f"[JOURNAL] Could not load journal: {e}")

    def _save(self):
        """Save journal to file."""
        try:
            data = {
                "trades": [asdict(t) for t in self.trades],
                "stats": self.get_stats(),
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.journal_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[JOURNAL] Failed to save: {e}")

    def _update_streaks(self, pnl: float):
        """Update win/loss streak counters."""
        if pnl > 0:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
            self._best_streak = max(self._best_streak, self._consecutive_wins)
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
            self._worst_streak = max(self._worst_streak, self._consecutive_losses)

    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        amount: float,
        position_size_usd: float,
        pnl_usd: float,
        reason_open: str,
        reason_close: str,
        signal_strength: float,
        entry_time: str,
        balance_before: float,
        balance_after: float,
    ):
        """Record a completed trade."""
        now = datetime.now()
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            duration = (now - entry_dt).total_seconds()
        except (ValueError, TypeError):
            duration = 0.0

        pnl_pct = (pnl_usd / position_size_usd * 100) if position_size_usd > 0 else 0.0

        trade = TradeRecord(
            id=self._next_id,
            symbol=symbol,
            side=side,
            entry_price=round(entry_price, 8),
            exit_price=round(exit_price, 8),
            amount=amount,
            position_size_usd=round(position_size_usd, 2),
            pnl_usd=round(pnl_usd, 4),
            pnl_pct=round(pnl_pct, 2),
            reason_open=reason_open,
            reason_close=reason_close,
            signal_strength=round(signal_strength, 3),
            entry_time=entry_time,
            exit_time=now.isoformat(),
            duration_seconds=round(duration, 1),
            balance_before=round(balance_before, 2),
            balance_after=round(balance_after, 2),
        )
        self._next_id += 1
        self.trades.append(trade)

        # Update streaks & drawdown
        self._update_streaks(pnl_usd)
        if balance_after > self._peak_balance:
            self._peak_balance = balance_after
        if self._peak_balance > 0:
            drawdown = self._peak_balance - balance_after
            drawdown_pct = drawdown / self._peak_balance * 100
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
                self._max_drawdown_pct = drawdown_pct

        self._save()

        # Log the trade
        result = "WIN" if pnl_usd > 0 else "LOSS"
        logger.info(
            f"[JOURNAL] #{trade.id} {result} | {side.upper()} {symbol} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_pct:+.1f}%) | "
            f"Balance: ${balance_after:.2f} | "
            f"{reason_close}"
        )

    def get_stats(self, today_only: bool = True) -> dict:
        """Calculate performance statistics. Filters by today if today_only=True."""
        today = datetime.now().strftime("%Y-%m-%d")
        trades = [
            t for t in self.trades
            if not today_only or t.exit_time.startswith(today)
        ]
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "message": "No trades today" if today_only else "No trades yet",
            }

        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        total = len(trades)

        total_pnl = sum(t.pnl_usd for t in trades)
        avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
        win_rate = len(wins) / total * 100

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average duration
        avg_duration = sum(t.duration_seconds for t in trades) / total

        # Per-symbol breakdown
        symbols: dict[str, dict] = {}
        for t in trades:
            if t.symbol not in symbols:
                symbols[t.symbol] = {"trades": 0, "wins": 0, "pnl": 0.0}
            symbols[t.symbol]["trades"] += 1
            if t.pnl_usd > 0:
                symbols[t.symbol]["wins"] += 1
            symbols[t.symbol]["pnl"] += t.pnl_usd

        # Best/worst trade
        best_trade = max(trades, key=lambda t: t.pnl_usd)
        worst_trade = min(trades, key=lambda t: t.pnl_usd)

        # Long vs Short breakdown
        longs = [t for t in trades if t.side == "long"]
        shorts = [t for t in trades if t.side == "short"]
        long_wr = (sum(1 for t in longs if t.pnl_usd > 0) / len(longs) * 100) if longs else 0
        short_wr = (sum(1 for t in shorts if t.pnl_usd > 0) / len(shorts) * 100) if shorts else 0

        return {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": round(best_trade.pnl_usd, 2),
            "worst_trade": round(worst_trade.pnl_usd, 2),
            "max_drawdown": round(self._max_drawdown, 2),
            "max_drawdown_pct": round(self._max_drawdown_pct, 1),
            "best_streak": self._best_streak,
            "worst_streak": self._worst_streak,
            "current_streak": (
                f"+{self._consecutive_wins}W"
                if self._consecutive_wins > 0
                else f"-{self._consecutive_losses}L"
            ),
            "avg_duration_min": round(avg_duration / 60, 1),
            "long_trades": len(longs),
            "long_win_rate": round(long_wr, 1),
            "short_trades": len(shorts),
            "short_win_rate": round(short_wr, 1),
            "per_symbol": {
                sym: {
                    "trades": d["trades"],
                    "win_rate": round(d["wins"] / d["trades"] * 100, 1),
                    "pnl": round(d["pnl"], 2),
                }
                for sym, d in symbols.items()
            },
        }

    def print_report(self):
        """Print formatted performance report to logger."""
        stats = self.get_stats()
        if stats["total_trades"] == 0:
            logger.info("[JOURNAL] No trades recorded yet")
            return

        logger.info("=" * 60)
        logger.info("  TRADE JOURNAL REPORT")
        logger.info("-" * 60)
        logger.info(
            f"  Trades: {stats['total_trades']} "
            f"(W:{stats['wins']} / L:{stats['losses']})"
        )
        logger.info(
            f"  Win rate: {stats['win_rate']}% | "
            f"Streak: {stats['current_streak']}"
        )
        logger.info(
            f"  Total PnL: ${stats['total_pnl']:+.2f} | "
            f"Profit factor: {stats['profit_factor']}"
        )
        logger.info(
            f"  Avg win: ${stats['avg_win']:+.2f} | "
            f"Avg loss: ${stats['avg_loss']:.2f}"
        )
        logger.info(
            f"  Best: ${stats['best_trade']:+.2f} | "
            f"Worst: ${stats['worst_trade']:.2f}"
        )
        logger.info(
            f"  Max drawdown: ${stats['max_drawdown']:.2f} "
            f"({stats['max_drawdown_pct']:.1f}%)"
        )
        logger.info(
            f"  Best streak: {stats['best_streak']}W | "
            f"Worst streak: {stats['worst_streak']}L"
        )
        logger.info(
            f"  Avg duration: {stats['avg_duration_min']:.1f} min"
        )
        logger.info(
            f"  LONG: {stats['long_trades']} trades, "
            f"{stats['long_win_rate']}% WR | "
            f"SHORT: {stats['short_trades']} trades, "
            f"{stats['short_win_rate']}% WR"
        )

        if stats["per_symbol"]:
            logger.info("-" * 60)
            logger.info("  PER SYMBOL:")
            for sym, d in sorted(
                stats["per_symbol"].items(),
                key=lambda x: x[1]["pnl"],
                reverse=True,
            ):
                logger.info(
                    f"    {sym}: {d['trades']} trades | "
                    f"WR: {d['win_rate']}% | "
                    f"PnL: ${d['pnl']:+.2f}"
                )

        logger.info("=" * 60)
