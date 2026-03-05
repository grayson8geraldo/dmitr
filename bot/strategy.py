"""
Trading Strategy Engine.
Combines market analysis with risk management to execute trades.
"""

import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger

from bot.exchange_connector import ExchangeConnector
from bot.market_analyzer import MarketAnalyzer, MarketState, Signal
from bot.position_manager import PositionManager, PositionSide
from bot.risk_manager import RiskManager
from bot.trade_journal import TradeJournal
from config import BotConfig


class TradingStrategy:
    """
    Orchestrates the full trading strategy:
    1. Scan coins for liquidation-based signals
    2. Validate signal against risk management
    3. Execute trades with proper sizing and SL/TP
    4. Manage open positions (averaging, hedging)
    """

    def __init__(
        self,
        config: BotConfig,
        exchange: ExchangeConnector,
        analyzer: MarketAnalyzer,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        journal: Optional['TradeJournal'] = None,
    ):
        self.config = config
        self.exchange = exchange
        self.analyzer = analyzer
        self.risk_mgr = risk_manager
        self.pos_mgr = position_manager
        self.journal = journal
        # Track signal info per symbol for journal
        self._signal_info: dict[str, dict] = {}

    async def scan_and_trade(self):
        """
        Main strategy loop iteration:
        1. Check existing positions (SL/TP/hedge/average)
        2. Scan for new opportunities
        3. Execute if signal found
        """
        # Step 1: Manage existing positions
        await self._manage_open_positions()

        # Step 2: Check if we can open new trades
        can_trade, reason = self.risk_mgr.can_open_trade()
        if not can_trade:
            logger.debug(f"Cannot open new trade: {reason}")
            return

        # Step 3: Scan coins for signals
        best_signal = await self._find_best_signal()
        if not best_signal or best_signal.signal == Signal.NONE:
            logger.debug("No actionable signals found")
            return

        # Step 4: Execute trade
        await self._execute_trade(best_signal)

    async def _find_best_signal(self) -> Optional[MarketState]:
        """Scan all preferred coins and return the strongest signal."""
        signals: list[MarketState] = []

        # Fetch OHLCV data in batches to avoid rate limits
        coins = [
            s for s in self.config.strategy.preferred_coins
            if not self.pos_mgr.has_position(s)
        ]

        if not coins:
            return None

        results = []
        batch_size = 5
        for i in range(0, len(coins), batch_size):
            batch = coins[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._analyze_coin(s) for s in batch],
                return_exceptions=True,
            )
            results.extend(batch_results)
            if i + batch_size < len(coins):
                await asyncio.sleep(0.5)

        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Analysis error: {result}")
                continue
            if result and result.signal != Signal.NONE:
                signals.append(result)

        if not signals:
            return None

        # Sort by signal strength, pick the best
        signals.sort(key=lambda s: s.signal_strength, reverse=True)

        best = signals[0]
        logger.info(
            f"Best signal: {best.signal.value.upper()} {best.symbol} | "
            f"Strength: {best.signal_strength:.2f} | "
            f"{best.signal_reason}"
        )
        return best

    async def _analyze_coin(self, symbol: str) -> Optional[MarketState]:
        """Analyze a single coin."""
        try:
            ohlcv_df = await self.exchange.fetch_ohlcv(
                symbol,
                self.config.strategy.analysis_timeframe,
                limit=300,
            )
            if ohlcv_df.empty:
                return None

            # Funding rate is free from exchange public API
            funding_rate = await self.exchange.get_funding_rate(symbol)

            state = await self.analyzer.analyze_symbol(
                symbol, ohlcv_df, funding_rate=funding_rate
            )
            return state

        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None

    async def _execute_trade(self, signal: MarketState):
        """Execute a trade based on the signal."""
        symbol = signal.symbol
        side_str = signal.signal.value

        # Calculate risk parameters
        trade_risk = self.risk_mgr.calculate_trade_risk(side_str)

        # Get current price
        ticker = await self.exchange.get_ticker(symbol)
        price = ticker["price"]
        if price <= 0:
            logger.error(f"Invalid price for {symbol}")
            return

        # Setup trading mode (leverage, margin, hedge)
        await self.exchange.setup_trading_mode(symbol)

        # Calculate SL/TP prices
        if signal.signal == Signal.LONG:
            sl_price = price * (1 - trade_risk.stop_loss_pct)
            tp_price = price * (1 + trade_risk.take_profit_pct)
        else:
            sl_price = price * (1 + trade_risk.stop_loss_pct)
            tp_price = price * (1 - trade_risk.take_profit_pct)

        # Calculate amount in base currency
        amount = await self.exchange.calculate_amount(
            symbol, trade_risk.position_size_usd, price
        )
        if amount <= 0:
            logger.warning(f"Cannot calculate valid amount for {symbol}")
            return

        # Place order
        order = None
        if signal.signal == Signal.LONG:
            order = await self.exchange.open_long(
                symbol, amount, sl_price, tp_price
            )
        elif signal.signal == Signal.SHORT:
            order = await self.exchange.open_short(
                symbol, amount, sl_price, tp_price
            )

        if order:
            # Track position locally
            pos_side = (
                PositionSide.LONG
                if signal.signal == Signal.LONG
                else PositionSide.SHORT
            )
            self.pos_mgr.open_position(
                symbol=symbol,
                side=pos_side,
                entry_price=price,
                amount=amount,
                position_size_usd=trade_risk.position_size_usd,
                leverage=self.config.exchange.default_leverage,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )
            self.risk_mgr.on_position_opened()

            # Store signal info for journal
            self._signal_info[symbol] = {
                "reason_open": signal.signal_reason or "",
                "signal_strength": signal.signal_strength,
                "entry_time": datetime.now().isoformat(),
                "balance_before": self.risk_mgr.current_balance,
            }

            logger.info(
                f"Trade executed | {side_str.upper()} {symbol} @ {price} | "
                f"Size: ${trade_risk.position_size_usd} | "
                f"SL: {sl_price:.6f} | TP: {tp_price:.6f}"
            )

    async def _manage_open_positions(self):
        """
        Check all open positions for:
        - Take profit hit (close with profit)
        - Stop loss hit (close with loss)
        - Averaging opportunity
        - Hedging necessity
        """
        symbols = list(self.pos_mgr.positions.keys())

        for symbol in symbols:
            position = self.pos_mgr.get_position(symbol)
            if not position:
                continue

            ticker = await self.exchange.get_ticker(symbol)
            current_price = ticker["price"]
            if current_price <= 0:
                continue

            pnl_pct = position.calculate_pnl_pct(current_price)

            # Check take profit
            if self.pos_mgr.check_take_profit(symbol, current_price):
                await self._close_position_on_exchange(
                    symbol, position, current_price, "TP"
                )
                continue

            # Check stop loss
            if self.pos_mgr.check_stop_loss(symbol, current_price):
                await self._close_position_on_exchange(
                    symbol, position, current_price, "SL"
                )
                continue

            # Check if should hedge
            funding_rate = await self.exchange.get_funding_rate(symbol)
            if self.risk_mgr.should_hedge(pnl_pct, funding_rate):
                if not position.hedge_position:
                    await self._hedge_position(symbol, position, current_price)
                continue

            # Check if should average
            if (
                pnl_pct < 0
                and self.risk_mgr.can_average_position(pnl_pct)
                and position.average_count < self.config.risk.max_averages
            ):
                await self._average_position(symbol, position, current_price)

    async def _close_position_on_exchange(
        self, symbol: str, position, current_price: float, reason: str
    ):
        """Close a position on the exchange."""
        side = position.side.value
        order = await self.exchange.close_position(
            symbol, side, position.amount
        )
        if order:
            balance_before = self.risk_mgr.current_balance
            pnl = self.pos_mgr.close_position(symbol, current_price)
            self.risk_mgr.record_trade_result(pnl, symbol)
            self.risk_mgr.on_position_closed()

            # Log to journal
            if self.journal:
                sig = self._signal_info.pop(symbol, {})
                self.journal.record_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    amount=position.amount,
                    position_size_usd=position.position_size_usd,
                    pnl_usd=pnl,
                    reason_open=sig.get("reason_open", ""),
                    reason_close=reason,
                    signal_strength=sig.get("signal_strength", 0),
                    entry_time=sig.get("entry_time", ""),
                    balance_before=balance_before,
                    balance_after=self.risk_mgr.current_balance,
                )

            logger.info(
                f"{reason} hit | Closed {side} {symbol} @ {current_price} | "
                f"PnL: ${pnl:.2f}"
            )

    async def _hedge_position(self, symbol, position, current_price):
        """Execute hedging on exchange."""
        logger.info(
            f"Hedging {position.side.value} {symbol} @ {current_price}"
        )

        # Open opposite position with same amount
        if position.side == PositionSide.SHORT:
            order = await self.exchange.open_long(
                symbol, position.amount, 0, 0
            )
        else:
            order = await self.exchange.open_short(
                symbol, position.amount, 0, 0
            )

        if order:
            self.pos_mgr.apply_hedge(symbol, current_price)

    async def _average_position(self, symbol, position, current_price):
        """Execute averaging on exchange."""
        logger.info(
            f"Averaging {position.side.value} {symbol} @ {current_price} | "
            f"Same amount: {position.amount}"
        )

        # Open same-side order with same coin amount
        if position.side == PositionSide.LONG:
            order = await self.exchange.open_long(
                symbol, position.amount,
                position.stop_loss_price, position.take_profit_price
            )
        else:
            order = await self.exchange.open_short(
                symbol, position.amount,
                position.stop_loss_price, position.take_profit_price
            )

        if order:
            self.pos_mgr.apply_averaging(
                symbol, current_price, position.amount
            )
