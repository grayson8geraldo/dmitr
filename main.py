"""
Deposit Acceleration Trading Bot.
Strategy: Liquidation-based entries with strict risk management.
Goal: $100 -> $1000 in 2 weeks.

Usage:
    python main.py              # Run with testnet (default)
    python main.py --live       # Run with real money (be careful!)
    python main.py --status     # Show current progress
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from config import BotConfig
from bot.exchange_connector import ExchangeConnector
from bot.market_analyzer import MarketAnalyzer
from bot.position_manager import PositionManager
from bot.risk_manager import RiskManager
from bot.strategy import TradingStrategy


def setup_logging(level: str = "INFO"):
    """Configure logging with loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        "bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="14 days",
        level="DEBUG",
    )


async def run_bot(config: BotConfig, live: bool = False):
    """Main bot execution loop."""
    load_dotenv()

    api_key = os.getenv("EXCHANGE_API_KEY", "")
    api_secret = os.getenv("EXCHANGE_API_SECRET", "")
    coinglass_key = os.getenv("COINGLASS_API_KEY", "")

    if not api_key or not api_secret:
        logger.error(
            "Missing API credentials. "
            "Copy .env.example to .env and fill in your keys."
        )
        return

    # Override testnet setting
    config.exchange.testnet = not live

    # Initialize components
    exchange = ExchangeConnector(config.exchange, api_key, api_secret)
    analyzer = MarketAnalyzer(config.strategy, coinglass_key)
    risk_manager = RiskManager(config.risk)
    position_manager = PositionManager(config.risk)

    strategy = TradingStrategy(
        config, exchange, analyzer, risk_manager, position_manager
    )

    try:
        await exchange.connect()

        # Get initial balance
        balance = await exchange.get_balance()
        risk_manager.update_balance(balance)

        mode = "LIVE" if live else "TESTNET"
        logger.info("=" * 60)
        logger.info(f"  DEPOSIT ACCELERATION BOT STARTED [{mode}]")
        logger.info(f"  Exchange: {config.exchange.name}")
        logger.info(f"  Balance: ${balance:.2f}")
        logger.info(f"  Target: ${config.risk.target_deposit:.2f}")
        logger.info(f"  Leverage: {config.exchange.default_leverage}x")
        logger.info(f"  Margin: Cross | Mode: Hedge")
        logger.info(f"  Risk per trade: {config.risk.max_risk_per_trade_pct:.1%}")
        logger.info(f"  TP: {config.risk.take_profit_pct:.1%} | "
                     f"SL: {config.risk.stop_loss_pct:.2%} | RR: 1:3")
        logger.info(f"  Coins: {len(config.strategy.preferred_coins)}")
        logger.info("=" * 60)

        if balance < 1:
            logger.error(
                f"Balance too low: ${balance:.2f}. "
                f"Deposit at least ${config.risk.initial_deposit} USDT."
            )
            return

        # Setup trading mode for all coins
        logger.info("Setting up trading mode for coins...")
        for symbol in config.strategy.preferred_coins:
            try:
                await exchange.setup_trading_mode(symbol)
            except Exception as e:
                logger.debug(f"Setup {symbol}: {e}")

        # Main loop
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.debug(f"--- Scan #{iteration} ---")

                # Update balance
                balance = await exchange.get_balance()
                risk_manager.update_balance(balance)

                # Check if target reached
                if balance >= config.risk.target_deposit:
                    logger.info("=" * 60)
                    logger.info(f"  TARGET REACHED! Balance: ${balance:.2f}")
                    logger.info("=" * 60)
                    break

                # Check daily loss limit
                if risk_manager.should_stop_trading():
                    logger.warning(
                        "Daily loss limit reached. "
                        "Waiting for next day..."
                    )
                    await asyncio.sleep(3600)  # Wait 1 hour, check again
                    continue

                # Run strategy
                await strategy.scan_and_trade()

                # Print progress periodically
                if iteration % 10 == 0:
                    report = risk_manager.get_progress_report()
                    positions = position_manager.get_all_positions_summary({})
                    logger.info(
                        f"Progress: ${report['current_balance']:.2f} "
                        f"({report['progress_pct']:.1f}% to target) | "
                        f"Daily PnL: ${report['daily_pnl']:.2f} | "
                        f"Trades today: {report['daily_trades']} | "
                        f"Open positions: {len(positions)}"
                    )

                await asyncio.sleep(config.scan_interval_seconds)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

    finally:
        # Cleanup
        await analyzer.close()
        await exchange.close()
        logger.info("Bot shutdown complete")


def show_status(config: BotConfig):
    """Display current bot status and progress."""
    print("\n" + "=" * 50)
    print("  DEPOSIT ACCELERATION BOT - STATUS")
    print("=" * 50)
    print(f"  Initial deposit: ${config.risk.initial_deposit}")
    print(f"  Target:          ${config.risk.target_deposit}")
    print(f"  Leverage:        {config.exchange.default_leverage}x")
    print(f"  Position size:   {config.risk.min_position_pct:.0%}-"
          f"{config.risk.max_position_pct:.0%} of deposit")
    print(f"  Risk per trade:  {config.risk.max_risk_per_trade_pct:.1%}")
    print(f"  RR ratio:        1:{config.risk.risk_reward_ratio:.0f}")
    print(f"  Take profit:     {config.risk.take_profit_pct:.0%} price move")
    print(f"  Stop loss:       {config.risk.stop_loss_pct:.2%} price move")
    print(f"  Max daily loss:  {config.risk.max_daily_loss_pct:.0%}")
    print(f"  Max positions:   {config.risk.max_open_positions}")
    print(f"  Coins tracked:   {len(config.strategy.preferred_coins)}")
    print("=" * 50)

    # Math projection
    print("\n  COMPOUND GROWTH PROJECTION (ideal):")
    balance = config.risk.initial_deposit
    day = 0
    win_rate = 0.8  # 80% as per strategy
    avg_win = config.risk.max_profit_per_trade_pct
    avg_loss = config.risk.max_risk_per_trade_pct
    trades_per_day = 15  # Conservative estimate

    while balance < config.risk.target_deposit and day < 14:
        day += 1
        daily_pnl = 0
        for _ in range(trades_per_day):
            import random
            if random.random() < win_rate:
                daily_pnl += balance * avg_win
            else:
                daily_pnl -= balance * avg_loss

            # Cap daily loss
            if daily_pnl <= -(balance * config.risk.max_daily_loss_pct):
                break

        balance += daily_pnl
        print(f"  Day {day:2d}: ${balance:>8.2f}")

    if balance >= config.risk.target_deposit:
        print(f"\n  Target reachable in ~{day} days with 80% win rate")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Deposit Acceleration Trading Bot"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run with real money (default: testnet)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show bot configuration and growth projection"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    config = BotConfig()

    if args.status:
        show_status(config)
        return

    setup_logging(args.log_level)

    if args.live:
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("  Press Ctrl+C within 10 seconds to cancel...")
        logger.warning("=" * 60)
        try:
            import time
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Cancelled by user")
            return

    asyncio.run(run_bot(config, live=args.live))


if __name__ == "__main__":
    main()
