#!/usr/bin/env python3
"""
Quick Backtest Script
Run backtests on multiple symbols with TA strategy
"""

import asyncio
import argparse
from datetime import datetime
from analysis.pattern_backtester import PatternBacktester
from data.historical_fetcher import BybitHistoricalDataFetcher
from utils.logger import logger


async def run_backtest_for_symbol(
    symbol: str,
    interval: str = "4h",
    days_back: int = 180,
    initial_balance: float = 10000,
    min_signal_strength: int = 3
):
    """Run backtest for a single symbol"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting backtest for {symbol}")
    logger.info(f"{'='*60}\n")
    
    fetcher = BybitHistoricalDataFetcher()
    
    try:
        # Fetch historical data
        logger.info(f"Fetching {days_back} days of {interval} data for {symbol}...")
        df = await fetcher.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            days_back=days_back
        )
        
        if df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return None
        
        logger.info(f"Data fetched: {len(df)} candles")
        logger.info(f"Date range: {datetime.fromtimestamp(df.iloc[0]['timestamp']/1000)} to {datetime.fromtimestamp(df.iloc[-1]['timestamp']/1000)}\n")
        
        # Run backtest
        backtester = PatternBacktester(
            initial_balance=initial_balance,
            max_position_size=0.02,
            max_leverage=5.0
        )
        
        results = await backtester.run_backtest(
            symbol=symbol,
            df=df,
            min_signal_strength=min_signal_strength,
            max_holding_period_hours=72
        )
        
        # Print results
        backtester.print_results(results)
        
        return results
    
    finally:
        await fetcher.close()


async def run_multi_symbol_backtest(
    symbols: list,
    interval: str = "4h",
    days_back: int = 180,
    initial_balance: float = 10000,
    min_signal_strength: int = 3
):
    """Run backtests for multiple symbols"""
    logger.info(f"\n🔬 MULTI-SYMBOL BACKTEST")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Interval: {interval}, Days: {days_back}")
    logger.info(f"Initial Balance: ${initial_balance}, Min Strength: {min_signal_strength}\n")
    
    all_results = {}
    
    for symbol in symbols:
        result = await run_backtest_for_symbol(
            symbol=symbol,
            interval=interval,
            days_back=days_back,
            initial_balance=initial_balance,
            min_signal_strength=min_signal_strength
        )
        
        if result:
            all_results[symbol] = result
        
        # Small delay between symbols
        await asyncio.sleep(1)
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-SYMBOL BACKTEST SUMMARY")
    print("="*60)
    
    for symbol, result in all_results.items():
        status = "✅" if result.win_rate >= 60 else "⚠️"
        print(f"\n{status} {symbol}:")
        print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%")
        print(f"  P&L: ${result.total_pnl:.2f}, Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max DD: {result.max_drawdown_pct:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")
    
    # Overall statistics
    total_trades = sum(r.total_trades for r in all_results.values())
    total_wins = sum(r.winning_trades for r in all_results.values())
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
    
    if overall_win_rate >= 60:
        print(f"  ✅ SUCCESS: Overall win rate meets >60% requirement!")
    else:
        print(f"  ⚠️ WARNING: Overall win rate below 60% target")
    
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(description='Run TA Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Multiple trading pairs for multi-symbol backtest')
    parser.add_argument('--interval', type=str, default='4h',
                       choices=['1h', '4h', '1d'],
                       help='Candle interval')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of historical data')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial balance')
    parser.add_argument('--min-strength', type=int, default=3,
                       choices=[1, 2, 3, 4, 5],
                       help='Minimum signal strength')
    
    args = parser.parse_args()
    
    if args.symbols:
        # Multi-symbol backtest
        await run_multi_symbol_backtest(
            symbols=args.symbols,
            interval=args.interval,
            days_back=args.days,
            initial_balance=args.balance,
            min_signal_strength=args.min_strength
        )
    else:
        # Single symbol backtest
        await run_backtest_for_symbol(
            symbol=args.symbol,
            interval=args.interval,
            days_back=args.days,
            initial_balance=args.balance,
            min_signal_strength=args.min_strength
        )


if __name__ == "__main__":
    # Example usage:
    # python run_backtest.py --symbol BTCUSDT --interval 4h --days 180
    # python run_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 4h --days 90
    
    asyncio.run(main())
