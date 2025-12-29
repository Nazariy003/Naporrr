"""
Pattern-Based Backtesting Engine
Tests trading strategies based on chart patterns and technical indicators

Validates >60% success rate requirement from problem statement
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from utils.logger import logger
from analysis.ta_signal_generator import TechnicalAnalysisSignalGenerator, TradingSignal
from data.historical_fetcher import BybitHistoricalDataFetcher


@dataclass
class Trade:
    """Single trade record"""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    direction: str = "BUY"  # "BUY" or "SELL"
    size: float = 0.0
    leverage: float = 1.0
    
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Pattern information
    chart_pattern: Optional[str] = None
    candlestick_pattern: Optional[str] = None
    
    # Exit reason
    exit_reason: str = "OPEN"  # "TP", "SL", "TIMEOUT", "SIGNAL"
    
    # P&L
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Metadata
    signal_strength: int = 0
    signal_confidence: float = 0.0


@dataclass
class BacktestResults:
    """Backtest performance results"""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L statistics
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0  # Total wins / Total losses
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Performance by pattern
    pattern_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    
    # Trades
    trades: List[Trade] = field(default_factory=list)


class PatternBacktester:
    """
    Backtesting engine for pattern-based trading strategies
    
    Tests:
    - Chart patterns (Bulkowski)
    - Candlestick patterns (Nison/Bigalow)
    - Technical indicators (Murphy)
    - Risk management (Bigalow)
    """
    
    def __init__(
        self,
        initial_balance: float = 10000,
        max_position_size: float = 0.02,  # 2% per trade
        max_leverage: float = 5.0,
        commission_rate: float = 0.0006  # 0.06% (Bybit futures taker fee)
    ):
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.commission_rate = commission_rate
        
        self.signal_generator = TechnicalAnalysisSignalGenerator()
        
        # State
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_history: List[float] = [initial_balance]
    
    async def run_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        min_signal_strength: int = 3,
        max_holding_period_hours: int = 72  # 3 days max
    ) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV data
            min_signal_strength: Minimum signal strength to trade (1-5)
            max_holding_period_hours: Maximum holding period
            
        Returns:
            BacktestResults with performance metrics
        """
        logger.info(f"🔬 Starting backtest for {symbol} with {len(df)} candles")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_history = [self.initial_balance]
        
        # Iterate through candles
        for i in range(200, len(df)):  # Start after 200 candles for MA200
            current_candle = df.iloc[i]
            current_price = current_candle['close']
            current_time = datetime.fromtimestamp(current_candle['timestamp'] / 1000)
            
            # Get historical data up to current point
            historical_df = df.iloc[:i+1].copy()
            
            # Check open positions for exit conditions
            self._check_exit_conditions(
                current_time=current_time,
                current_price=current_price,
                max_holding_hours=max_holding_period_hours
            )
            
            # Generate signal if no open position (one position at a time for simplicity)
            if len(self.open_trades) == 0:
                signal = self.signal_generator.generate_signal(
                    symbol=symbol,
                    df=historical_df,
                    current_portfolio_risk=0.0
                )
                
                # Enter trade if signal meets criteria
                if signal.action in ["BUY", "SELL"] and signal.strength >= min_signal_strength:
                    self._enter_trade(
                        signal=signal,
                        entry_time=current_time,
                        entry_price=current_price
                    )
            
            # Update equity
            self._update_equity(current_price)
        
        # Close any remaining open trades at last price
        final_price = df.iloc[-1]['close']
        final_time = datetime.fromtimestamp(df.iloc[-1]['timestamp'] / 1000)
        for trade in self.open_trades[:]:
            self._exit_trade(
                trade=trade,
                exit_time=final_time,
                exit_price=final_price,
                reason="BACKTEST_END"
            )
        
        # Calculate results
        results = self._calculate_results()
        
        logger.info(f"✅ Backtest completed: {results.total_trades} trades, {results.win_rate:.1f}% win rate")
        
        return results
    
    def _enter_trade(
        self,
        signal: TradingSignal,
        entry_time: datetime,
        entry_price: float
    ):
        """Enter a new trade"""
        # Calculate position size
        risk_amount = self.current_balance * signal.position_size_pct
        
        # Account for leverage
        leverage = min(signal.leverage_recommended, self.max_leverage)
        position_value = risk_amount * leverage
        
        # Calculate size in contracts/coins
        size = position_value / entry_price
        
        # Commission
        commission = position_value * self.commission_rate
        self.current_balance -= commission
        
        trade = Trade(
            symbol=signal.symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            direction=signal.action,
            size=size,
            leverage=leverage,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            chart_pattern=signal.chart_pattern,
            candlestick_pattern=signal.candlestick_pattern,
            signal_strength=signal.strength,
            signal_confidence=signal.confidence
        )
        
        self.open_trades.append(trade)
        
        logger.info(
            f"📈 ENTRY {signal.action}: {signal.symbol} @ {entry_price:.2f} "
            f"(size={size:.4f}, leverage={leverage}x, SL={signal.stop_loss:.2f}, TP={signal.take_profit:.2f}) "
            f"Pattern: {signal.chart_pattern or signal.candlestick_pattern}"
        )
    
    def _check_exit_conditions(
        self,
        current_time: datetime,
        current_price: float,
        max_holding_hours: int
    ):
        """Check if any open trades should be exited"""
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            # Check stop-loss
            if trade.direction == "BUY" and trade.stop_loss > 0:
                if current_price <= trade.stop_loss:
                    self._exit_trade(trade, current_time, current_price, "SL")
                    continue
            elif trade.direction == "SELL" and trade.stop_loss > 0:
                if current_price >= trade.stop_loss:
                    self._exit_trade(trade, current_time, current_price, "SL")
                    continue
            
            # Check take-profit
            if trade.direction == "BUY" and trade.take_profit > 0:
                if current_price >= trade.take_profit:
                    self._exit_trade(trade, current_time, current_price, "TP")
                    continue
            elif trade.direction == "SELL" and trade.take_profit > 0:
                if current_price <= trade.take_profit:
                    self._exit_trade(trade, current_time, current_price, "TP")
                    continue
            
            # Check timeout
            holding_time = (current_time - trade.entry_time).total_seconds() / 3600  # hours
            if holding_time >= max_holding_hours:
                self._exit_trade(trade, current_time, current_price, "TIMEOUT")
                continue
    
    def _exit_trade(
        self,
        trade: Trade,
        exit_time: datetime,
        exit_price: float,
        reason: str
    ):
        """Exit a trade"""
        # Calculate P&L
        if trade.direction == "BUY":
            price_diff = exit_price - trade.entry_price
        else:  # SELL
            price_diff = trade.entry_price - exit_price
        
        # P&L with leverage
        pnl = (price_diff / trade.entry_price) * (trade.size * trade.entry_price) * trade.leverage
        pnl_pct = (price_diff / trade.entry_price) * 100 * trade.leverage
        
        # Commission on exit
        position_value = trade.size * exit_price
        commission = position_value * self.commission_rate
        pnl -= commission
        
        # Update balance
        self.current_balance += pnl
        
        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        
        # Move to closed trades
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        logger.info(
            f"📉 EXIT {trade.direction}: {trade.symbol} @ {exit_price:.2f} "
            f"({reason}) P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) "
            f"Balance: ${self.current_balance:.2f}"
        )
    
    def _update_equity(self, current_price: float):
        """Update equity curve"""
        # Calculate unrealized P&L from open trades
        unrealized_pnl = 0
        for trade in self.open_trades:
            if trade.direction == "BUY":
                price_diff = current_price - trade.entry_price
            else:
                price_diff = trade.entry_price - current_price
            
            pnl = (price_diff / trade.entry_price) * (trade.size * trade.entry_price) * trade.leverage
            unrealized_pnl += pnl
        
        current_equity = self.current_balance + unrealized_pnl
        self.equity_history.append(current_equity)
        
        # Update peak
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate backtest results"""
        results = BacktestResults()
        
        if not self.closed_trades:
            logger.warning("No trades executed in backtest")
            return results
        
        # Basic statistics
        results.total_trades = len(self.closed_trades)
        results.winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        results.losing_trades = sum(1 for t in self.closed_trades if t.pnl < 0)
        results.win_rate = (results.winning_trades / results.total_trades) * 100 if results.total_trades > 0 else 0
        
        # P&L statistics
        results.total_pnl = sum(t.pnl for t in self.closed_trades)
        
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl < 0]
        
        if winning_trades:
            results.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            results.largest_win = max(t.pnl for t in winning_trades)
        
        if losing_trades:
            results.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            results.largest_loss = min(t.pnl for t in losing_trades)
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        results.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        equity_array = np.array(self.equity_history)
        peak_array = np.maximum.accumulate(equity_array)
        drawdown_array = (equity_array - peak_array) / peak_array
        results.max_drawdown_pct = abs(drawdown_array.min()) * 100
        results.max_drawdown = abs((equity_array - peak_array).min())
        
        # Sharpe ratio (simplified)
        if len(self.equity_history) > 1:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            if returns.std() > 0:
                from config.settings import settings
                trading_days = settings.technical_analysis.backtest_trading_days_per_year
                results.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_days)  # Annualized
        
        # Pattern performance
        pattern_stats = {}
        for trade in self.closed_trades:
            pattern = trade.chart_pattern or trade.candlestick_pattern
            if pattern:
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }
                
                pattern_stats[pattern]['trades'] += 1
                if trade.pnl > 0:
                    pattern_stats[pattern]['wins'] += 1
                pattern_stats[pattern]['total_pnl'] += trade.pnl
        
        # Calculate win rates for each pattern
        for pattern, stats in pattern_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
        
        results.pattern_performance = pattern_stats
        results.equity_curve = self.equity_history
        results.trades = self.closed_trades
        
        return results
    
    def print_results(self, results: BacktestResults):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total Trades:     {results.total_trades}")
        print(f"  Winning Trades:   {results.winning_trades}")
        print(f"  Losing Trades:    {results.losing_trades}")
        print(f"  Win Rate:         {results.win_rate:.2f}%")
        
        print(f"\n💰 P&L STATISTICS:")
        print(f"  Total P&L:        ${results.total_pnl:.2f}")
        print(f"  Average Win:      ${results.avg_win:.2f}")
        print(f"  Average Loss:     ${results.avg_loss:.2f}")
        print(f"  Largest Win:      ${results.largest_win:.2f}")
        print(f"  Largest Loss:     ${results.largest_loss:.2f}")
        print(f"  Profit Factor:    {results.profit_factor:.2f}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown:     ${results.max_drawdown:.2f} ({results.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:     {results.sharpe_ratio:.2f}")
        
        if results.pattern_performance:
            print(f"\n📈 PATTERN PERFORMANCE:")
            for pattern, stats in results.pattern_performance.items():
                print(f"  {pattern}:")
                print(f"    Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1f}%, P&L: ${stats['total_pnl']:.2f}")
        
        print("\n" + "="*60)
        
        # Validation check
        if results.win_rate >= 60:
            print(f"✅ SUCCESS: Win rate {results.win_rate:.1f}% meets >60% requirement!")
        else:
            print(f"⚠️ WARNING: Win rate {results.win_rate:.1f}% below 60% target")
        
        print("="*60 + "\n")


async def main():
    """Example usage"""
    # Fetch historical data
    fetcher = BybitHistoricalDataFetcher()
    
    try:
        df = await fetcher.fetch_historical_data(
            symbol="BTCUSDT",
            interval="4h",
            days_back=180  # 6 months
        )
        
        if df.empty:
            logger.error("Failed to fetch historical data")
            return
        
        # Run backtest
        backtester = PatternBacktester(
            initial_balance=10000,
            max_position_size=0.02,
            max_leverage=5.0
        )
        
        results = await backtester.run_backtest(
            symbol="BTCUSDT",
            df=df,
            min_signal_strength=3,
            max_holding_period_hours=72
        )
        
        # Print results
        backtester.print_results(results)
    
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())