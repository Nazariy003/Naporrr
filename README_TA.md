# Crypto Trading Bot - Technical Analysis Mode

## Overview

This crypto trading bot has been completely rewritten and optimized to implement an automated trading algorithm for Bybit futures based on **Technical Analysis** from renowned trading literature.

## Trading Strategy Foundation

The bot's strategy is based on four authoritative books on technical analysis:

### 1. Thomas Bulkowski's "Encyclopedia of Chart Patterns"
- **Chart Patterns** with 60-83% success rates:
  - Double Bottom (bullish reversal)
  - Head and Shoulders (bearish reversal)
  - Triangle Patterns (continuation/reversal)
- All patterns validated with volume confirmation

### 2. John Murphy's "Technical Analysis of the Financial Markets"
- **Trend Identification**: 200-day Moving Average (primary trend)
- **Momentum Indicators**:
  - RSI (Relative Strength Index): Overbought >70, Oversold <30
  - MACD (Moving Average Convergence Divergence): Trend confirmation
  - Stochastic Oscillator: Momentum confirmation
- **Volatility Indicators**:
  - Bollinger Bands: Volatility measurement
  - ATR (Average True Range): Risk management

### 3. Steve Nison's "Japanese Candlestick Charting Techniques"
- **Candlestick Patterns** with 60-70% reliability:
  - Hammer (bullish reversal at support)
  - Bullish/Bearish Engulfing
  - Morning Star (bullish reversal)
  - Doji (indecision/reversal at extremes)
- All patterns require volume confirmation

### 4. Steven Bigalow's "High Profit Candlestick Patterns"
- **Risk Management**:
  - Stop-loss: 1-2% below pattern low (longs) or above pattern high (shorts)
  - Take-profit: 2:1 reward-risk ratio based on pattern height
  - Position sizing: 1-2% risk per trade, max 8% portfolio
  - Leverage: 2-5x based on signal strength and volatility

## Key Features

### Multi-Pair Trading
- Supports 5-10 trading pairs simultaneously
- Default pairs: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, AVAXUSDT, TRXUSDT, AAVEUSDT, STRKUSDT
- Intelligent batch processing for API rate limit management

### Entry Logic

**BUY Signals** (all conditions must be met):
1. Bullish chart pattern (Double Bottom) OR bullish candlestick pattern (Hammer/Engulfing/Morning Star)
2. Trend confirmation: Price > 200-day MA (uptrend)
3. RSI < 40 (preferably < 30 for stronger signal)
4. MACD bullish trend
5. Volume confirmation: Pattern shows increased volume

**SELL Signals** (all conditions must be met):
1. Bearish chart pattern (Head & Shoulders) OR bearish candlestick pattern (Bearish Engulfing)
2. Trend confirmation: Price < 200-day MA (downtrend)
3. RSI > 60 (preferably > 70 for stronger signal)
4. MACD bearish trend
5. Volume confirmation: Pattern shows increased volume

### Exit Logic

**Stop-Loss** (Bigalow's risk management):
- Long positions: 1-2% below pattern low
- Short positions: 1-2% above pattern high
- Alternative: 1.5x ATR if no pattern level available

**Take-Profit** (2:1 reward-risk ratio):
- Based on pattern height projection
- Minimum 2:1 reward-risk ratio
- Alternative: 3x ATR for 2:1 ratio

### Risk Management

**Position Sizing**:
- Base: 1.5-2% of portfolio per trade
- Adjusted for signal strength (1-5 scale)
- Reduced by 30% in high volatility
- Maximum portfolio exposure: 8%

**Leverage**:
- Minimum: 2x
- Maximum: 5x
- Default: 3x
- Increased for stronger signals (strength 5)
- Reduced in high volatility or weak trends

### Backtesting

The bot includes a comprehensive backtesting module that:
- Tests strategies on historical Bybit data
- Validates >60% success rate requirement
- Provides detailed metrics:
  - Win rate (%)
  - Profit factor
  - Maximum drawdown
  - Sharpe ratio
  - Pattern-specific performance

## Installation

### Prerequisites

1. Python 3.9 or higher
2. TA-Lib library (required for technical indicators)

### TA-Lib Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

**macOS:**
```bash
brew install ta-lib
```

**Windows:**
Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Enabling Technical Analysis Mode

Edit `config/settings.py` or set in your environment:

```python
class TechnicalAnalysisSettings(BaseSettings):
    enable_ta_mode: bool = True  # Enable TA mode (disable for legacy imbalance mode)
```

### Key Configuration Parameters

**Signal Generation**:
```python
min_signal_strength: int = 3  # Minimum signal strength (1-5) to trade
min_confidence: float = 65.0  # Minimum confidence % to trade
```

**Pattern Detection**:
```python
double_bottom_min_confidence: float = 60.0
head_shoulders_min_confidence: float = 60.0
triangle_min_confidence: float = 60.0
hammer_min_confidence: float = 60.0
engulfing_min_confidence: float = 65.0
morning_star_min_confidence: float = 70.0
```

**Risk Management**:
```python
risk_per_trade_pct: float = 0.015  # 1.5% per trade
max_portfolio_risk_pct: float = 0.08  # 8% max
min_risk_reward_ratio: float = 1.5
target_risk_reward_ratio: float = 2.0
```

**Leverage**:
```python
min_leverage: float = 2.0
max_leverage: float = 5.0
default_leverage: float = 3.0
```

**Trading Pairs**:
```python
trade_pairs: list = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT",
    "AAVEUSDT", "STRKUSDT"
]
```

## Running the Bot

### Live Trading (Demo Mode)

```bash
python main.py
```

The bot will start in the mode specified in `config/settings.py`:
- `mode: "DEMO"` - Paper trading with virtual funds (safe for testing)
- `mode: "LIVE"` - Real trading with actual funds (⚠️ use with caution)

### Backtesting

Run backtest on historical data:

```bash
python -m analysis.pattern_backtester
```

Or use the backtesting module programmatically:

```python
import asyncio
from analysis.pattern_backtester import PatternBacktester
from data.historical_fetcher import BybitHistoricalDataFetcher

async def run_backtest():
    # Fetch historical data
    fetcher = BybitHistoricalDataFetcher()
    df = await fetcher.fetch_historical_data(
        symbol="BTCUSDT",
        interval="4h",
        days_back=180
    )
    
    # Run backtest
    backtester = PatternBacktester(
        initial_balance=10000,
        max_position_size=0.02,
        max_leverage=5.0
    )
    
    results = await backtester.run_backtest(
        symbol="BTCUSDT",
        df=df,
        min_signal_strength=3
    )
    
    # Print results
    backtester.print_results(results)
    await fetcher.close()

asyncio.run(run_backtest())
```

## Bot Operation

### Startup

When started in TA mode, the bot will:
1. Connect to Bybit WebSocket and REST API
2. Start collecting real-time orderbook and trade data
3. Aggregate trades into OHLCV candles (1-hour default)
4. Wait for sufficient historical data (210 candles minimum for 200-MA)
5. Begin pattern detection and signal generation

### Data Collection

The bot needs historical candle data for technical analysis:
- **Minimum**: 210 candles (for 200-day MA calculation)
- **Timeframe**: 1-hour candles (configurable)
- **Source**: Aggregated from real-time trade data or fetched from Bybit API

Initial startup may take time to build sufficient historical data. The bot will log progress:
```
⏳ [TA_DATA] BTCUSDT: Building candle history... (150/210)
✅ [TA_DATA] BTCUSDT: Sufficient candle data ready for TA analysis
```

### Signal Generation

The bot analyzes each symbol every 2 seconds (configurable):
1. Updates OHLCV candles from latest trades
2. Runs pattern detection (chart patterns + candlestick patterns)
3. Calculates technical indicators (trend, RSI, MACD, Stochastic, Bollinger)
4. Combines signals with confidence scoring
5. Generates BUY/SELL/HOLD action with strength 1-5

Strong signals (strength ≥ 3) are logged:
```
🎯 [TA_SIGNAL] BTCUSDT: BUY3 confidence=75% patterns=DOUBLE_BOTTOM 
   trend=BULLISH rsi=OVERSOLD reason=double_bottom_70%, rsi_oversold_28, macd_bullish
```

### Trade Execution

When a signal meets the minimum requirements (strength ≥ 3 by default):
```
💼 [ORCH_TA] Executing BUY3 for BTCUSDT (confidence=75%, reason=double_bottom_70%, rsi_oversold_28)
   Entry: 42500.00, SL: 41650.00, TP: 44200.00, R:R=2.0:1, Size: 2.0%, Leverage: 3x
   Pattern: DOUBLE_BOTTOM, Trend: BULLISH, RSI: OVERSOLD, MACD: BULLISH
```

## Performance Monitoring

### Log Files

- `logs/bot.log` - Main bot log
- `logs/errors.log` - Error log
- `logs/trades.csv` - Trade history

### Success Rate Validation

The backtesting module validates that patterns meet the >60% success rate requirement from the problem statement. Example output:

```
📊 TRADE STATISTICS:
  Total Trades:     45
  Winning Trades:   32
  Losing Trades:    13
  Win Rate:         71.11%

✅ SUCCESS: Win rate 71.11% meets >60% requirement!
```

## Modes of Operation

### TA Mode (New)
- **Enable**: `enable_ta_mode = True`
- Uses chart patterns, candlestick patterns, and technical indicators
- Based on Bulkowski, Murphy, Nison, and Bigalow methodologies
- Requires historical candle data (210+ candles)

### Legacy Mode (Original)
- **Enable**: `enable_ta_mode = False`
- Uses orderbook imbalance and volume analysis
- Based on O'Hara market microstructure theory
- Works with real-time orderbook depth

Both modes can run simultaneously (toggle in settings).

## Emergency Features

The bot includes emergency position management:
- Automatically detects stuck positions (no update for 5+ minutes)
- Forces closure to prevent capital lockup
- Logs all emergency actions

## API Rate Limits

The bot manages Bybit API rate limits through:
- Batch processing (3 symbols per batch)
- Intelligent caching
- Adaptive request delays
- Priority queueing for critical requests

## Troubleshooting

### "Insufficient TA data"
- **Cause**: Not enough historical candles for analysis
- **Solution**: Wait for bot to collect 210+ candles, or use historical data fetcher

### "TA-Lib not available"
- **Cause**: TA-Lib library not installed
- **Solution**: Install TA-Lib (see Installation section)

### Low win rate in backtesting
- **Cause**: Parameters may need tuning for specific market conditions
- **Solution**: Adjust confidence thresholds, timeframes, or risk parameters

### No signals generated
- **Cause**: Market conditions don't meet pattern criteria
- **Solution**: Normal - bot waits for high-quality setups. Adjust min_signal_strength if needed.

## References

1. Bulkowski, Thomas N. "Encyclopedia of Chart Patterns" (2021)
2. Murphy, John J. "Technical Analysis of the Financial Markets" (1999)
3. Nison, Steve. "Japanese Candlestick Charting Techniques" (2001)
4. Bigalow, Stephen W. "High Profit Candlestick Patterns" (2005)

## License

[Your License Here]

## Disclaimer

⚠️ **Trading cryptocurrencies involves significant risk of loss. This bot is provided for educational purposes. Past performance does not guarantee future results. Always test thoroughly in demo mode before using real funds. Use at your own risk.**
