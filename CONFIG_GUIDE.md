# Technical Analysis Configuration Guide

This guide explains all configuration parameters for the Technical Analysis (TA) mode.

## Table of Contents
1. [Enabling TA Mode](#enabling-ta-mode)
2. [Signal Generation Settings](#signal-generation-settings)
3. [Pattern Detection Settings](#pattern-detection-settings)
4. [Technical Indicator Settings](#technical-indicator-settings)
5. [Risk Management Settings](#risk-management-settings)
6. [Trading Pair Settings](#trading-pair-settings)
7. [Backtesting Settings](#backtesting-settings)
8. [Advanced Settings](#advanced-settings)

## Enabling TA Mode

Location: `config/settings.py` → `TechnicalAnalysisSettings`

```python
enable_ta_mode: bool = True  # Set to True for TA mode, False for legacy mode
```

**Impact**: This is the master switch that determines whether the bot uses:
- **True**: Chart patterns + candlestick patterns + technical indicators (Murphy, Bulkowski, Nison, Bigalow)
- **False**: Orderbook imbalance + volume analysis (O'Hara market microstructure)

## Signal Generation Settings

### Minimum Requirements for Trading

```python
min_signal_strength: int = 3  # Range: 1-5
```
- Minimum signal strength to execute a trade
- **Recommended**: 3 (medium-strong signals)
- Lower = more trades but lower quality
- Higher = fewer trades but higher quality

```python
min_confidence: float = 65.0  # Range: 0-100
```
- Minimum confidence percentage to trade
- **Recommended**: 65.0 for balanced approach
- Based on pattern success rates from Bulkowski's research

### Confirmation Requirements

```python
require_trend_confirmation: bool = True
```
- Require trend alignment with 200-day MA
- **Recommended**: True (Murphy's "trend is your friend")
- BUY only in uptrends, SELL only in downtrends

```python
require_volume_confirmation: bool = True
```
- Require increased volume on pattern formation
- **Recommended**: True (Bulkowski/Nison confirmation)
- Patterns without volume are less reliable

```python
require_indicator_confirmation: bool = False
```
- Require both RSI and MACD confirmation
- **Recommended**: False (too strict, reduces opportunities)
- Set to True for ultra-conservative approach

## Pattern Detection Settings

### Chart Patterns (Bulkowski)

```python
enable_chart_patterns: bool = True
```
- Enable/disable chart pattern detection
- Includes: Double Bottom, Head & Shoulders, Triangles

**Individual Pattern Confidence Thresholds:**

```python
double_bottom_min_confidence: float = 60.0  # Bullish reversal
head_shoulders_min_confidence: float = 60.0  # Bearish reversal
triangle_min_confidence: float = 60.0  # Continuation/reversal
```

**Tuning Guide**:
- **60-65%**: Standard (matches Bulkowski's average success rates)
- **70-75%**: Conservative (fewer but higher quality signals)
- **50-55%**: Aggressive (more signals but lower reliability)

### Candlestick Patterns (Nison/Bigalow)

```python
enable_candlestick_patterns: bool = True
```
- Enable/disable candlestick pattern detection
- Includes: Hammer, Engulfing, Morning Star, Doji

**Individual Pattern Confidence Thresholds:**

```python
hammer_min_confidence: float = 60.0  # Bullish reversal at support
engulfing_min_confidence: float = 65.0  # Reversal (bullish or bearish)
morning_star_min_confidence: float = 70.0  # Strong bullish reversal
```

**Tuning Guide**:
- Hammer: 60% at support, lower (55%) if volume confirmed
- Engulfing: 65% with volume, 70% for strong confirmation
- Morning Star: 70% (most reliable, 3-candle pattern)

## Technical Indicator Settings

### Trend Analysis (Murphy)

```python
enable_trend_analysis: bool = True
```
- Use 200-day MA for primary trend identification
- **Recommended**: True (essential for trend following)
- Based on Murphy's trend analysis

### Momentum Indicators

```python
enable_rsi: bool = True  # Relative Strength Index
```
- Detects overbought (>70) and oversold (<30) conditions
- **Critical for entry timing**

```python
enable_macd: bool = True  # Moving Average Convergence Divergence
```
- Confirms trend direction
- **Recommended**: True for trend confirmation

```python
enable_stochastic: bool = True  # Stochastic Oscillator
```
- Additional momentum confirmation
- **Optional**: Can disable if too many filters

### Volatility Indicators

```python
enable_bollinger_bands: bool = True
```
- Measures volatility and price extremes
- **Used for**: Volatility assessment

## Risk Management Settings

Based on Steven Bigalow's "High Profit Candlestick Patterns"

### Position Sizing

```python
risk_per_trade_pct: float = 0.015  # 1.5% per trade
```
- Maximum risk per individual trade
- **Conservative**: 0.01 (1%)
- **Standard**: 0.015 (1.5%)
- **Aggressive**: 0.02 (2%)

```python
max_portfolio_risk_pct: float = 0.08  # 8% max
```
- Maximum total portfolio exposure
- **Conservative**: 0.05 (5%)
- **Standard**: 0.08 (8%)
- **Aggressive**: 0.10 (10%)

```python
base_position_size_pct: float = 0.02  # 2% base size
```
- Base position size as % of portfolio
- Adjusted by signal strength and volatility

**Size Adjustments**:
```python
strong_signal_multiplier: float = 1.5  # Increase for strength 5
high_volatility_reduction: float = 0.7  # Reduce 30% in high vol
```

### Stop-Loss Configuration

```python
stop_loss_pct_min: float = 0.01  # 1%
stop_loss_pct_max: float = 0.02  # 2%
stop_loss_atr_multiplier: float = 1.5  # 1.5x ATR
```

**Stop-Loss Placement**:
- **Pattern-based** (preferred): 1-2% below pattern low (longs) or above high (shorts)
- **ATR-based** (fallback): 1.5x Average True Range

**Tuning**:
- Tighter stops (1%): Less risk but more stopouts
- Wider stops (2%): More breathing room but larger losses

### Take-Profit Configuration

```python
min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1
target_risk_reward_ratio: float = 2.0  # Target 2:1
```

**Take-Profit Calculation**:
- **Pattern projection** (preferred): Based on pattern height
- **ATR-based** (fallback): 3x ATR for 2:1 ratio from 1.5 ATR stop

**Tuning**:
- 1.5:1 ratio: More wins but smaller profits
- 2:1 ratio: Standard (Bigalow recommendation)
- 3:1 ratio: Fewer wins but larger profits

### Leverage Settings

```python
min_leverage: float = 2.0
max_leverage: float = 5.0
default_leverage: float = 3.0
```

**Leverage Strategy**:
- Strength 5 signals: Up to 5x
- Strength 4 signals: 4x
- Strength 3 signals: 3x
- Strength 1-2: 2x
- Reduced in high volatility

**Safety**:
- **Conservative**: max_leverage = 3.0
- **Standard**: max_leverage = 5.0
- **Aggressive**: max_leverage = 10.0 (⚠️ high risk)

### Holding Periods

```python
max_holding_period_hours: int = 72  # 3 days
min_holding_period_hours: int = 1  # 1 hour
```

**Purpose**:
- Max: Prevent stuck positions
- Min: Avoid premature exits

**Crypto-Specific**:
- Crypto volatility: 48-72 hours typical
- Patterns can play out quickly
- Adjust based on timeframe (4h candles = longer holds)

## Trading Pair Settings

Location: `config/settings.py` → `PairsSettings`

```python
trade_pairs: list = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "BNBUSDT",   # Binance Coin
    "SOLUSDT",   # Solana
    "ADAUSDT",   # Cardano
    "DOGEUSDT",  # Dogecoin
    "AVAXUSDT",  # Avalanche
    "TRXUSDT",   # Tron
    "AAVEUSDT",  # Aave
    "STRKUSDT"   # Stark
]
```

**Recommendations**:
- Start with 5-10 pairs as per problem statement
- Focus on high liquidity pairs (BTC, ETH, SOL)
- Avoid very low liquidity pairs (patterns less reliable)

```python
excluded_pairs: list = ["HFTUSDT"]  # Low liquidity pairs to skip
```

## Backtesting Settings

```python
backtest_enabled: bool = False
backtest_initial_balance: float = 10000.0
backtest_commission_rate: float = 0.0006  # 0.06% Bybit
```

**Running Backtests**:
```bash
python run_backtest.py --symbol BTCUSDT --interval 4h --days 180
```

**Interpreting Results**:
- Win rate ≥ 60%: Good (meets requirement)
- Win rate ≥ 70%: Excellent
- Profit factor ≥ 1.5: Profitable
- Sharpe ratio ≥ 1.0: Good risk-adjusted returns

## Advanced Settings

### Candle Aggregation

```python
candle_timeframe_seconds: int = 3600  # 1 hour
min_candles_required: int = 210  # For 200-MA
```

**Timeframe Options**:
- 3600 (1h): Standard, good balance
- 14400 (4h): Longer trends, fewer signals
- 86400 (1d): Very long trends, rare signals

**Data Requirements**:
- Minimum: 210 candles (for 200-day MA)
- More is better for pattern reliability

### Multi-Pair Optimization

```python
# In trading/orchestrator_ta.py
batch_size = 3  # Process 3 symbols at a time
decision_interval_sec = 2.0  # Check every 2 seconds
```

**Tuning for Performance**:
- More pairs: Increase batch size
- Faster CPU: Reduce decision interval
- API rate limits: Increase decision interval

## Recommended Configurations

### Conservative (Low Risk)

```python
min_signal_strength = 4
min_confidence = 70.0
require_trend_confirmation = True
require_volume_confirmation = True
require_indicator_confirmation = True
risk_per_trade_pct = 0.01
max_leverage = 3.0
```

### Balanced (Default)

```python
min_signal_strength = 3
min_confidence = 65.0
require_trend_confirmation = True
require_volume_confirmation = True
require_indicator_confirmation = False
risk_per_trade_pct = 0.015
max_leverage = 5.0
```

### Aggressive (High Risk)

```python
min_signal_strength = 2
min_confidence = 60.0
require_trend_confirmation = True
require_volume_confirmation = False
require_indicator_confirmation = False
risk_per_trade_pct = 0.02
max_leverage = 5.0
```

## Testing Your Configuration

1. **Start with backtesting**:
   ```bash
   python run_backtest.py --symbol BTCUSDT --interval 4h --days 180
   ```

2. **Validate win rate ≥ 60%**

3. **Test in demo mode**:
   Set `mode = "DEMO"` in settings

4. **Monitor for 1-2 weeks** before live trading

5. **Gradually move to live** with small position sizes

## Common Issues

### Too Few Signals
- Lower `min_signal_strength` to 2
- Lower `min_confidence` to 60.0
- Disable `require_indicator_confirmation`

### Too Many False Signals
- Increase `min_signal_strength` to 4
- Increase confidence thresholds
- Enable `require_indicator_confirmation`

### Low Win Rate in Backtest
- Check if patterns are being detected correctly
- Ensure volume confirmation is working
- Adjust stop-loss distances (may be too tight)
- Verify trend confirmation is appropriate

### High Drawdown
- Reduce `max_portfolio_risk_pct`
- Reduce `max_leverage`
- Tighten `stop_loss_pct_min/max`

## Support

For issues or questions:
1. Check logs in `logs/bot.log`
2. Review README_TA.md for details
3. Run backtests to validate configuration
4. Test in demo mode before live trading

---

**Remember**: Always test thoroughly in demo mode before using real funds. Past performance does not guarantee future results.
