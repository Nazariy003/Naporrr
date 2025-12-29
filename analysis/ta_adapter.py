"""
Technical Analysis Adapter
Bridges the new TA-based signal generator with the existing bot infrastructure
"""

import time
import pandas as pd
from typing import Dict, Any, Optional
from collections import deque
from utils.logger import logger
from config.settings import settings
from analysis.ta_signal_generator import TechnicalAnalysisSignalGenerator
from data.storage import DataStorage


class TechnicalAnalysisAdapter:
    """
    Adapter for integrating TA-based analysis with existing bot infrastructure
    
    Converts:
    - Orderbook + Trades data → OHLCV DataFrame
    - TA signals → Old signal format for compatibility
    """
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.ta_generator = TechnicalAnalysisSignalGenerator()
        
        # Candle aggregation (create OHLCV from trades)
        self.candle_timeframe_sec = 3600  # 1 hour candles
        self.candles: Dict[str, deque] = {}  # symbol → deque of candles
        self.current_candle: Dict[str, Dict] = {}  # symbol → current forming candle
        
        # Minimum candles needed for TA (200 for MA200)
        self.min_candles_for_ta = 210
    
    def update_from_trades(self, symbol: str):
        """
        Update OHLCV candles from trade data in storage
        
        This aggregates real-time trade data into OHLCV candles for TA analysis
        """
        try:
            # Get trade data from storage
            trade_data = self.storage.get_trade_data(symbol)
            
            if not trade_data or len(trade_data) == 0:
                return
            
            # Initialize symbol data if needed
            if symbol not in self.candles:
                self.candles[symbol] = deque(maxlen=250)  # Keep 250 candles
                self.current_candle[symbol] = None
            
            # Aggregate trades into candles
            current_time = time.time()
            
            for trade in trade_data:
                trade_time = trade.get('timestamp', current_time)
                trade_price = float(trade.get('price', 0))
                trade_volume = float(trade.get('volume', 0))
                
                if trade_price <= 0:
                    continue
                
                # Determine candle timestamp (aligned to hour)
                candle_ts = int(trade_time // self.candle_timeframe_sec) * self.candle_timeframe_sec
                
                # Initialize or update current candle
                if self.current_candle[symbol] is None or self.current_candle[symbol]['timestamp'] != candle_ts:
                    # Close previous candle if exists
                    if self.current_candle[symbol] is not None:
                        self.candles[symbol].append(self.current_candle[symbol])
                    
                    # Start new candle
                    self.current_candle[symbol] = {
                        'timestamp': candle_ts,
                        'open': trade_price,
                        'high': trade_price,
                        'low': trade_price,
                        'close': trade_price,
                        'volume': trade_volume
                    }
                else:
                    # Update current candle
                    candle = self.current_candle[symbol]
                    candle['high'] = max(candle['high'], trade_price)
                    candle['low'] = min(candle['low'], trade_price)
                    candle['close'] = trade_price
                    candle['volume'] += trade_volume
        
        except Exception as e:
            logger.error(f"Error updating candles from trades for {symbol}: {e}")
    
    def update_from_orderbook(self, symbol: str):
        """
        Create a synthetic candle from current orderbook (for real-time updates)
        
        When we don't have enough trade history, we can use orderbook mid-price
        """
        try:
            orderbook = self.storage.get_order_book(symbol)
            
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                return
            
            # Get mid price
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            if best_bid <= 0 or best_ask <= 0:
                return
            
            mid_price = (best_bid + best_ask) / 2
            current_time = time.time()
            candle_ts = int(current_time // self.candle_timeframe_sec) * self.candle_timeframe_sec
            
            # Initialize symbol data if needed
            if symbol not in self.candles:
                self.candles[symbol] = deque(maxlen=250)
                self.current_candle[symbol] = None
            
            # Update or create current candle
            if self.current_candle[symbol] is None or self.current_candle[symbol]['timestamp'] != candle_ts:
                # Close previous candle
                if self.current_candle[symbol] is not None:
                    self.candles[symbol].append(self.current_candle[symbol])
                
                # Start new candle
                self.current_candle[symbol] = {
                    'timestamp': candle_ts,
                    'open': mid_price,
                    'high': mid_price,
                    'low': mid_price,
                    'close': mid_price,
                    'volume': 0  # No volume from orderbook
                }
            else:
                # Update current candle
                candle = self.current_candle[symbol]
                candle['high'] = max(candle['high'], mid_price)
                candle['low'] = min(candle['low'], mid_price)
                candle['close'] = mid_price
        
        except Exception as e:
            logger.error(f"Error updating candle from orderbook for {symbol}: {e}")
    
    def get_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get OHLCV DataFrame for symbol
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            or None if not enough data
        """
        try:
            if symbol not in self.candles:
                return None
            
            # Combine closed candles with current forming candle
            all_candles = list(self.candles[symbol])
            
            if self.current_candle.get(symbol):
                all_candles.append(self.current_candle[symbol])
            
            if len(all_candles) < self.min_candles_for_ta:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles)
            
            # Ensure correct types
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error creating DataFrame for {symbol}: {e}")
            return None
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate TA-based trading signal
        
        Returns signal in format compatible with old signal generator:
        {
            "symbol": str,
            "action": "BUY" | "SELL" | "HOLD",
            "strength": 0-5,
            "score_smoothed": float,
            "score_raw": float,
            ...
        }
        """
        try:
            # Update candles from latest data
            self.update_from_orderbook(symbol)
            
            # Get DataFrame
            df = self.get_dataframe(symbol)
            
            if df is None or len(df) < self.min_candles_for_ta:
                logger.debug(
                    f"[TA_ADAPTER] {symbol}: Not enough candles "
                    f"({len(df) if df is not None else 0}/{self.min_candles_for_ta})"
                )
                return self._create_hold_signal(symbol, "insufficient_data")
            
            # Generate TA signal
            ta_signal = self.ta_generator.generate_signal(
                symbol=symbol,
                df=df,
                current_portfolio_risk=0.0  # TODO: Calculate from open positions
            )
            
            # Convert to old format for compatibility
            old_format_signal = {
                # Basic signal
                "symbol": symbol,
                "action": ta_signal.action,
                "strength": ta_signal.strength,
                
                # Scores
                "score_smoothed": ta_signal.confidence / 100,
                "score_raw": ta_signal.confidence / 100,
                "imbalance_score": 0,  # Not used in TA mode
                "momentum_score": 0,   # Not used in TA mode
                
                # Pattern info
                "chart_pattern": ta_signal.chart_pattern,
                "candlestick_pattern": ta_signal.candlestick_pattern,
                
                # Confirmations
                "trend": ta_signal.trend,
                "trend_confirmed": ta_signal.trend_confirmed,
                "volume_confirmed": ta_signal.volume_confirmed,
                "indicator_confirmed": ta_signal.indicator_confirmed,
                
                # Technical indicators
                "rsi_signal": ta_signal.rsi_signal,
                "macd_signal": ta_signal.macd_signal,
                
                # Risk management
                "entry_price": ta_signal.entry_price,
                "stop_loss": ta_signal.stop_loss,
                "take_profit": ta_signal.take_profit,
                "risk_reward_ratio": ta_signal.risk_reward_ratio,
                "position_size_pct": ta_signal.position_size_pct,
                "leverage_recommended": ta_signal.leverage_recommended,
                
                # Metadata
                "spike": False,
                "spoof_filtered_volume": 0,
                "spread_bps": None,
                "volatility": 0,
                "cooldown_until": 0,
                "cooldown_active": False,
                "reason": ta_signal.reason,
                
                # Additional
                "confidence": ta_signal.confidence,
                "factors": {},
                "ohara_score": 0,
                "mtf_data": {}
            }
            
            # Log strong signals
            if ta_signal.strength >= 3:
                logger.info(
                    f"🎯 [TA_SIGNAL] {symbol}: {ta_signal.action}{ta_signal.strength} "
                    f"confidence={ta_signal.confidence:.0f}% "
                    f"patterns={ta_signal.chart_pattern or ta_signal.candlestick_pattern} "
                    f"trend={ta_signal.trend} rsi={ta_signal.rsi_signal} "
                    f"reason={ta_signal.reason}"
                )
            
            return old_format_signal
        
        except Exception as e:
            logger.error(f"Error generating TA signal for {symbol}: {e}")
            return self._create_hold_signal(symbol, "error")
    
    def _create_hold_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Create a HOLD signal in old format"""
        return {
            "symbol": symbol,
            "action": "HOLD",
            "strength": 0,
            "score_smoothed": 0,
            "score_raw": 0,
            "imbalance_score": 0,
            "momentum_score": 0,
            "chart_pattern": None,
            "candlestick_pattern": None,
            "trend": "NEUTRAL",
            "trend_confirmed": False,
            "volume_confirmed": False,
            "indicator_confirmed": False,
            "rsi_signal": "NEUTRAL",
            "macd_signal": "NEUTRAL",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward_ratio": 0,
            "position_size_pct": 0,
            "leverage_recommended": 0,
            "spike": False,
            "spoof_filtered_volume": 0,
            "spread_bps": None,
            "volatility": 0,
            "cooldown_until": 0,
            "cooldown_active": False,
            "reason": reason,
            "confidence": 0,
            "factors": {},
            "ohara_score": 0,
            "mtf_data": {}
        }
    
    def get_candle_count(self, symbol: str) -> int:
        """Get number of candles available for symbol"""
        if symbol not in self.candles:
            return 0
        return len(self.candles[symbol])
    
    def has_sufficient_data(self, symbol: str) -> bool:
        """Check if symbol has enough candles for TA analysis"""
        return self.get_candle_count(symbol) >= self.min_candles_for_ta