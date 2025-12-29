"""
Technical Indicators Module
Based on John Murphy's "Technical Analysis of the Financial Markets"

This module implements key technical indicators used for trend identification,
momentum analysis, and trading signal confirmation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from utils.logger import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using pandas-ta fallback")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available")


@dataclass
class TrendData:
    """Trend identification based on Moving Averages (Murphy, Chapter 9)"""
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0-100
    sma_200: float
    sma_50: float
    ema_20: float
    price_vs_ma200: float  # percentage above/below 200-day MA
    golden_cross: bool  # 50 SMA crosses above 200 SMA (bullish)
    death_cross: bool   # 50 SMA crosses below 200 SMA (bearish)


@dataclass
class MomentumData:
    """Momentum indicators (Murphy, Chapter 10)"""
    rsi: float  # 0-100, overbought >70, oversold <30
    rsi_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    stoch_k: float  # 0-100
    stoch_d: float  # 0-100
    stoch_signal: str  # "OVERBOUGHT", "OVERSOLD", "NEUTRAL"


@dataclass
class VolatilityData:
    """Volatility indicators (Murphy, Chapter 11)"""
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float  # 0-1, where price is within bands
    atr: float  # Average True Range
    volatility_level: str  # "HIGH", "MEDIUM", "LOW"


class TechnicalIndicators:
    """
    Technical Indicators Calculator
    
    Implements indicators from John Murphy's "Technical Analysis of the Financial Markets":
    - Moving Averages (SMA, EMA) for trend identification (Chapter 9)
    - RSI for overbought/oversold conditions (Chapter 10)
    - MACD for trend confirmation (Chapter 10)
    - Stochastic Oscillator for momentum (Chapter 10)
    - Bollinger Bands for volatility (Chapter 11)
    - ATR for risk management (Chapter 11)
    """
    
    def __init__(self):
        self.use_talib = TALIB_AVAILABLE
        
    def calculate_trend(
        self, 
        df: pd.DataFrame,
        close_col: str = 'close'
    ) -> TrendData:
        """
        Calculate trend using Moving Averages
        
        Murphy's principle: "The trend is your friend"
        - 200-day MA: Primary trend identifier
        - 50-day MA: Intermediate trend
        - 20-day EMA: Short-term trend
        
        Args:
            df: DataFrame with OHLCV data
            close_col: Column name for close prices
            
        Returns:
            TrendData with trend analysis
        """
        try:
            close = df[close_col].values
            
            # Calculate Moving Averages (Murphy, Chapter 9)
            if self.use_talib:
                sma_200 = talib.SMA(close, timeperiod=200)
                sma_50 = talib.SMA(close, timeperiod=50)
                ema_20 = talib.EMA(close, timeperiod=20)
            else:
                sma_200 = df[close_col].rolling(window=200).mean().values
                sma_50 = df[close_col].rolling(window=50).mean().values
                ema_20 = df[close_col].ewm(span=20).mean().values
            
            current_price = close[-1]
            current_sma_200 = sma_200[-1] if not np.isnan(sma_200[-1]) else current_price
            current_sma_50 = sma_50[-1] if not np.isnan(sma_50[-1]) else current_price
            current_ema_20 = ema_20[-1] if not np.isnan(ema_20[-1]) else current_price
            
            # Price position relative to 200-day MA (primary trend identifier)
            price_vs_ma200 = ((current_price - current_sma_200) / current_sma_200) * 100
            
            # Golden Cross / Death Cross detection (Murphy, Chapter 9)
            golden_cross = False
            death_cross = False
            
            if len(sma_50) >= 2 and len(sma_200) >= 2:
                prev_50 = sma_50[-2]
                prev_200 = sma_200[-2]
                
                # Golden Cross: 50 SMA crosses above 200 SMA (bullish signal)
                if prev_50 <= prev_200 and current_sma_50 > current_sma_200:
                    golden_cross = True
                    logger.info("📈 Golden Cross detected - Strong bullish signal!")
                
                # Death Cross: 50 SMA crosses below 200 SMA (bearish signal)
                if prev_50 >= prev_200 and current_sma_50 < current_sma_200:
                    death_cross = True
                    logger.info("📉 Death Cross detected - Strong bearish signal!")
            
            # Determine trend direction
            if current_price > current_sma_200 and current_sma_50 > current_sma_200:
                trend = "BULLISH"
                strength = min(100, 50 + abs(price_vs_ma200) * 10)
            elif current_price < current_sma_200 and current_sma_50 < current_sma_200:
                trend = "BEARISH"
                strength = min(100, 50 + abs(price_vs_ma200) * 10)
            else:
                trend = "NEUTRAL"
                strength = 30 + abs(price_vs_ma200) * 5
            
            return TrendData(
                trend=trend,
                strength=strength,
                sma_200=current_sma_200,
                sma_50=current_sma_50,
                ema_20=current_ema_20,
                price_vs_ma200=price_vs_ma200,
                golden_cross=golden_cross,
                death_cross=death_cross
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return self._get_default_trend_data()
    
    def calculate_momentum(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close'
    ) -> MomentumData:
        """
        Calculate momentum indicators
        
        Murphy's momentum indicators (Chapter 10):
        - RSI: Oversold <30, Overbought >70
        - MACD: Trend following indicator
        - Stochastic: Momentum oscillator
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            MomentumData with momentum analysis
        """
        try:
            close = df[close_col].values
            high = df[high_col].values
            low = df[low_col].values
            
            # RSI - Relative Strength Index (Murphy, Chapter 10)
            # Oversold: RSI < 30 (potential buy signal)
            # Overbought: RSI > 70 (potential sell signal)
            if self.use_talib:
                rsi = talib.RSI(close, timeperiod=14)
            else:
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).values
            
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            
            # RSI signal interpretation
            if current_rsi > 70:
                rsi_signal = "OVERBOUGHT"
            elif current_rsi < 30:
                rsi_signal = "OVERSOLD"
            else:
                rsi_signal = "NEUTRAL"
            
            # MACD - Moving Average Convergence Divergence (Murphy, Chapter 10)
            # Standard settings: 12, 26, 9
            if self.use_talib:
                macd, macd_signal, macd_hist = talib.MACD(
                    close, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
            else:
                ema_12 = pd.Series(close).ewm(span=12).mean()
                ema_26 = pd.Series(close).ewm(span=26).mean()
                macd = (ema_12 - ema_26).values
                macd_signal = pd.Series(macd).ewm(span=9).mean().values
                macd_hist = macd - macd_signal
            
            current_macd = macd[-1] if not np.isnan(macd[-1]) else 0
            current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            current_macd_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # MACD trend interpretation
            if current_macd > current_macd_signal and current_macd_hist > 0:
                macd_trend = "BULLISH"
            elif current_macd < current_macd_signal and current_macd_hist < 0:
                macd_trend = "BEARISH"
            else:
                macd_trend = "NEUTRAL"
            
            # Stochastic Oscillator (Murphy, Chapter 10)
            # %K > 80: Overbought, %K < 20: Oversold
            if self.use_talib:
                stoch_k, stoch_d = talib.STOCH(
                    high, low, close,
                    fastk_period=14,
                    slowk_period=3,
                    slowd_period=3
                )
            else:
                # Calculate %K
                low_14 = pd.Series(low).rolling(window=14).min()
                high_14 = pd.Series(high).rolling(window=14).max()
                stoch_k = ((pd.Series(close) - low_14) / (high_14 - low_14) * 100).values
                stoch_d = pd.Series(stoch_k).rolling(window=3).mean().values
            
            current_stoch_k = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
            current_stoch_d = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
            
            # Stochastic signal interpretation
            if current_stoch_k > 80:
                stoch_signal = "OVERBOUGHT"
            elif current_stoch_k < 20:
                stoch_signal = "OVERSOLD"
            else:
                stoch_signal = "NEUTRAL"
            
            return MomentumData(
                rsi=current_rsi,
                rsi_signal=rsi_signal,
                macd=current_macd,
                macd_signal=current_macd_signal,
                macd_histogram=current_macd_hist,
                macd_trend=macd_trend,
                stoch_k=current_stoch_k,
                stoch_d=current_stoch_d,
                stoch_signal=stoch_signal
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return self._get_default_momentum_data()
    
    def calculate_volatility(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close'
    ) -> VolatilityData:
        """
        Calculate volatility indicators
        
        Murphy's volatility measures (Chapter 11):
        - Bollinger Bands: 2 standard deviations
        - ATR: Average True Range for stop-loss calculation
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            VolatilityData with volatility analysis
        """
        try:
            close = df[close_col].values
            high = df[high_col].values
            low = df[low_col].values
            
            # Bollinger Bands (Murphy, Chapter 11)
            # Standard: 20-period SMA with 2 standard deviations
            if self.use_talib:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2
                )
            else:
                bb_middle = pd.Series(close).rolling(window=20).mean().values
                std = pd.Series(close).rolling(window=20).std().values
                bb_upper = bb_middle + (2 * std)
                bb_lower = bb_middle - (2 * std)
            
            current_price = close[-1]
            current_bb_upper = bb_upper[-1] if not np.isnan(bb_upper[-1]) else current_price
            current_bb_middle = bb_middle[-1] if not np.isnan(bb_middle[-1]) else current_price
            current_bb_lower = bb_lower[-1] if not np.isnan(bb_lower[-1]) else current_price
            
            # Bollinger Band width (volatility measure)
            bb_width = ((current_bb_upper - current_bb_lower) / current_bb_middle) * 100
            
            # Price position within bands (0 = at lower band, 1 = at upper band)
            if current_bb_upper != current_bb_lower:
                bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            else:
                bb_position = 0.5
            
            # ATR - Average True Range (Murphy, Chapter 11)
            # Used for stop-loss placement
            if self.use_talib:
                atr = talib.ATR(high, low, close, timeperiod=14)
            else:
                # True Range calculation
                tr1 = high - low
                tr2 = np.abs(high - np.roll(close, 1))
                tr3 = np.abs(low - np.roll(close, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = pd.Series(tr).rolling(window=14).mean().values
            
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0
            
            # Determine volatility level based on BB width
            if bb_width > 5:
                volatility_level = "HIGH"
            elif bb_width > 2:
                volatility_level = "MEDIUM"
            else:
                volatility_level = "LOW"
            
            return VolatilityData(
                bb_upper=current_bb_upper,
                bb_middle=current_bb_middle,
                bb_lower=current_bb_lower,
                bb_width=bb_width,
                bb_position=bb_position,
                atr=current_atr,
                volatility_level=volatility_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return self._get_default_volatility_data()
    
    def _get_default_trend_data(self) -> TrendData:
        """Return default trend data on error"""
        return TrendData(
            trend="NEUTRAL",
            strength=0,
            sma_200=0,
            sma_50=0,
            ema_20=0,
            price_vs_ma200=0,
            golden_cross=False,
            death_cross=False
        )
    
    def _get_default_momentum_data(self) -> MomentumData:
        """Return default momentum data on error"""
        return MomentumData(
            rsi=50,
            rsi_signal="NEUTRAL",
            macd=0,
            macd_signal=0,
            macd_histogram=0,
            macd_trend="NEUTRAL",
            stoch_k=50,
            stoch_d=50,
            stoch_signal="NEUTRAL"
        )
    
    def _get_default_volatility_data(self) -> VolatilityData:
        """Return default volatility data on error"""
        return VolatilityData(
            bb_upper=0,
            bb_middle=0,
            bb_lower=0,
            bb_width=0,
            bb_position=0.5,
            atr=0,
            volatility_level="UNKNOWN"
        )