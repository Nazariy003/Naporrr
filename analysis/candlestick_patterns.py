"""
Candlestick Pattern Recognition Module
Based on:
- Steve Nison's "Japanese Candlestick Charting Techniques"
- Steven Bigalow's "High Profit Candlestick Patterns"

Implements high-reliability candlestick patterns (60-70% success rate)
with volume confirmation and indicator combinations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from utils.logger import logger


@dataclass
class CandlestickPatternData:
    """Candlestick pattern data"""
    pattern_name: str
    detected: bool
    direction: str  # "BULLISH" or "BEARISH"
    confidence: float  # 0-100
    reliability: float  # Historical success rate (60-70% per Nison/Bigalow)
    location: str  # "SUPPORT", "RESISTANCE", "MIDDLE"
    volume_confirmed: bool
    needs_confirmation: bool  # Whether next candle should confirm


@dataclass
class HammerPattern(CandlestickPatternData):
    """
    Hammer Pattern (Nison, Chapter on Reversal Patterns)
    
    Reliability: 60-70% at support levels
    Formation: Small body, long lower shadow (2-3x body), little/no upper shadow
    Confirmation: Next candle closes above hammer high
    """
    body_size: float
    lower_shadow_ratio: float
    upper_shadow_ratio: float


@dataclass
class EngulfingPattern(CandlestickPatternData):
    """
    Engulfing Pattern (Nison, Chapter on Reversal Patterns)
    
    Reliability: 60-70% with volume confirmation
    Formation: Second candle completely engulfs first candle's body
    Volume: Should be higher on engulfing candle
    """
    prev_body_size: float
    curr_body_size: float
    engulfing_ratio: float


@dataclass
class MorningStarPattern(CandlestickPatternData):
    """
    Morning Star Pattern (Nison, Chapter on Major Reversal Patterns)
    
    Reliability: 65-75% at support (Nison)
    Formation: 
    - Day 1: Long bearish candle
    - Day 2: Small body with gap down (star)
    - Day 3: Long bullish candle closing in Day 1's body
    """
    first_candle_size: float
    star_size: float
    third_candle_size: float
    gap_size: float


@dataclass
class DojiPattern(CandlestickPatternData):
    """
    Doji Pattern (Nison, Chapter on Indecision Patterns)
    
    Reliability: 55-65% at extremes (reversal signal)
    Formation: Open and close are very close (small/no body)
    Context: Signals indecision, potential reversal at trend extremes
    """
    body_ratio: float  # Body size / total range
    doji_type: str  # "GRAVESTONE", "DRAGONFLY", "LONG_LEGGED", "STANDARD"


class CandlestickPatternDetector:
    """
    Candlestick Pattern Detection Engine
    
    Implements pattern recognition from Nison and Bigalow's research.
    All patterns include volume confirmation for higher reliability.
    """
    
    def __init__(self):
        self.min_body_ratio = 0.1  # Minimum body size as % of range
        self.doji_threshold = 0.05  # Max body ratio for doji
    
    def detect_hammer(
        self,
        df: pd.DataFrame,
        min_shadow_ratio: float = 2.0
    ) -> Optional[HammerPattern]:
        """
        Detect Hammer pattern
        
        Nison's criteria:
        - Small body (color doesn't matter much)
        - Long lower shadow (at least 2x body size)
        - Little or no upper shadow
        - Appears after downtrend (bullish reversal)
        
        Bigalow's addition:
        - Volume should be above average
        - RSI should be oversold (<30)
        
        Args:
            df: DataFrame with OHLCV data
            min_shadow_ratio: Minimum lower shadow / body ratio
            
        Returns:
            HammerPattern if detected, None otherwise
        """
        try:
            if len(df) < 5:
                return None
            
            # Get current candle
            current = df.iloc[-1]
            prev_candles = df.iloc[-5:-1]
            
            open_price = current['open']
            close_price = current['close']
            high_price = current['high']
            low_price = current['low']
            
            # Calculate body and shadows
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            
            lower_shadow = body_bottom - low_price
            upper_shadow = high_price - body_top
            
            # Nison's criteria
            # 1. Small body (less than 30% of range)
            body_ratio = body_size / total_range
            if body_ratio > 0.3:
                return None
            
            # 2. Long lower shadow (at least 2x body size)
            if body_size == 0:
                lower_shadow_ratio = 100  # Very small body
            else:
                lower_shadow_ratio = lower_shadow / body_size
            
            if lower_shadow_ratio < min_shadow_ratio:
                return None
            
            # 3. Little or no upper shadow (less than body size)
            upper_shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
            if upper_shadow > body_size:
                return None
            
            # 4. Should appear after downtrend
            recent_trend = self._detect_recent_trend(prev_candles)
            if recent_trend != "BEARISH":
                return None  # Not valid at top or in neutral trend
            
            # Volume confirmation (Bigalow)
            avg_volume = prev_candles['volume'].mean()
            volume_confirmed = current['volume'] > avg_volume * 1.2
            
            # Location (support vs middle)
            location = self._determine_location(df, current['close'])
            
            # Confidence calculation
            confidence = 60  # Base reliability (Nison)
            if volume_confirmed:
                confidence += 10
            if location == "SUPPORT":
                confidence += 10  # Higher reliability at support
            if lower_shadow_ratio >= 3:
                confidence += 5  # Very long shadow
            
            logger.info(
                f"🔨 Hammer detected: "
                f"body_ratio={body_ratio:.2f}, "
                f"lower_shadow_ratio={lower_shadow_ratio:.1f}x, "
                f"volume_confirmed={volume_confirmed}, "
                f"confidence={confidence}%"
            )
            
            return HammerPattern(
                pattern_name="HAMMER",
                detected=True,
                direction="BULLISH",
                confidence=confidence,
                reliability=65,  # Nison's average at support
                location=location,
                volume_confirmed=volume_confirmed,
                needs_confirmation=True,  # Next candle should close above hammer high
                body_size=body_size,
                lower_shadow_ratio=lower_shadow_ratio,
                upper_shadow_ratio=upper_shadow_ratio
            )
            
        except Exception as e:
            logger.error(f"Error detecting hammer: {e}")
            return None
    
    def detect_engulfing(
        self,
        df: pd.DataFrame
    ) -> Optional[EngulfingPattern]:
        """
        Detect Bullish/Bearish Engulfing pattern
        
        Nison's criteria:
        - Market in downtrend (for bullish) or uptrend (for bearish)
        - First candle is in direction of trend
        - Second candle's body completely engulfs first candle's body
        
        Bigalow's addition:
        - Volume on engulfing candle should be significantly higher
        - Works best near support/resistance
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            EngulfingPattern if detected, None otherwise
        """
        try:
            if len(df) < 3:
                return None
            
            # Get last two candles
            prev = df.iloc[-2]
            current = df.iloc[-1]
            earlier_candles = df.iloc[-7:-2]
            
            # Calculate bodies
            prev_body_top = max(prev['open'], prev['close'])
            prev_body_bottom = min(prev['open'], prev['close'])
            prev_body_size = abs(prev['close'] - prev['open'])
            prev_is_bullish = prev['close'] > prev['open']
            
            curr_body_top = max(current['open'], current['close'])
            curr_body_bottom = min(current['open'], current['close'])
            curr_body_size = abs(current['close'] - current['open'])
            curr_is_bullish = current['close'] > current['open']
            
            if prev_body_size == 0 or curr_body_size == 0:
                return None
            
            # Check if engulfing
            is_engulfing = (
                curr_body_top > prev_body_top and
                curr_body_bottom < prev_body_bottom
            )
            
            if not is_engulfing:
                return None
            
            # Determine pattern type
            if not prev_is_bullish and curr_is_bullish:
                # Bullish engulfing
                direction = "BULLISH"
                trend_required = "BEARISH"
                pattern_name = "BULLISH_ENGULFING"
            elif prev_is_bullish and not curr_is_bullish:
                # Bearish engulfing
                direction = "BEARISH"
                trend_required = "BULLISH"
                pattern_name = "BEARISH_ENGULFING"
            else:
                return None  # Both same color, not valid engulfing
            
            # Check if in appropriate trend
            recent_trend = self._detect_recent_trend(earlier_candles)
            if recent_trend != trend_required:
                return None
            
            # Volume confirmation (Bigalow: engulfing candle should have high volume)
            avg_volume = earlier_candles['volume'].mean()
            volume_confirmed = current['volume'] > avg_volume * 1.5
            
            # Engulfing ratio (how much bigger)
            engulfing_ratio = curr_body_size / prev_body_size
            
            # Location
            location = self._determine_location(df, current['close'])
            
            # Confidence
            confidence = 65  # Base reliability (Nison)
            if volume_confirmed:
                confidence += 10
            if engulfing_ratio >= 2:
                confidence += 5  # Larger engulfing more reliable
            if location in ["SUPPORT", "RESISTANCE"]:
                confidence += 10
            
            logger.info(
                f"🔄 {pattern_name} detected: "
                f"engulfing_ratio={engulfing_ratio:.1f}x, "
                f"volume_confirmed={volume_confirmed}, "
                f"confidence={confidence}%"
            )
            
            return EngulfingPattern(
                pattern_name=pattern_name,
                detected=True,
                direction=direction,
                confidence=confidence,
                reliability=68,  # Nison's average with volume
                location=location,
                volume_confirmed=volume_confirmed,
                needs_confirmation=False,  # Strong pattern, doesn't need confirmation
                prev_body_size=prev_body_size,
                curr_body_size=curr_body_size,
                engulfing_ratio=engulfing_ratio
            )
            
        except Exception as e:
            logger.error(f"Error detecting engulfing: {e}")
            return None
    
    def detect_morning_star(
        self,
        df: pd.DataFrame
    ) -> Optional[MorningStarPattern]:
        """
        Detect Morning Star pattern (bullish reversal)
        
        Nison's criteria:
        - Day 1: Long bearish candle in downtrend
        - Day 2: Small body (star) gaps down
        - Day 3: Long bullish candle closing well into Day 1's body
        
        Evening Star is opposite (bearish reversal)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            MorningStarPattern if detected, None otherwise
        """
        try:
            if len(df) < 5:
                return None
            
            # Get three candles
            first = df.iloc[-3]
            star = df.iloc[-2]
            third = df.iloc[-1]
            earlier = df.iloc[-7:-3]
            
            # First candle should be long bearish
            first_body = abs(first['close'] - first['open'])
            first_range = first['high'] - first['low']
            first_is_bearish = first['close'] < first['open']
            
            if not first_is_bearish or first_body < first_range * 0.6:
                return None
            
            # Star should have small body
            star_body = abs(star['close'] - star['open'])
            star_range = star['high'] - star['low']
            
            if star_body > first_body * 0.3:
                return None  # Star too big
            
            # Gap down (star's high below first's close)
            gap_size = first['close'] - star['high']
            if gap_size < 0:
                return None  # No gap
            
            # Third candle should be long bullish
            third_body = abs(third['close'] - third['open'])
            third_range = third['high'] - third['low']
            third_is_bullish = third['close'] > third['open']
            
            if not third_is_bullish or third_body < third_range * 0.6:
                return None
            
            # Third should close well into first's body (at least 50%)
            first_midpoint = (first['open'] + first['close']) / 2
            if third['close'] < first_midpoint:
                return None
            
            # Should appear after downtrend
            recent_trend = self._detect_recent_trend(earlier)
            if recent_trend != "BEARISH":
                return None
            
            # Volume confirmation (volume increases on third candle)
            avg_volume = earlier['volume'].mean()
            volume_confirmed = third['volume'] > avg_volume * 1.3
            
            # Location
            location = self._determine_location(df, third['close'])
            
            # Confidence
            confidence = 70  # High reliability (Nison)
            if volume_confirmed:
                confidence += 10
            if location == "SUPPORT":
                confidence += 10
            if gap_size > first_body * 0.1:
                confidence += 5  # Larger gap more reliable
            
            logger.info(
                f"⭐ Morning Star detected: "
                f"gap={gap_size:.2f}, "
                f"volume_confirmed={volume_confirmed}, "
                f"confidence={confidence}%"
            )
            
            return MorningStarPattern(
                pattern_name="MORNING_STAR",
                detected=True,
                direction="BULLISH",
                confidence=confidence,
                reliability=72,  # Nison's average
                location=location,
                volume_confirmed=volume_confirmed,
                needs_confirmation=False,
                first_candle_size=first_body,
                star_size=star_body,
                third_candle_size=third_body,
                gap_size=gap_size
            )
            
        except Exception as e:
            logger.error(f"Error detecting morning star: {e}")
            return None
    
    def detect_doji(
        self,
        df: pd.DataFrame
    ) -> Optional[DojiPattern]:
        """
        Detect Doji pattern
        
        Nison's criteria:
        - Open and close are very close (small body)
        - Signals indecision
        - Reversal signal at trend extremes
        
        Types:
        - Gravestone: Long upper shadow, no lower shadow (bearish at top)
        - Dragonfly: Long lower shadow, no upper shadow (bullish at bottom)
        - Long-legged: Long shadows both sides
        - Standard: Small shadows
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DojiPattern if detected, None otherwise
        """
        try:
            if len(df) < 3:
                return None
            
            current = df.iloc[-1]
            prev_candles = df.iloc[-5:-1]
            
            open_price = current['open']
            close_price = current['close']
            high_price = current['high']
            low_price = current['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            # Check if doji (very small body)
            body_ratio = body_size / total_range
            if body_ratio > self.doji_threshold:
                return None
            
            # Determine doji type
            body_mid = (open_price + close_price) / 2
            upper_shadow = high_price - body_mid
            lower_shadow = body_mid - low_price
            
            if upper_shadow > total_range * 0.6 and lower_shadow < total_range * 0.1:
                doji_type = "GRAVESTONE"
                direction = "BEARISH"
            elif lower_shadow > total_range * 0.6 and upper_shadow < total_range * 0.1:
                doji_type = "DRAGONFLY"
                direction = "BULLISH"
            elif upper_shadow > total_range * 0.3 and lower_shadow > total_range * 0.3:
                doji_type = "LONG_LEGGED"
                direction = "NEUTRAL"
            else:
                doji_type = "STANDARD"
                direction = "NEUTRAL"
            
            # Check trend context
            recent_trend = self._detect_recent_trend(prev_candles)
            
            # Doji more significant at extremes
            location = self._determine_location(df, current['close'])
            
            # Only signal if at trend extreme
            if recent_trend == "BULLISH" and location == "RESISTANCE":
                direction = "BEARISH"  # Reversal signal
                confidence = 60
            elif recent_trend == "BEARISH" and location == "SUPPORT":
                direction = "BULLISH"  # Reversal signal
                confidence = 60
            else:
                confidence = 40  # Low confidence in middle of range
            
            # Volume
            avg_volume = prev_candles['volume'].mean()
            volume_confirmed = current['volume'] > avg_volume
            
            if volume_confirmed:
                confidence += 5
            
            logger.info(
                f"🎯 {doji_type} Doji detected: "
                f"body_ratio={body_ratio:.3f}, "
                f"direction={direction}, "
                f"confidence={confidence}%"
            )
            
            return DojiPattern(
                pattern_name=f"{doji_type}_DOJI",
                detected=True,
                direction=direction,
                confidence=confidence,
                reliability=58,  # Moderate reliability (Nison)
                location=location,
                volume_confirmed=volume_confirmed,
                needs_confirmation=True,  # Needs next candle confirmation
                body_ratio=body_ratio,
                doji_type=doji_type
            )
            
        except Exception as e:
            logger.error(f"Error detecting doji: {e}")
            return None
    
    def _detect_recent_trend(self, df: pd.DataFrame) -> str:
        """
        Detect recent trend direction
        
        Returns: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if len(df) < 3:
            return "NEUTRAL"
        
        # Simple trend: compare recent closes
        recent_closes = df['close'].values
        
        # Calculate trend
        slope = (recent_closes[-1] - recent_closes[0]) / len(recent_closes)
        avg_price = recent_closes.mean()
        
        slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0
        
        if slope_pct > 0.5:
            return "BULLISH"
        elif slope_pct < -0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _determine_location(self, df: pd.DataFrame, price: float) -> str:
        """
        Determine if price is at support, resistance, or middle
        
        Returns: "SUPPORT", "RESISTANCE", or "MIDDLE"
        """
        if len(df) < 20:
            return "MIDDLE"
        
        recent = df.iloc[-50:] if len(df) >= 50 else df
        
        high_20 = recent['high'].max()
        low_20 = recent['low'].min()
        range_20 = high_20 - low_20
        
        if range_20 == 0:
            return "MIDDLE"
        
        # Position in range
        position = (price - low_20) / range_20
        
        if position < 0.2:
            return "SUPPORT"
        elif position > 0.8:
            return "RESISTANCE"
        else:
            return "MIDDLE"
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect all candlestick patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all detected patterns
        """
        patterns = {
            'hammer': None,
            'engulfing': None,
            'morning_star': None,
            'doji': None,
            'any_pattern_detected': False,
            'bullish_patterns': [],
            'bearish_patterns': []
        }
        
        try:
            # Detect each pattern
            patterns['hammer'] = self.detect_hammer(df)
            patterns['engulfing'] = self.detect_engulfing(df)
            patterns['morning_star'] = self.detect_morning_star(df)
            patterns['doji'] = self.detect_doji(df)
            
            # Collect bullish and bearish patterns
            for key, pattern in patterns.items():
                if pattern and hasattr(pattern, 'direction'):
                    if pattern.direction == "BULLISH":
                        patterns['bullish_patterns'].append(pattern.pattern_name)
                    elif pattern.direction == "BEARISH":
                        patterns['bearish_patterns'].append(pattern.pattern_name)
            
            # Check if any pattern detected
            patterns['any_pattern_detected'] = len(patterns['bullish_patterns']) > 0 or len(patterns['bearish_patterns']) > 0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting all candlestick patterns: {e}")
            return patterns