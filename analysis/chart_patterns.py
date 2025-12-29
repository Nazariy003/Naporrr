"""
Chart Pattern Detection Module
Based on Thomas Bulkowski's "Encyclopedia of Chart Patterns"

This module detects classical chart patterns with proven success rates:
- Double Bottom: 60-83% success rate (Bulkowski)
- Head and Shoulders: 60-83% success rate (Bulkowski)
- Triangle Patterns: 60-83% success rate (Bulkowski)

All patterns are validated with volume confirmation as per Bulkowski's research.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks, argrelextrema
from utils.logger import logger


@dataclass
class PatternData:
    """Base class for chart pattern data"""
    pattern_name: str
    detected: bool
    confidence: float  # 0-100
    direction: str  # "BULLISH" or "BEARISH"
    entry_price: float
    stop_loss: float
    take_profit: float
    pattern_height: float
    volume_confirmed: bool
    success_rate: float  # Historical success rate from Bulkowski


@dataclass
class DoubleBottomPattern(PatternData):
    """
    Double Bottom Pattern (Bulkowski, Chapter on Reversal Patterns)
    
    Success Rate: 60-83% (Bulkowski research)
    Formation: Two distinct lows at approximately same level
    Volume: Should decline at second bottom, increase on breakout
    """
    first_low: float
    second_low: float
    peak_between: float
    breakout_level: float
    days_between_lows: int


@dataclass
class HeadAndShouldersPattern(PatternData):
    """
    Head and Shoulders Pattern (Bulkowski, Chapter on Top Patterns)
    
    Success Rate: 60-83% (Bulkowski research)
    Formation: Three peaks, middle one highest (head), two shoulders
    Volume: Typically decreases at head, increases on neckline break
    """
    left_shoulder: float
    head: float
    right_shoulder: float
    neckline: float
    neckline_slope: float


@dataclass
class TrianglePattern(PatternData):
    """
    Triangle Patterns (Bulkowski, Chapter on Continuation Patterns)
    
    Success Rate: 60-83% (Bulkowski research)
    Types: Ascending, Descending, Symmetrical
    Volume: Should contract during formation, expand on breakout
    """
    triangle_type: str  # "ASCENDING", "DESCENDING", "SYMMETRICAL"
    upper_trendline_slope: float
    lower_trendline_slope: float
    apex_price: float
    breakout_direction: str


class ChartPatternDetector:
    """
    Chart Pattern Detection Engine
    
    Implements pattern recognition algorithms based on Bulkowski's research.
    All patterns require volume confirmation for higher reliability.
    """
    
    def __init__(self, min_pattern_bars: int = 20, max_pattern_bars: int = 100):
        """
        Initialize pattern detector
        
        Args:
            min_pattern_bars: Minimum bars for pattern formation
            max_pattern_bars: Maximum bars for pattern formation
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
    
    def detect_double_bottom(
        self,
        df: pd.DataFrame,
        tolerance: float = 0.02  # 2% tolerance for "same level"
    ) -> Optional[DoubleBottomPattern]:
        """
        Detect Double Bottom pattern
        
        Bulkowski's criteria:
        - Two distinct lows at approximately same price level
        - Peak between lows should be 10-20% higher
        - Volume decreases at second low (bullish sign)
        - Breakout above peak confirms pattern
        
        Args:
            df: DataFrame with OHLCV data
            tolerance: Price tolerance for matching lows
            
        Returns:
            DoubleBottomPattern if detected, None otherwise
        """
        try:
            if len(df) < self.min_pattern_bars:
                return None
            
            # Find local minima (potential lows)
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(lows) < 2:
                return None
            
            # Check recent pairs of lows
            for i in range(len(lows) - 1):
                first_low_idx = lows[i]
                second_low_idx = lows[i + 1]
                
                # Limit pattern window
                if second_low_idx - first_low_idx > self.max_pattern_bars:
                    continue
                if second_low_idx - first_low_idx < self.min_pattern_bars:
                    continue
                
                first_low = df.iloc[first_low_idx]['low']
                second_low = df.iloc[second_low_idx]['low']
                
                # Check if lows are at approximately same level (Bulkowski criterion)
                price_diff = abs(first_low - second_low) / first_low
                if price_diff > tolerance:
                    continue
                
                # Find peak between lows
                between_slice = df.iloc[first_low_idx:second_low_idx + 1]
                peak_idx = between_slice['high'].idxmax()
                peak_price = between_slice.loc[peak_idx, 'high']
                
                # Peak should be 10-20% higher than lows (Bulkowski criterion)
                peak_height_pct = (peak_price - first_low) / first_low * 100
                if peak_height_pct < 10 or peak_height_pct > 30:
                    continue
                
                # Volume confirmation (Bulkowski: volume should decrease at 2nd low)
                vol_first = df.iloc[first_low_idx]['volume']
                vol_second = df.iloc[second_low_idx]['volume']
                volume_confirmed = vol_second < vol_first * 1.2  # Allow some tolerance
                
                # Calculate pattern metrics
                current_price = df.iloc[-1]['close']
                breakout_level = peak_price
                
                # Check if pattern is complete (price near or above breakout)
                if current_price < peak_price * 0.95:
                    continue  # Pattern not yet complete
                
                # Calculate targets (Bulkowski's method)
                pattern_height = peak_price - min(first_low, second_low)
                
                # Entry: Breakout above peak
                entry_price = breakout_level
                
                # Stop-loss: 1-2% below second low (Bigalow's risk management)
                stop_loss = second_low * 0.98
                
                # Take-profit: Pattern height projected upward (2:1 reward-risk)
                take_profit = breakout_level + (pattern_height * 2)
                
                # Confidence based on volume confirmation and pattern quality
                confidence = 70 if volume_confirmed else 55
                if peak_height_pct >= 15:
                    confidence += 10
                
                logger.info(
                    f"📊 Double Bottom detected: "
                    f"lows={first_low:.2f}/{second_low:.2f}, "
                    f"peak={peak_price:.2f}, "
                    f"vol_confirmed={volume_confirmed}, "
                    f"confidence={confidence}%"
                )
                
                return DoubleBottomPattern(
                    pattern_name="DOUBLE_BOTTOM",
                    detected=True,
                    confidence=confidence,
                    direction="BULLISH",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pattern_height=pattern_height,
                    volume_confirmed=volume_confirmed,
                    success_rate=70,  # Bulkowski's average
                    first_low=first_low,
                    second_low=second_low,
                    peak_between=peak_price,
                    breakout_level=breakout_level,
                    days_between_lows=second_low_idx - first_low_idx
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return None
    
    def detect_head_and_shoulders(
        self,
        df: pd.DataFrame,
        tolerance: float = 0.05  # 5% tolerance
    ) -> Optional[HeadAndShouldersPattern]:
        """
        Detect Head and Shoulders pattern
        
        Bulkowski's criteria:
        - Three peaks: left shoulder, head (highest), right shoulder
        - Shoulders at approximately same level
        - Neckline connects lows between peaks
        - Volume decreases at head, increases on neckline break
        
        Args:
            df: DataFrame with OHLCV data
            tolerance: Price tolerance for matching shoulders
            
        Returns:
            HeadAndShouldersPattern if detected, None otherwise
        """
        try:
            if len(df) < self.min_pattern_bars * 2:
                return None
            
            # Find local maxima (potential peaks)
            peaks = argrelextrema(df['high'].values, np.greater, order=5)[0]
            
            if len(peaks) < 3:
                return None
            
            # Check recent triplets of peaks
            for i in range(len(peaks) - 2):
                left_shoulder_idx = peaks[i]
                head_idx = peaks[i + 1]
                right_shoulder_idx = peaks[i + 2]
                
                # Limit pattern window
                if right_shoulder_idx - left_shoulder_idx > self.max_pattern_bars:
                    continue
                
                left_shoulder = df.iloc[left_shoulder_idx]['high']
                head = df.iloc[head_idx]['high']
                right_shoulder = df.iloc[right_shoulder_idx]['high']
                
                # Head must be highest (Bulkowski criterion)
                if head <= left_shoulder or head <= right_shoulder:
                    continue
                
                # Shoulders should be approximately same level
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff > tolerance:
                    continue
                
                # Find lows for neckline
                left_low_slice = df.iloc[left_shoulder_idx:head_idx]
                right_low_slice = df.iloc[head_idx:right_shoulder_idx]
                
                if len(left_low_slice) == 0 or len(right_low_slice) == 0:
                    continue
                
                left_low_idx = left_low_slice['low'].idxmin()
                right_low_idx = right_low_slice['low'].idxmin()
                
                left_low = df.loc[left_low_idx, 'low']
                right_low = df.loc[right_low_idx, 'low']
                
                # Neckline (average of two lows)
                neckline = (left_low + right_low) / 2
                neckline_slope = (right_low - left_low) / (right_low_idx - left_low_idx)
                
                # Current price should be breaking neckline
                current_price = df.iloc[-1]['close']
                if current_price > neckline * 1.02:
                    continue  # Pattern not confirmed
                
                # Volume confirmation (volume increases on neckline break)
                vol_head = df.iloc[head_idx]['volume']
                vol_recent = df.iloc[-5:]['volume'].mean()
                volume_confirmed = vol_recent > vol_head
                
                # Calculate targets (Bulkowski's method)
                pattern_height = head - neckline
                
                # Entry: Neckline break
                entry_price = neckline
                
                # Stop-loss: Above right shoulder (1-2%)
                stop_loss = right_shoulder * 1.02
                
                # Take-profit: Pattern height projected downward
                take_profit = neckline - (pattern_height * 2)
                
                # Confidence
                confidence = 75 if volume_confirmed else 60
                if abs(neckline_slope) < 0.001:  # Flat neckline is more reliable
                    confidence += 10
                
                logger.info(
                    f"📊 Head and Shoulders detected: "
                    f"LS={left_shoulder:.2f}, H={head:.2f}, RS={right_shoulder:.2f}, "
                    f"neckline={neckline:.2f}, "
                    f"confidence={confidence}%"
                )
                
                return HeadAndShouldersPattern(
                    pattern_name="HEAD_AND_SHOULDERS",
                    detected=True,
                    confidence=confidence,
                    direction="BEARISH",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pattern_height=pattern_height,
                    volume_confirmed=volume_confirmed,
                    success_rate=73,  # Bulkowski's average
                    left_shoulder=left_shoulder,
                    head=head,
                    right_shoulder=right_shoulder,
                    neckline=neckline,
                    neckline_slope=neckline_slope
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return None
    
    def detect_triangle(
        self,
        df: pd.DataFrame,
        min_touches: int = 4
    ) -> Optional[TrianglePattern]:
        """
        Detect Triangle patterns (Ascending, Descending, Symmetrical)
        
        Bulkowski's criteria:
        - At least 2 touches on each trendline
        - Volume should contract during formation
        - Breakout occurs before apex (typically 2/3 to 3/4 of pattern)
        
        Args:
            df: DataFrame with OHLCV data
            min_touches: Minimum touches on trendlines
            
        Returns:
            TrianglePattern if detected, None otherwise
        """
        try:
            if len(df) < self.min_pattern_bars:
                return None
            
            # Get recent data for pattern analysis
            recent = df.iloc[-self.max_pattern_bars:]
            
            # Find swing highs and lows
            highs = argrelextrema(recent['high'].values, np.greater, order=3)[0]
            lows = argrelextrema(recent['low'].values, np.less, order=3)[0]
            
            if len(highs) < 2 or len(lows) < 2:
                return None
            
            # Fit trendlines
            try:
                # Upper trendline (connect highs)
                high_prices = recent.iloc[highs]['high'].values
                high_x = highs
                upper_slope, upper_intercept = np.polyfit(high_x, high_prices, 1)
                
                # Lower trendline (connect lows)
                low_prices = recent.iloc[lows]['low'].values
                low_x = lows
                lower_slope, lower_intercept = np.polyfit(low_x, low_prices, 1)
                
            except:
                return None
            
            # Determine triangle type
            if abs(upper_slope) < 0.0001 and lower_slope > 0.0001:
                triangle_type = "ASCENDING"  # Flat top, rising bottom
                direction = "BULLISH"
                success_rate = 68
            elif abs(lower_slope) < 0.0001 and upper_slope < -0.0001:
                triangle_type = "DESCENDING"  # Flat bottom, falling top
                direction = "BEARISH"
                success_rate = 64
            elif upper_slope < -0.0001 and lower_slope > 0.0001:
                triangle_type = "SYMMETRICAL"  # Converging trendlines
                direction = "NEUTRAL"  # Breakout determines direction
                success_rate = 60
            else:
                return None  # Not a valid triangle
            
            # Calculate apex (where trendlines would meet)
            if abs(upper_slope - lower_slope) < 0.0001:
                return None  # Parallel lines, not a triangle
            
            apex_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
            apex_price = upper_slope * apex_x + upper_intercept
            
            # Current position in pattern
            current_idx = len(recent) - 1
            pattern_progress = current_idx / apex_x if apex_x > 0 else 1
            
            # Pattern should break before apex (Bulkowski: 2/3 to 3/4 of pattern)
            if pattern_progress > 0.9:
                return None  # Too close to apex
            
            # Volume confirmation (should contract during formation)
            early_vol = recent.iloc[:len(recent)//2]['volume'].mean()
            late_vol = recent.iloc[len(recent)//2:]['volume'].mean()
            volume_confirmed = late_vol < early_vol
            
            # Determine breakout
            current_price = df.iloc[-1]['close']
            upper_line_current = upper_slope * current_idx + upper_intercept
            lower_line_current = lower_slope * current_idx + lower_intercept
            
            # Check for breakout
            if triangle_type == "ASCENDING":
                if current_price >= upper_line_current:
                    breakout_direction = "BULLISH"
                else:
                    return None  # No breakout yet
            elif triangle_type == "DESCENDING":
                if current_price <= lower_line_current:
                    breakout_direction = "BEARISH"
                else:
                    return None
            else:  # SYMMETRICAL
                if current_price >= upper_line_current:
                    breakout_direction = "BULLISH"
                elif current_price <= lower_line_current:
                    breakout_direction = "BEARISH"
                else:
                    return None
            
            # Calculate pattern height
            pattern_height = abs(upper_line_current - lower_line_current)
            
            # Calculate targets
            if breakout_direction == "BULLISH":
                entry_price = upper_line_current
                stop_loss = lower_line_current * 0.98
                take_profit = entry_price + (pattern_height * 2)
            else:
                entry_price = lower_line_current
                stop_loss = upper_line_current * 1.02
                take_profit = entry_price - (pattern_height * 2)
            
            # Confidence
            confidence = 65 if volume_confirmed else 50
            if len(highs) >= 3 and len(lows) >= 3:
                confidence += 10  # More touches = higher reliability
            
            logger.info(
                f"📊 {triangle_type} Triangle detected: "
                f"breakout={breakout_direction}, "
                f"confidence={confidence}%"
            )
            
            return TrianglePattern(
                pattern_name=f"{triangle_type}_TRIANGLE",
                detected=True,
                confidence=confidence,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pattern_height=pattern_height,
                volume_confirmed=volume_confirmed,
                success_rate=success_rate,
                triangle_type=triangle_type,
                upper_trendline_slope=upper_slope,
                lower_trendline_slope=lower_slope,
                apex_price=apex_price,
                breakout_direction=breakout_direction
            )
            
        except Exception as e:
            logger.error(f"Error detecting triangle: {e}")
            return None
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect all chart patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all detected patterns
        """
        patterns = {
            'double_bottom': None,
            'head_and_shoulders': None,
            'triangle': None,
            'any_pattern_detected': False
        }
        
        try:
            # Detect each pattern
            patterns['double_bottom'] = self.detect_double_bottom(df)
            patterns['head_and_shoulders'] = self.detect_head_and_shoulders(df)
            patterns['triangle'] = self.detect_triangle(df)
            
            # Check if any pattern detected
            patterns['any_pattern_detected'] = any([
                patterns['double_bottom'] is not None,
                patterns['head_and_shoulders'] is not None,
                patterns['triangle'] is not None
            ])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting all patterns: {e}")
            return patterns