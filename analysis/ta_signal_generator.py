"""
Technical Analysis Signal Generator
Integrates chart patterns, candlestick patterns, and technical indicators
for comprehensive trading signal generation.

Based on:
- Thomas Bulkowski's "Encyclopedia of Chart Patterns"
- John Murphy's "Technical Analysis of the Financial Markets"
- Steve Nison's "Japanese Candlestick Charting Techniques"
- Steven Bigalow's "High Profit Candlestick Patterns"
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from utils.logger import logger
from config.settings import settings
from analysis.technical_indicators import TechnicalIndicators
from analysis.chart_patterns import ChartPatternDetector
from analysis.candlestick_patterns import CandlestickPatternDetector


@dataclass
class TradingSignal:
    """Complete trading signal with all analysis"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    strength: int  # 1-5
    confidence: float  # 0-100
    
    # Entry/Exit levels
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Pattern information
    chart_pattern: Optional[str]
    candlestick_pattern: Optional[str]
    
    # Indicator signals
    trend: str  # "BULLISH", "BEARISH", "NEUTRAL"
    rsi_signal: str
    macd_signal: str
    
    # Confirmation
    trend_confirmed: bool
    volume_confirmed: bool
    indicator_confirmed: bool
    
    # Risk metrics
    position_size_pct: float  # % of portfolio
    leverage_recommended: float  # 2-5x
    
    # Reason
    reason: str


class TechnicalAnalysisSignalGenerator:
    """
    Comprehensive Signal Generator
    
    Combines:
    1. Chart patterns (Bulkowski) - 60-83% success rate
    2. Candlestick patterns (Nison/Bigalow) - 60-70% reliability
    3. Technical indicators (Murphy) - Trend/momentum confirmation
    4. Risk management (Bigalow) - Position sizing, stop-loss, take-profit
    """
    
    def __init__(self):
        self.tech_indicators = TechnicalIndicators()
        self.chart_pattern_detector = ChartPatternDetector()
        self.candlestick_detector = CandlestickPatternDetector()
        
        # Murphy's trend identification: 200-day MA is primary
        self.primary_trend_period = 200
        
        # Bigalow's risk management defaults
        self.default_risk_per_trade = 0.015  # 1.5% per trade
        self.max_portfolio_risk = 0.08  # 8% max portfolio risk
        self.default_leverage = 3  # 3x leverage default for crypto
    
    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_portfolio_risk: float = 0.0
    ) -> TradingSignal:
        """
        Generate comprehensive trading signal
        
        Entry Logic (from problem statement):
        - BUY: Bullish patterns (double bottom + hammer/engulfing) + trend up + RSI < 30
        - SELL: Bearish patterns (head & shoulders + bearish engulfing) + trend down + RSI > 70
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            df: DataFrame with OHLCV data (must have columns: open, high, low, close, volume, timestamp)
            current_portfolio_risk: Current portfolio risk percentage
            
        Returns:
            TradingSignal with complete analysis
        """
        try:
            # Step 1: Trend Identification (Murphy - 200-day MA)
            trend_data = self.tech_indicators.calculate_trend(df)
            
            # Step 2: Momentum Indicators (Murphy)
            momentum_data = self.tech_indicators.calculate_momentum(df)
            
            # Step 3: Volatility Indicators (Murphy)
            volatility_data = self.tech_indicators.calculate_volatility(df)
            
            # Step 4: Chart Pattern Detection (Bulkowski)
            chart_patterns = self.chart_pattern_detector.detect_all_patterns(df)
            
            # Step 5: Candlestick Pattern Detection (Nison/Bigalow)
            candlestick_patterns = self.candlestick_detector.detect_all_patterns(df)
            
            # Step 6: Combine signals and determine action
            signal = self._combine_signals(
                symbol=symbol,
                df=df,
                trend_data=trend_data,
                momentum_data=momentum_data,
                volatility_data=volatility_data,
                chart_patterns=chart_patterns,
                candlestick_patterns=candlestick_patterns,
                current_portfolio_risk=current_portfolio_risk
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._create_hold_signal(symbol, "error")
    
    def _combine_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        trend_data: Any,
        momentum_data: Any,
        volatility_data: Any,
        chart_patterns: Dict,
        candlestick_patterns: Dict,
        current_portfolio_risk: float
    ) -> TradingSignal:
        """
        Combine all signals using Murphy's, Bulkowski's, and Bigalow's principles
        
        Entry Requirements:
        1. Trend confirmation (Murphy: 200-day MA)
        2. Pattern detection (Bulkowski: chart patterns OR Nison: candlestick patterns)
        3. Indicator confirmation (Murphy: RSI + MACD)
        4. Volume confirmation (Bulkowski/Nison: increased volume)
        """
        current_price = df.iloc[-1]['close']
        
        # Initialize signal components
        action = "HOLD"
        strength = 0
        confidence = 0
        chart_pattern_name = None
        candlestick_pattern_name = None
        reasons = []
        
        # Check for BULLISH setup
        bullish_score = 0
        bullish_reasons = []
        
        # 1. Bullish Trend (Murphy: price > 200 MA)
        if trend_data.trend == "BULLISH":
            bullish_score += 30
            bullish_reasons.append("trend_bullish_200ma")
        
        # 2. Bullish Chart Pattern (Bulkowski)
        if chart_patterns.get('double_bottom'):
            pattern = chart_patterns['double_bottom']
            if pattern.confidence >= 60:
                bullish_score += 25
                bullish_reasons.append(f"double_bottom_{pattern.confidence:.0f}%")
                chart_pattern_name = "DOUBLE_BOTTOM"
        
        if chart_patterns.get('triangle'):
            pattern = chart_patterns['triangle']
            if pattern.direction == "BULLISH" and pattern.confidence >= 60:
                bullish_score += 20
                bullish_reasons.append(f"{pattern.triangle_type}_triangle")
                if not chart_pattern_name:
                    chart_pattern_name = pattern.triangle_type + "_TRIANGLE"
        
        # 3. Bullish Candlestick Pattern (Nison/Bigalow)
        if candlestick_patterns.get('hammer'):
            pattern = candlestick_patterns['hammer']
            if pattern.confidence >= 60:
                bullish_score += 20
                bullish_reasons.append(f"hammer_{pattern.confidence:.0f}%")
                candlestick_pattern_name = "HAMMER"
        
        if candlestick_patterns.get('engulfing'):
            pattern = candlestick_patterns['engulfing']
            if pattern.direction == "BULLISH" and pattern.confidence >= 65:
                bullish_score += 25
                bullish_reasons.append(f"bullish_engulfing_{pattern.confidence:.0f}%")
                if not candlestick_pattern_name:
                    candlestick_pattern_name = "BULLISH_ENGULFING"
        
        if candlestick_patterns.get('morning_star'):
            pattern = candlestick_patterns['morning_star']
            if pattern.confidence >= 70:
                bullish_score += 30
                bullish_reasons.append(f"morning_star_{pattern.confidence:.0f}%")
                if not candlestick_pattern_name:
                    candlestick_pattern_name = "MORNING_STAR"
        
        # 4. RSI Oversold (Murphy: RSI < 30 = buy signal)
        if momentum_data.rsi < 30:
            bullish_score += 15
            bullish_reasons.append(f"rsi_oversold_{momentum_data.rsi:.0f}")
        elif momentum_data.rsi < 40:
            bullish_score += 8
            bullish_reasons.append(f"rsi_low_{momentum_data.rsi:.0f}")
        
        # 5. MACD Bullish (Murphy)
        if momentum_data.macd_trend == "BULLISH":
            bullish_score += 10
            bullish_reasons.append("macd_bullish")
        
        # Check for BEARISH setup
        bearish_score = 0
        bearish_reasons = []
        
        # 1. Bearish Trend (Murphy: price < 200 MA)
        if trend_data.trend == "BEARISH":
            bearish_score += 30
            bearish_reasons.append("trend_bearish_200ma")
        
        # 2. Bearish Chart Pattern (Bulkowski)
        if chart_patterns.get('head_and_shoulders'):
            pattern = chart_patterns['head_and_shoulders']
            if pattern.confidence >= 60:
                bearish_score += 30
                bearish_reasons.append(f"head_shoulders_{pattern.confidence:.0f}%")
                chart_pattern_name = "HEAD_AND_SHOULDERS"
        
        if chart_patterns.get('triangle'):
            pattern = chart_patterns['triangle']
            if pattern.direction == "BEARISH" and pattern.confidence >= 60:
                bearish_score += 20
                bearish_reasons.append(f"{pattern.triangle_type}_triangle")
                if not chart_pattern_name:
                    chart_pattern_name = pattern.triangle_type + "_TRIANGLE"
        
        # 3. Bearish Candlestick Pattern (Nison/Bigalow)
        if candlestick_patterns.get('engulfing'):
            pattern = candlestick_patterns['engulfing']
            if pattern.direction == "BEARISH" and pattern.confidence >= 65:
                bearish_score += 25
                bearish_reasons.append(f"bearish_engulfing_{pattern.confidence:.0f}%")
                if not candlestick_pattern_name:
                    candlestick_pattern_name = "BEARISH_ENGULFING"
        
        # 4. RSI Overbought (Murphy: RSI > 70 = sell signal)
        if momentum_data.rsi > 70:
            bearish_score += 15
            bearish_reasons.append(f"rsi_overbought_{momentum_data.rsi:.0f}")
        elif momentum_data.rsi > 60:
            bearish_score += 8
            bearish_reasons.append(f"rsi_high_{momentum_data.rsi:.0f}")
        
        # 5. MACD Bearish (Murphy)
        if momentum_data.macd_trend == "BEARISH":
            bearish_score += 10
            bearish_reasons.append("macd_bearish")
        
        # Determine action based on scores
        min_score_for_trade = settings.technical_analysis.min_score_for_trade  # From configuration
        
        if bullish_score >= min_score_for_trade and bullish_score > bearish_score:
            action = "BUY"
            confidence = min(100, bullish_score)
            reasons = bullish_reasons
            
            # Calculate strength (1-5)
            if bullish_score >= 90:
                strength = 5
            elif bullish_score >= 80:
                strength = 4
            elif bullish_score >= 70:
                strength = 3
            elif bullish_score >= 60:
                strength = 2
            else:
                strength = 1
                
        elif bearish_score >= min_score_for_trade and bearish_score > bullish_score:
            action = "SELL"
            confidence = min(100, bearish_score)
            reasons = bearish_reasons
            
            # Calculate strength (1-5)
            if bearish_score >= 90:
                strength = 5
            elif bearish_score >= 80:
                strength = 4
            elif bearish_score >= 70:
                strength = 3
            elif bearish_score >= 60:
                strength = 2
            else:
                strength = 1
        
        # Calculate entry/exit levels
        entry_price, stop_loss, take_profit, risk_reward = self._calculate_levels(
            action=action,
            current_price=current_price,
            chart_patterns=chart_patterns,
            candlestick_patterns=candlestick_patterns,
            volatility_data=volatility_data,
            trend_data=trend_data
        )
        
        # Calculate position size (Bigalow's risk management)
        position_size_pct = self._calculate_position_size(
            action=action,
            strength=strength,
            current_portfolio_risk=current_portfolio_risk,
            volatility_level=volatility_data.volatility_level
        )
        
        # Calculate leverage (2-5x per problem statement)
        leverage_recommended = self._calculate_leverage(
            strength=strength,
            volatility_level=volatility_data.volatility_level,
            trend_strength=trend_data.strength
        )
        
        # Confirmations
        trend_confirmed = self._check_trend_confirmation(trend_data, action)
        volume_confirmed = self._check_volume_confirmation(df, chart_patterns, candlestick_patterns)
        indicator_confirmed = self._check_indicator_confirmation(momentum_data, action)
        
        # Create signal
        reason = ", ".join(reasons) if reasons else "no_signal"
        
        logger.info(
            f"📊 Signal for {symbol}: {action}{strength} "
            f"(confidence={confidence:.0f}%, bullish_score={bullish_score}, bearish_score={bearish_score}) "
            f"Reasons: {reason}"
        )
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            chart_pattern=chart_pattern_name,
            candlestick_pattern=candlestick_pattern_name,
            trend=trend_data.trend,
            rsi_signal=momentum_data.rsi_signal,
            macd_signal=momentum_data.macd_trend,
            trend_confirmed=trend_confirmed,
            volume_confirmed=volume_confirmed,
            indicator_confirmed=indicator_confirmed,
            position_size_pct=position_size_pct,
            leverage_recommended=leverage_recommended,
            reason=reason
        )
    
    def _calculate_levels(
        self,
        action: str,
        current_price: float,
        chart_patterns: Dict,
        candlestick_patterns: Dict,
        volatility_data: Any,
        trend_data: Any
    ) -> Tuple[float, float, float, float]:
        """
        Calculate entry, stop-loss, and take-profit levels
        
        Bigalow's risk management:
        - Stop-loss: 1-2% below pattern low (for longs) or above pattern high (for shorts)
        - Take-profit: 2:1 reward-risk ratio based on pattern height
        """
        entry_price = current_price
        stop_loss = 0
        take_profit = 0
        risk_reward = 2.0  # Default 2:1
        
        if action == "HOLD":
            return entry_price, stop_loss, take_profit, 0
        
        # Use pattern-based levels if available
        pattern_found = False
        
        if action == "BUY":
            # Try chart patterns first
            if chart_patterns.get('double_bottom'):
                pattern = chart_patterns['double_bottom']
                entry_price = pattern.entry_price
                stop_loss = pattern.stop_loss
                take_profit = pattern.take_profit
                pattern_found = True
            elif chart_patterns.get('triangle') and chart_patterns['triangle'].direction == "BULLISH":
                pattern = chart_patterns['triangle']
                entry_price = pattern.entry_price
                stop_loss = pattern.stop_loss
                take_profit = pattern.take_profit
                pattern_found = True
            
            # Try candlestick patterns
            elif candlestick_patterns.get('hammer'):
                # Stop-loss 1-2% below hammer low (Bigalow)
                stop_loss = current_price * 0.98
                # Take-profit 2:1 reward-risk
                risk = current_price - stop_loss
                take_profit = current_price + (risk * 2)
                pattern_found = True
            elif candlestick_patterns.get('morning_star'):
                # Stop-loss below morning star pattern
                stop_loss = current_price * 0.985
                risk = current_price - stop_loss
                take_profit = current_price + (risk * 2)
                pattern_found = True
            
            # Fallback: Use ATR-based levels
            if not pattern_found:
                atr = volatility_data.atr
                stop_loss = current_price - (atr * 1.5)  # 1.5 ATR stop
                take_profit = current_price + (atr * 3)   # 2:1 reward-risk
        
        elif action == "SELL":
            # Try chart patterns first
            if chart_patterns.get('head_and_shoulders'):
                pattern = chart_patterns['head_and_shoulders']
                entry_price = pattern.entry_price
                stop_loss = pattern.stop_loss
                take_profit = pattern.take_profit
                pattern_found = True
            elif chart_patterns.get('triangle') and chart_patterns['triangle'].direction == "BEARISH":
                pattern = chart_patterns['triangle']
                entry_price = pattern.entry_price
                stop_loss = pattern.stop_loss
                take_profit = pattern.take_profit
                pattern_found = True
            
            # Try candlestick patterns
            elif candlestick_patterns.get('engulfing') and candlestick_patterns['engulfing'].direction == "BEARISH":
                # Stop-loss 1-2% above pattern high (Bigalow)
                stop_loss = current_price * 1.02
                # Take-profit 2:1 reward-risk
                risk = stop_loss - current_price
                take_profit = current_price - (risk * 2)
                pattern_found = True
            
            # Fallback: Use ATR-based levels
            if not pattern_found:
                atr = volatility_data.atr
                stop_loss = current_price + (atr * 1.5)  # 1.5 ATR stop
                take_profit = current_price - (atr * 3)   # 2:1 reward-risk
        
        # Calculate actual risk-reward ratio
        if stop_loss != 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 2.0
        
        return entry_price, stop_loss, take_profit, risk_reward
    
    def _calculate_position_size(
        self,
        action: str,
        strength: int,
        current_portfolio_risk: float,
        volatility_level: str
    ) -> float:
        """
        Calculate position size as % of portfolio
        
        Bigalow's risk management:
        - 1-2% risk per trade
        - Max 5-10% portfolio exposure
        - Reduce size in high volatility
        """
        if action == "HOLD":
            return 0.0
        
        # Base size depends on strength
        if strength >= 4:
            base_size = 0.02  # 2% for strong signals
        elif strength >= 3:
            base_size = 0.015  # 1.5% for medium signals
        else:
            base_size = 0.01  # 1% for weak signals
        
        # Adjust for volatility
        if volatility_level == "HIGH":
            base_size *= 0.7  # Reduce 30% in high volatility
        elif volatility_level == "LOW":
            base_size *= 1.2  # Increase 20% in low volatility
        
        # Check portfolio risk limit
        if current_portfolio_risk + base_size > self.max_portfolio_risk:
            base_size = max(0, self.max_portfolio_risk - current_portfolio_risk)
        
        return round(base_size, 4)
    
    def _calculate_leverage(
        self,
        strength: int,
        volatility_level: str,
        trend_strength: float
    ) -> float:
        """
        Calculate recommended leverage (2-5x per problem statement)
        
        Bigalow's principle: Higher leverage for stronger signals in stable trends
        """
        # Base leverage
        if strength >= 5:
            leverage = 5.0
        elif strength >= 4:
            leverage = 4.0
        elif strength >= 3:
            leverage = 3.0
        else:
            leverage = 2.0
        
        # Reduce in high volatility
        if volatility_level == "HIGH":
            leverage = max(2.0, leverage - 1.0)
        
        # Reduce if trend weak
        if trend_strength < 50:
            leverage = max(2.0, leverage - 0.5)
        
        return leverage
    
    def _check_trend_confirmation(self, trend_data: Any, action: str) -> bool:
        """Check if trend confirms the action (Murphy's principle)"""
        if action == "BUY":
            return trend_data.trend == "BULLISH"
        elif action == "SELL":
            return trend_data.trend == "BEARISH"
        return False
    
    def _check_volume_confirmation(
        self,
        df: pd.DataFrame,
        chart_patterns: Dict,
        candlestick_patterns: Dict
    ) -> bool:
        """Check if volume confirms the pattern (Bulkowski/Nison)"""
        # Check chart patterns
        for pattern_key in ['double_bottom', 'head_and_shoulders', 'triangle']:
            if chart_patterns.get(pattern_key):
                pattern = chart_patterns[pattern_key]
                if hasattr(pattern, 'volume_confirmed') and pattern.volume_confirmed:
                    return True
        
        # Check candlestick patterns
        for pattern_key in ['hammer', 'engulfing', 'morning_star']:
            if candlestick_patterns.get(pattern_key):
                pattern = candlestick_patterns[pattern_key]
                if hasattr(pattern, 'volume_confirmed') and pattern.volume_confirmed:
                    return True
        
        return False
    
    def _check_indicator_confirmation(self, momentum_data: Any, action: str) -> bool:
        """Check if indicators confirm the action (Murphy)"""
        if action == "BUY":
            # RSI oversold AND MACD bullish
            return momentum_data.rsi < 40 and momentum_data.macd_trend == "BULLISH"
        elif action == "SELL":
            # RSI overbought AND MACD bearish
            return momentum_data.rsi > 60 and momentum_data.macd_trend == "BEARISH"
        return False
    
    def _create_hold_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a HOLD signal"""
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            strength=0,
            confidence=0,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            risk_reward_ratio=0,
            chart_pattern=None,
            candlestick_pattern=None,
            trend="NEUTRAL",
            rsi_signal="NEUTRAL",
            macd_signal="NEUTRAL",
            trend_confirmed=False,
            volume_confirmed=False,
            indicator_confirmed=False,
            position_size_pct=0,
            leverage_recommended=0,
            reason=reason
        )