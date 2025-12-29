# analysis/volume.py
import time
import math
import statistics
import numpy as np
from typing import Dict, Any, List, Tuple, Deque
from collections import deque, defaultdict
from datetime import datetime
from config.settings import settings
from data.storage import DataStorage, TradeEntry
from utils.logger import logger

class MultiTimeframeVolumeAnalyzer:
    """Багатотаймфреймовий аналізатор обсягів"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.timeframes = [60, 300, 1800]  # 1, 5, 30 хвилин
        self.timeframe_weights = [0.4, 0.35, 0.25]
        self.volume_history = defaultdict(lambda: defaultdict(deque))
        self.momentum_history = defaultdict(lambda: defaultdict(deque))
        self.volatility_history = defaultdict(lambda: defaultdict(deque))
        
    def analyze_multi_timeframe(self, symbol: str) -> Dict[str, Any]:
        """Багатотаймфреймовий аналіз обсягів та моментуму"""
        current_time = time.time()
        trades = self.storage.get_trades(symbol)
        
        results = {
            'timeframes': {},
            'convergence': {},
            'combined': {}
        }
        
        # Аналіз для кожного таймфрейму
        for tf in self.timeframes:
            tf_key = f'tf_{tf}'
            tf_results = self._analyze_timeframe(symbol, trades, tf, current_time)
            results['timeframes'][tf_key] = tf_results
            
            # Оновлення історії
            self._update_history(symbol, tf, tf_results, current_time)
        
        # Аналіз конвергенції
        convergence = self._analyze_volume_convergence(symbol, results['timeframes'])
        results['convergence'] = convergence
        
        # Комбіновані значення
        combined = self._calculate_combined_values(symbol, results['timeframes'], convergence)
        results['combined'] = combined
        
        return results
    
    def _analyze_timeframe(self, symbol: str, trades: List[TradeEntry], 
                          timeframe: int, current_time: float) -> Dict[str, Any]:
        """Аналіз одного таймфрейму"""
        cutoff = current_time - timeframe
        
        period_trades = [t for t in trades if t.ts >= cutoff] if trades else []
        
        if not period_trades:
            return self._get_empty_timeframe_data(timeframe)
        
        # Розрахунок метрик
        volume_metrics = self._calculate_volume_metrics(period_trades)
        momentum_metrics = self._calculate_momentum_metrics(period_trades)
        volatility_metrics = self._calculate_volatility_metrics(period_trades)
        tape_analysis = self._analyze_tape_patterns(period_trades)
        
        return {
            'timeframe_sec': timeframe,
            'trades_count': len(period_trades),
            'volume': volume_metrics,
            'momentum': momentum_metrics,
            'volatility': volatility_metrics,
            'tape_analysis': tape_analysis,
            'timestamp': current_time
        }
    
    def _calculate_volume_metrics(self, trades: List[TradeEntry]) -> Dict[str, Any]:
        """Розрахунок метрик обсягу"""
        if not trades:
            return {'total': 0, 'buy': 0, 'sell': 0, 'vwap': 0}
        
        total_volume = sum(t.size * t.price for t in trades)
        buy_volume = sum(t.size * t.price for t in trades if t.side.lower() == 'buy')
        sell_volume = sum(t.size * t.price for t in trades if t.side.lower() == 'sell')
        
        if total_volume > 0:
            vwap = sum(t.price * t.size for t in trades) / sum(t.size for t in trades)
        else:
            vwap = 0
        
        return {
            'total': total_volume,
            'buy': buy_volume,
            'sell': sell_volume,
            'imbalance': (buy_volume - sell_volume) / total_volume * 100 if total_volume > 0 else 0,
            'vwap': vwap
        }
    
    def _calculate_momentum_metrics(self, trades: List[TradeEntry]) -> Dict[str, Any]:
        """Розрахунок метрик моментуму"""
        if len(trades) < 5:
            return {'score': 0, 'trend': 0, 'acceleration': 0}
        
        segment_size = max(1, len(trades) // 5)
        segments = []
        
        for i in range(0, len(trades), segment_size):
            segment = trades[i:i + segment_size]
            if segment:
                buy_vol = sum(t.size * t.price for t in segment if t.side.lower() == 'buy')
                sell_vol = sum(t.size * t.price for t in segment if t.side.lower() == 'sell')
                total = buy_vol + sell_vol
                
                if total > 0:
                    momentum = (buy_vol - sell_vol) / total * 100
                    segments.append(momentum)
        
        if not segments:
            return {'score': 0, 'trend': 0, 'acceleration': 0}
        
        x = np.arange(len(segments))
        y = np.array(segments)
        
        try:
            trend = np.polyfit(x, y, 1)[0]
            
            if len(segments) >= 3:
                acceleration = np.polyfit(x, y, 2)[0] * 2
            else:
                acceleration = 0
        except:
            trend = 0
            acceleration = 0
        
        return {
            'score': np.mean(segments),
            'trend': trend,
            'acceleration': acceleration,
            'segments': segments
        }
    
    def _calculate_volatility_metrics(self, trades: List[TradeEntry]) -> Dict[str, Any]:
        """Розрахунок метрик волатильності"""
        if len(trades) < 3:
            return {'atr': 0, 'range': 0, 'std': 0, 'score': 0}
        
        prices = [t.price for t in trades]
        
        true_ranges = []
        for i in range(1, len(trades)):
            high = max(trades[i].price, trades[i-1].price)
            low = min(trades[i].price, trades[i-1].price)
            true_ranges.append(high - low)
        
        atr = np.mean(true_ranges) if true_ranges else 0
        
        price_range = max(prices) - min(prices)
        range_pct = price_range / np.mean(prices) * 100 if np.mean(prices) > 0 else 0
        
        std = np.std(prices) if len(prices) > 1 else 0
        std_pct = std / np.mean(prices) * 100 if np.mean(prices) > 0 else 0
        
        volatility_score = (range_pct * 0.4 + std_pct * 0.4 + (atr / np.mean(prices) * 100) * 0.2) \
                          if np.mean(prices) > 0 else 0
        
        return {
            'atr': atr,
            'range': range_pct,
            'std': std_pct,
            'score': volatility_score
        }
    
    def _analyze_tape_patterns(self, trades: List[TradeEntry]) -> Dict[str, Any]:
        """Аналіз патернів стрічки"""
        if len(trades) < 10:
            return {'large_orders': 0, 'absorption': False, 'sequences': []}
        
        large_threshold = 10000
        large_buys = sum(1 for t in trades if t.size * t.price >= large_threshold and t.side.lower() == 'buy')
        large_sells = sum(1 for t in trades if t.size * t.price >= large_threshold and t.side.lower() == 'sell')
        
        sequences = []
        current_side = trades[0].side.lower()
        current_count = 1
        
        for i in range(1, len(trades)):
            if trades[i].side.lower() == current_side:
                current_count += 1
            else:
                sequences.append({'side': current_side, 'count': current_count})
                current_side = trades[i].side.lower()
                current_count = 1
        
        sequences.append({'side': current_side, 'count': current_count})
        
        absorption = self._detect_absorption(trades)
        
        return {
            'large_orders': {'buy': large_buys, 'sell': large_sells, 'net': large_buys - large_sells},
            'absorption': absorption,
            'sequences': sequences[-5:],
            'dominant_side': max(set([s['side'] for s in sequences]), 
                               key=[s['side'] for s in sequences].count) if sequences else 'NONE'
        }
    
    def _detect_absorption(self, trades: List[TradeEntry]) -> Dict[str, bool]:
        """Виявлення патернів поглинання"""
        if len(trades) < 20:
            return {'bullish': False, 'bearish': False}
        
        half = len(trades) // 2
        first_half = trades[:half]
        second_half = trades[half:]
        
        first_prices = [t.price for t in first_half]
        second_prices = [t.price for t in second_half]
        
        first_avg = np.mean(first_prices) if first_prices else 0
        second_avg = np.mean(second_prices) if second_prices else 0
        
        first_buy_vol = sum(t.size * t.price for t in first_half if t.side.lower() == 'buy')
        first_sell_vol = sum(t.size * t.price for t in first_half if t.side.lower() == 'sell')
        
        second_buy_vol = sum(t.size * t.price for t in second_half if t.side.lower() == 'buy')
        second_sell_vol = sum(t.size * t.price for t in second_half if t.side.lower() == 'sell')
        
        bullish_absorption = (second_avg < first_avg and second_buy_vol > first_buy_vol * 1.5)
        bearish_absorption = (second_avg > first_avg and second_sell_vol > first_sell_vol * 1.5)
        
        return {'bullish': bullish_absorption, 'bearish': bearish_absorption}
    
    def _update_history(self, symbol: str, timeframe: int, data: Dict, timestamp: float):
        """Оновлення історії"""
        vol_key = f'volume_{timeframe}'
        self.volume_history[symbol][vol_key].append({
            'timestamp': timestamp,
            'value': data['volume']['total']
        })
        
        mom_key = f'momentum_{timeframe}'
        self.momentum_history[symbol][mom_key].append({
            'timestamp': timestamp,
            'value': data['momentum']['score']
        })
        
        vol_key = f'volatility_{timeframe}'
        self.volatility_history[symbol][vol_key].append({
            'timestamp': timestamp,
            'value': data['volatility']['score']
        })
    
    def _analyze_volume_convergence(self, symbol: str, timeframe_data: Dict) -> Dict[str, Any]:
        """Аналіз конвергенції обсягів"""
        if len(timeframe_data) < 2:
            return {'score': 0, 'confirmed': False, 'alignment': 'NONE'}
        
        volume_trends = []
        momentum_trends = []
        
        for tf_key, data in timeframe_data.items():
            volume_trends.append(data['volume']['imbalance'])
            momentum_trends.append(data['momentum']['trend'])
        
        positive_volume = sum(1 for v in volume_trends if v > 10)
        negative_volume = sum(1 for v in volume_trends if v < -10)
        
        positive_momentum = sum(1 for m in momentum_trends if m > 0.5)
        negative_momentum = sum(1 for m in momentum_trends if m < -0.5)
        
        convergence_score = 0.0
        confirmed = False
        alignment = 'NONE'
        
        if positive_volume >= 2 and positive_momentum >= 2:
            convergence_score = positive_volume / 3.0
            confirmed = True
            alignment = 'BULLISH'
        elif negative_volume >= 2 and negative_momentum >= 2:
            convergence_score = negative_volume / 3.0
            confirmed = True
            alignment = 'BEARISH'
        elif len(volume_trends) >= 2:
            convergence_score = 0.5
            alignment = 'MIXED'
        
        return {
            'score': convergence_score,
            'confirmed': confirmed,
            'alignment': alignment,
            'volume_trends': volume_trends,
            'momentum_trends': momentum_trends
        }
    
    def _calculate_combined_values(self, symbol: str, timeframe_data: Dict, 
                                 convergence: Dict) -> Dict[str, Any]:
        """Розрахунок комбінованих значень"""
        if not timeframe_data:
            return {'volume': 0, 'momentum': 0, 'volatility': 0, 'confidence': 0}
        
        weights = {'tf_60': 0.4, 'tf_300': 0.35, 'tf_1800': 0.25}
        
        combined_volume = 0
        combined_momentum = 0
        combined_volatility = 0
        total_weight = 0
        
        convergence_score = convergence.get('score', 0)
        
        for tf_key, weight in weights.items():
            if tf_key in timeframe_data:
                data = timeframe_data[tf_key]
                
                conv_bonus = 1.0 + convergence_score * 0.5
                adjusted_weight = weight * conv_bonus
                
                combined_volume += data['volume']['imbalance'] * adjusted_weight
                combined_momentum += data['momentum']['score'] * adjusted_weight
                combined_volatility += data['volatility']['score'] * adjusted_weight
                total_weight += adjusted_weight
        
        if total_weight > 0:
            combined_volume /= total_weight
            combined_momentum /= total_weight
            combined_volatility /= total_weight
        
        confidence = min(1.0, convergence_score * 1.5)
        
        return {
            'volume': combined_volume,
            'momentum': combined_momentum,
            'volatility': combined_volatility,
            'confidence': confidence,
            'weighted': True
        }
    
    def _get_empty_timeframe_data(self, timeframe: int) -> Dict[str, Any]:
        """Порожні дані для таймфрейму"""
        return {
            'timeframe_sec': timeframe,
            'trades_count': 0,
            'volume': {'total': 0, 'buy': 0, 'sell': 0, 'imbalance': 0, 'vwap': 0},
            'momentum': {'score': 0, 'trend': 0, 'acceleration': 0, 'segments': []},
            'volatility': {'atr': 0, 'range': 0, 'std': 0, 'score': 0},
            'tape_analysis': {'large_orders': {'buy': 0, 'sell': 0, 'net': 0},
                             'absorption': {'bullish': False, 'bearish': False},
                             'sequences': [], 'dominant_side': 'NONE'},
            'timestamp': time.time()
        }

class TapeAnalyzer:
    def __init__(self, large_trade_threshold_usdt: float = 5000.0):
        self.large_trade_threshold = large_trade_threshold_usdt
        
    def analyze_tape_patterns(self, symbol: str, trades: List[TradeEntry], window_seconds: int = 30) -> Dict[str, Any]:
        now = time.time()
        window_start = now - window_seconds
        recent_trades = [t for t in trades if t.ts >= window_start]
        
        if len(recent_trades) < 3:
            return {
                "large_buys_count": 0, "large_sells_count": 0, "large_net": 0,
                "volume_acceleration": 0.0, "absorption_bullish": False, "absorption_bearish": False,
                "buy_volume": 0.0, "sell_volume": 0.0, "total_trades": 0, "trade_sequences": {}
            }
        
        recent_trades.sort(key=lambda x: x.ts)
        
        large_buys = [t for t in recent_trades if t.size * t.price >= self.large_trade_threshold and t.side.lower() == "buy"]
        large_sells = [t for t in recent_trades if t.size * t.price >= self.large_trade_threshold and t.side.lower() == "sell"]
        
        mid_time = window_start + (window_seconds / 2)
        first_half = [t for t in recent_trades if t.ts < mid_time]
        second_half = [t for t in recent_trades if t.ts >= mid_time]
        
        volume_first = sum(t.size * t.price for t in first_half) if first_half else 0.1
        volume_second = sum(t.size * t.price for t in second_half) if second_half else 0.1
        
        volume_acceleration = 0.0
        if volume_first > 0:
            volume_acceleration = (volume_second - volume_first) / volume_first * 100.0
        
        absorption_bullish, absorption_bearish = self.detect_absorption_patterns(recent_trades)
        sequences = self.analyze_trade_sequences(recent_trades)
        
        return {
            "large_buys_count": len(large_buys), "large_sells_count": len(large_sells), "large_net": len(large_buys) - len(large_sells),
            "volume_acceleration": volume_acceleration, "absorption_bullish": absorption_bullish, "absorption_bearish": absorption_bearish,
            "buy_volume": sum(t.size * t.price for t in recent_trades if t.side.lower() == "buy"),
            "sell_volume": sum(t.size * t.price for t in recent_trades if t.side.lower() == "sell"),
            "total_trades": len(recent_trades), "trade_sequences": sequences
        }
    
    def detect_absorption_patterns(self, trades: List[TradeEntry]) -> Tuple[bool, bool]:
        if len(trades) < 6:
            return False, False
        
        split_idx = len(trades) // 2
        first_half = trades[:split_idx]
        second_half = trades[split_idx:]
        
        avg_first = sum(t.price for t in first_half) / len(first_half)
        avg_second = sum(t.price for t in second_half) / len(second_half)
        
        price_falling = avg_second < avg_first
        price_rising = avg_second > avg_first
        
        large_buys_second = [t for t in second_half if t.size * t.price >= self.large_trade_threshold and t.side.lower() == "buy"]
        large_sells_second = [t for t in second_half if t.size * t.price >= self.large_trade_threshold and t.side.lower() == "sell"]
        
        absorption_bullish = price_falling and len(large_buys_second) > len(large_sells_second)
        absorption_bearish = price_rising and len(large_sells_second) > len(large_buys_second)
        
        return absorption_bullish, absorption_bearish
    
    def analyze_trade_sequences(self, trades: List[TradeEntry]) -> Dict[str, Any]:
        if len(trades) < 3:
            return {}
        
        sequences = []
        current_side = trades[0].side.lower()
        current_sequence = [trades[0]]
        
        for trade in trades[1:]:
            if trade.side.lower() == current_side:
                current_sequence.append(trade)
            else:
                sequences.append({'side': current_side, 'count': len(current_sequence), 'total_volume': sum(t.size * t.price for t in current_sequence)})
                current_side = trade.side.lower()
                current_sequence = [trade]
        
        if current_sequence:
            sequences.append({'side': current_side, 'count': len(current_sequence), 'total_volume': sum(t.size * t.price for t in current_sequence)})
        
        buy_sequences = [s for s in sequences if s['side'] == 'buy']
        sell_sequences = [s for s in sequences if s['side'] == 'sell']
        
        longest_buy = max(buy_sequences, key=lambda x: x['count']) if buy_sequences else None
        longest_sell = max(sell_sequences, key=lambda x: x['count']) if sell_sequences else None
        
        return {'longest_buy_sequence': longest_buy, 'longest_sell_sequence': longest_sell, 'total_sequences': len(sequences)}

class AdaptiveVolumeAnalyzer:
    """✅ АДАПТИВНИЙ АНАЛІЗАТОР ОБСЯГІВ І ВЕЛИКИХ ОРДЕРІВ"""
    
    def __init__(self, config):
        self.cfg = config.volume
        self.imb_cfg = config.imbalance
        
        self.volume_history: Dict[str, Deque] = {}
        self.order_size_history: Dict[str, Deque] = {}
        self.ema_fast: Dict[str, float] = {}
        self.ema_slow: Dict[str, float] = {}
    
    def update_volume(self, symbol: str, volume: float):
        """Оновлюємо історію обсягів"""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.cfg.volume_lookback_periods)
        
        self.volume_history[symbol].append({
            'volume': volume,
            'timestamp': datetime.now()
        })
        
        if self.cfg.enable_ema_volume_analysis:
            self._update_ema(symbol, volume)
    
    def update_order_size(self, symbol: str, order_size: float, side: str):
        """Оновлюємо історію розмірів ордерів"""
        if symbol not in self.order_size_history:
            max_len = self.imb_cfg.large_order_lookback_periods if self.imb_cfg.enable_adaptive_large_orders else 100
            self.order_size_history[symbol] = deque(maxlen=max_len)
        
        self.order_size_history[symbol].append({
            'size': order_size,
            'side': side,
            'timestamp': datetime.now()
        })
    
    def _update_ema(self, symbol: str, volume: float):
        """Оновлюємо EMA для швидкого аналізу"""
        alpha_fast = 2 / (self.cfg.ema_fast_period + 1)
        alpha_slow = 2 / (self.cfg.ema_slow_period + 1)
        
        if symbol not in self.ema_fast:
            self.ema_fast[symbol] = volume
            self.ema_slow[symbol] = volume
        else:
            self.ema_fast[symbol] = alpha_fast * volume + (1 - alpha_fast) * self.ema_fast[symbol]
            self.ema_slow[symbol] = alpha_slow * volume + (1 - alpha_slow) * self.ema_slow[symbol]
    
    def analyze_volume(self, symbol: str, current_volume: float) -> Dict:
        """✅ АДАПТИВНИЙ аналіз обсягу - використовує 3 методи та консенсус"""
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < self.cfg.volume_min_samples:
            return {
                'classification': 'INSUFFICIENT_DATA',
                'zscore': 0.0,
                'percentile': 50.0,
                'ema_ratio': 1.0,
                'methods': {}
            }
        
        results = {}
        
        if self.cfg.enable_adaptive_volume_analysis:
            zscore, zscore_class = self._calculate_zscore(symbol, current_volume)
            results['zscore'] = {'score': zscore, 'class': zscore_class}
        
        if self.cfg.enable_percentile_method:
            percentile, perc_class = self._calculate_percentile(symbol, current_volume)
            results['percentile'] = {'score': percentile, 'class': perc_class}
        
        if self.cfg.enable_ema_volume_analysis and symbol in self.ema_slow:
            ema_ratio, ema_class = self._calculate_ema_ratio(symbol, current_volume)
            results['ema'] = {'score': ema_ratio, 'class': ema_class}
        
        final_class = self._consensus_classification(results)
        
        return {
            'classification': final_class,
            'zscore': results.get('zscore', {}).get('score', 0.0),
            'percentile': results.get('percentile', {}).get('score', 50.0),
            'ema_ratio': results.get('ema', {}).get('score', 1.0),
            'methods': results
        }
    
    def _calculate_zscore(self, symbol: str, current_volume: float) -> Tuple[float, str]:
        """Z-Score метод (перевірена статистика)"""
        volumes = [h['volume'] for h in self.volume_history[symbol]]
        
        mean = np.mean(volumes)
        std = np.std(volumes)
        
        if std == 0:
            return 0.0, "NO_VOLATILITY"
        
        zscore = (current_volume - mean) / std
        
        if zscore > self.cfg.volume_zscore_threshold_very_high:
            classification = "EXTREMELY_HIGH"
        elif zscore > self.cfg.volume_zscore_threshold_high:
            classification = "VERY_HIGH"
        elif zscore > 0.5:
            classification = "HIGH"
        elif zscore > self.cfg.volume_zscore_threshold_low:
            classification = "NORMAL"
        else:
            classification = "LOW"
        
        logger.debug(f"[VOLUME_ZSCORE] {symbol}: vol={current_volume:.0f}, "
                    f"mean={mean:.0f}, std={std:.0f}, zscore={zscore:.2f}, class={classification}")
        
        return zscore, classification
    
    def _calculate_percentile(self, symbol: str, current_volume: float) -> Tuple[float, str]:
        """Percentile метод (топ X%)"""
        volumes = [h['volume'] for h in self.volume_history[symbol]]
        
        percentile = (np.sum(np.array(volumes) <= current_volume) / len(volumes)) * 100
        
        if percentile >= self.cfg.volume_percentile_very_high:
            classification = "EXTREMELY_HIGH"
        elif percentile >= self.cfg.volume_percentile_high:
            classification = "HIGH"
        elif percentile >= self.cfg.volume_percentile_low:
            classification = "NORMAL"
        else:
            classification = "LOW"
        
        return percentile, classification
    
    def _calculate_ema_ratio(self, symbol: str, current_volume: float) -> Tuple[float, str]:
        """EMA-based метод (швидка адаптація)"""
        if self.ema_slow[symbol] == 0:
            return 1.0, "NORMAL"
        
        ratio = current_volume / self.ema_slow[symbol]
        
        if ratio > self.cfg.ema_ratio_very_high:
            classification = "EXTREMELY_HIGH"
        elif ratio > self.cfg.ema_ratio_high:
            classification = "HIGH"
        elif ratio > 0.7:
            classification = "NORMAL"
        else:
            classification = "LOW"
        
        return ratio, classification
    
    def _consensus_classification(self, results: Dict) -> str:
        """Консенсус між 3 методами"""
        classifications = [m['class'] for m in results.values()]
        
        high_votes = sum(1 for c in classifications if 'HIGH' in c or 'EXTREME' in c)
        low_votes = sum(1 for c in classifications if 'LOW' in c)
        
        if high_votes >= 2:
            return "HIGH"
        elif low_votes >= 2:
            return "LOW"
        else:
            return "NORMAL"
    
    def analyze_large_order(self, symbol: str, order_size: float, side: str) -> Dict:
        """✅ АДАПТИВНИЙ аналіз великого ордера (Z-Score метод)"""
        if not self.imb_cfg.enable_adaptive_large_orders:
            is_large = order_size > self.imb_cfg.large_order_min_notional_abs
            return {
                'is_large': is_large,
                'classification': 'LARGE' if is_large else 'NORMAL',
                'zscore': 0.0,
                'method': 'static'
            }
        
        if symbol not in self.order_size_history or len(self.order_size_history[symbol]) < self.imb_cfg.large_order_min_samples:
            return {
                'is_large': False,
                'classification': 'INSUFFICIENT_DATA',
                'zscore': 0.0,
                'method': 'adaptive'
            }
        
        sizes = [h['size'] for h in self.order_size_history[symbol]]
        
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        if std_size == 0:
            return {
                'is_large': False,
                'classification': 'NO_VOLATILITY',
                'zscore': 0.0,
                'method': 'adaptive'
            }
        
        zscore = (order_size - mean_size) / std_size
        
        if zscore > 3.0:
            classification = "WHALE"
            is_large = True
        elif zscore > self.imb_cfg.large_order_zscore_threshold:
            classification = "VERY_LARGE"
            is_large = True
        elif zscore > 1.0:
            classification = "LARGE"
            is_large = True
        else:
            classification = "NORMAL"
            is_large = False
        
        if is_large:
            logger.info(f"[LARGE_ORDER_ADAPTIVE] {symbol}: size={order_size:.0f} USD, "
                       f"mean={mean_size:.0f}, std={std_size:.0f}, zscore={zscore:.2f}, "
                       f"class={classification}, side={side}")
        
        return {
            'is_large': is_large,
            'classification': classification,
            'zscore': zscore,
            'mean_size': mean_size,
            'std_size': std_size,
            'method': 'adaptive'
        }
    
    def get_large_order_flow(self, symbol: str, lookback_sec: int = 60) -> Dict:
        """Аналіз потоку великих ордерів"""
        if symbol not in self.order_size_history:
            return {
                'direction': 'NEUTRAL',
                'buy_volume': 0,
                'sell_volume': 0,
                'imbalance': 0,
                'count': 0
            }
        
        now = datetime.now()
        recent_orders = [
            o for o in self.order_size_history[symbol]
            if (now - o['timestamp']).total_seconds() <= lookback_sec
        ]
        
        if not recent_orders:
            return {
                'direction': 'NEUTRAL',
                'buy_volume': 0,
                'sell_volume': 0,
                'imbalance': 0,
                'count': 0
            }
        
        large_orders = []
        for order in recent_orders:
            analysis = self.analyze_large_order(symbol, order['size'], order['side'])
            if analysis['is_large']:
                large_orders.append(order)
        
        if not large_orders:
            return {
                'direction': 'NEUTRAL',
                'buy_volume': 0,
                'sell_volume': 0,
                'imbalance': 0,
                'count': 0
            }
        
        buy_volume = sum(o['size'] for o in large_orders if o['side'] == 'buy')
        sell_volume = sum(o['size'] for o in large_orders if o['side'] == 'sell')
        
        total = buy_volume + sell_volume
        if total == 0:
            return {
                'direction': 'NEUTRAL',
                'buy_volume': 0,
                'sell_volume': 0,
                'imbalance': 0,
                'count': 0
            }
        
        buy_ratio = buy_volume / total
        imbalance = (buy_ratio - 0.5) * 200
        
        if buy_ratio > 0.7:
            direction = "STRONG_BUY"
        elif buy_ratio > 0.6:
            direction = "MEDIUM_BUY"
        elif buy_ratio > 0.4:
            direction = "NEUTRAL"
        elif buy_ratio > 0.3:
            direction = "MEDIUM_SELL"
        else:
            direction = "STRONG_SELL"
        
        logger.info(f"[LARGE_ORDER_FLOW_ADAPTIVE] {symbol}: {direction} "
                   f"(buy={buy_volume:.0f}, sell={sell_volume:.0f}, count={len(large_orders)})")
        
        return {
            'direction': direction,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'imbalance': imbalance,
            'count': len(large_orders)
        }
    
    def get_statistics(self, symbol: str) -> Dict:
        """Отримати статистику адаптивного аналізу"""
        if symbol not in self.volume_history:
            return {}
        
        volumes = [h['volume'] for h in self.volume_history[symbol]]
        
        return {
            'symbol': symbol,
            'mean_volume': np.mean(volumes) if volumes else 0,
            'std_volume': np.std(volumes) if volumes else 0,
            'min_volume': np.min(volumes) if volumes else 0,
            'max_volume': np.max(volumes) if volumes else 0,
            'samples': len(volumes),
            'ema_fast': self.ema_fast.get(symbol, 0),
            'ema_slow': self.ema_slow.get(symbol, 0),
            'adaptive_enabled': self.cfg.enable_adaptive_volume_analysis
        }

class VolumeAnalyzer:
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.cfg = settings.volume
        self.adaptive_cfg = settings.adaptive
        self.ohara_cfg = settings.ohara
        
        # 🆕 Багатотаймфреймовий аналізатор
        self.mtf_analyzer = MultiTimeframeVolumeAnalyzer(storage)
        
        # Ініціалізація інших компонентів
        self.tape_analyzer = TapeAnalyzer(large_trade_threshold_usdt=5000)
        self._last_calculations = {}
        self._adaptive_windows_cache = {}
        self.adaptive_analyzer = AdaptiveVolumeAnalyzer(settings)
        self._trade_frequency_baseline = {}
        self._volume_baseline = {}
        
        # 🆕 Додаємо відсутні атрибути
        self._ensure_mtf_attributes()

    def _ensure_mtf_attributes(self):
        """Забезпечуємо наявність необхідних MTF атрибутів"""
        if not hasattr(self.cfg, 'enable_multi_timeframe_analysis'):
            self.cfg.enable_multi_timeframe_analysis = True

    def get_adaptive_window(self, base_window: int, symbol: str, current_volatility: float) -> int:
        """Адаптивне розширення/звуження вікна на основі волатильності"""
        if not self.adaptive_cfg.enable_adaptive_windows:
            return base_window

        cache_key = f"{symbol}_{base_window}"
        if cache_key in self._adaptive_windows_cache:
            cached_data = self._adaptive_windows_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 30:
                return cached_data['window']

        base_vol = self.adaptive_cfg.base_volatility_threshold
        if current_volatility <= base_vol * 0.7:
            multiplier = self.adaptive_cfg.low_volatility_multiplier
        elif current_volatility >= base_vol * 2.0:
            multiplier = self.adaptive_cfg.high_volatility_multiplier
        else:
            multiplier = 1.0

        adaptive_window = int(base_window * multiplier)
        
        max_window = int(base_window * self.adaptive_cfg.max_window_expansion)
        min_window = int(base_window * self.adaptive_cfg.min_window_reduction)
        adaptive_window = max(min_window, min(adaptive_window, max_window))

        self._adaptive_windows_cache[cache_key] = {
            'window': adaptive_window,
            'timestamp': time.time(),
            'volatility': current_volatility,
            'multiplier': multiplier
        }

        logger.debug(f"[ADAPTIVE_WINDOW] {symbol}: vol={current_volatility:.3f}%, "
                    f"base={base_window}s, adaptive={adaptive_window}s, multiplier={multiplier:.2f}")

        return adaptive_window

    def calculate_vwap(self, trades: List[TradeEntry], total_volume: float) -> float:
        """Розрахунок Volume Weighted Average Price (VWAP)"""
        if total_volume <= self.cfg.vwap_min_volume:
            return 0.0
        
        try:
            vwap_numerator = sum(t.price * t.size for t in trades)
            total_quantity = sum(t.size for t in trades)
            
            if total_quantity == 0:
                return 0.0
                
            vwap = vwap_numerator / total_quantity
            
            logger.debug(f"[VWAP_CALC] {len(trades)} trades, total_quantity={total_quantity}, vwap={vwap:.6f}")
            
            return vwap
            
        except Exception as e:
            logger.error(f"[VWAP_ERROR] Calculation failed: {e}")
            return 0.0

    def compute(self, symbol: str) -> Dict[str, Any]:
        """Оновлений compute з MTF аналізом"""
        trades = self.storage.get_trades(symbol)
        now = time.time()
        
        # 🆕 Багатотаймфреймовий аналіз
        mtf_enabled = getattr(self.cfg, 'enable_multi_timeframe_analysis', True)
        
        if mtf_enabled:
            mtf_results = self.mtf_analyzer.analyze_multi_timeframe(symbol)
            combined_momentum = mtf_results['combined']['momentum']
            combined_volatility = mtf_results['combined']['volatility']
            convergence = mtf_results['convergence']
        else:
            # Fallback до оригінального розрахунку
            mtf_results = {}
            combined_momentum = 0
            combined_volatility = 0
            convergence = {'score': 0, 'confirmed': False}
        
        if len(trades) < 5:
            if symbol in self._last_calculations:
                result = self._last_calculations[symbol]
            else:
                result = self._get_default_volume_data(symbol, now)
            
            # 🆕 Додаємо MTF дані навіть при недостатніх даних
            if mtf_enabled:
                result["mtf_analysis"] = mtf_results
            
            self._last_calculations[symbol] = result
            return result
        
        # Адаптивний аналіз обсягу
        current_volume = sum(t.size * t.price for t in trades[-30:])
        self.adaptive_analyzer.update_volume(symbol, current_volume)
        
        for trade in trades[-30:]:
            order_size = trade.size * trade.price
            side = 'sell' if trade.side.lower() == 'sell' else 'buy'
            self.adaptive_analyzer.update_order_size(symbol, order_size, side)
        
        # Адаптивні вікна
        adaptive_short_window = self.get_adaptive_window(self.cfg.short_window_sec, symbol, combined_volatility)
        adaptive_long_window = self.get_adaptive_window(self.cfg.long_window_sec, symbol, combined_volatility)
        
        # Базові метрики
        volume_metrics = self._calculate_adaptive_volume_metrics(symbol, trades, now, 
                                                               adaptive_short_window, adaptive_long_window)
        
        # 🆕 Замінюємо ключові значення на MTF
        volume_metrics['momentum_score'] = combined_momentum
        if mtf_enabled:
            volume_metrics['total_volume_short'] = mtf_results['timeframes'].get('tf_60', {}).get('volume', {}).get('total', 0)
            volume_metrics['total_volume_long'] = mtf_results['timeframes'].get('tf_300', {}).get('volume', {}).get('total', 0)
        
        # Волатильність
        volatility_metrics = self._calculate_volatility_metrics(symbol, trades, now)
        volatility_metrics['recent_volatility'] = combined_volatility
        
        # Адаптивний аналіз
        adaptive_volume_analysis = self.adaptive_analyzer.analyze_volume(symbol, current_volume)
        large_order_flow = self.adaptive_analyzer.get_large_order_flow(symbol, lookback_sec=60)
        
        # O'Hara методи
        if self.cfg.enable_trade_frequency_analysis:
            frequency_data = self._analyze_trade_frequency(symbol, trades, now)
            volume_metrics['frequency_data'] = frequency_data
        else:
            volume_metrics['frequency_data'] = self._empty_frequency_data()
        
        if self.cfg.enable_volume_confirmation:
            volume_confirm = self._analyze_volume_confirmation_adaptive(symbol, trades, now, 
                                                                      volume_metrics, adaptive_volume_analysis)
            volume_metrics['volume_confirmation'] = volume_confirm
        else:
            volume_metrics['volume_confirmation'] = self._empty_volume_confirmation()
        
        volume_metrics['large_order_data'] = self._create_large_order_data_from_adaptive(large_order_flow)
        
        # Об'єднання результатів
        result = {**volume_metrics, **volatility_metrics}
        result["volatility"] = combined_volatility
        result["adaptive_windows"] = {
            "short_sec": adaptive_short_window,
            "long_sec": adaptive_long_window,
            "volatility": combined_volatility
        }
        
        # 🆕 Додаємо MTF дані
        if mtf_enabled:
            result["mtf_analysis"] = {
                'timeframes': mtf_results.get('timeframes', {}),
                'convergence': convergence,
                'combined': mtf_results.get('combined', {})
            }
        
        result["adaptive_volume_analysis"] = adaptive_volume_analysis
        result["adaptive_statistics"] = self.adaptive_analyzer.get_statistics(symbol)
        
        self._last_calculations[symbol] = result
        
        logger.debug(f"[VOLUME_MTF] {symbol}: "
                    f"MTF Mom={combined_momentum:.1f}%, "
                    f"MTF Vol={combined_volatility:.2f}%, "
                    f"Conv={convergence.get('score', 0):.2f}")
        
        return result

    def _calculate_volatility_metrics(self, symbol: str, trades: List[TradeEntry], now: float) -> Dict[str, Any]:
        """Правильний розрахунок волатильності з реальними даними"""
        position_lifetime_minutes = settings.risk.position_lifetime_minutes
        window_seconds = position_lifetime_minutes * 60
        
        recent_trades = [t for t in trades if t.ts >= now - window_seconds]
        
        if len(recent_trades) < 10:
            return {
                "range_position_lifetime": 0.1,
                "atr_position_lifetime": 0.05,
                "recent_volatility": 0.1,
                "volatility_score": 10.0,
                "position_lifetime_minutes": position_lifetime_minutes
            }
        
        recent_trades.sort(key=lambda x: x.ts)
        
        prices = [t.price for t in recent_trades]
        high_price = max(prices)
        low_price = min(prices)
        avg_price = statistics.mean(prices)
        
        if avg_price == 0:
            return {
                "range_position_lifetime": 0.1,
                "atr_position_lifetime": 0.05,
                "recent_volatility": 0.1,
                "volatility_score": 10.0,
                "position_lifetime_minutes": position_lifetime_minutes
            }
        
        true_ranges = []
        for i in range(1, len(recent_trades)):
            current_high = max(recent_trades[i].price, recent_trades[i-1].price)
            current_low = min(recent_trades[i].price, recent_trades[i-1].price)
            true_range = current_high - current_low
            true_ranges.append(true_range)
        
        atr = statistics.mean(true_ranges) if true_ranges else 0
        atr_pct = (atr / avg_price) * 100 if avg_price > 0 else 0.05
        
        price_range_pct = ((high_price - low_price) / avg_price) * 100
        
        if len(prices) >= 2:
            price_std = statistics.stdev(prices)
            volatility_std = (price_std / avg_price) * 100
        else:
            volatility_std = 0.1
        
        combined_volatility = max(0.1, (price_range_pct + atr_pct + volatility_std) / 3)
        
        logger.info(f"[VOLATILITY_REAL] {symbol}: {len(recent_trades)} trades, "
                   f"range={price_range_pct:.3f}%, atr={atr_pct:.3f}%, std={volatility_std:.3f}%")
        
        return {
            "range_position_lifetime": round(price_range_pct, 3),
            "atr_position_lifetime": round(atr_pct, 3),
            "recent_volatility": round(volatility_std, 3),
            "volatility_score": self._calculate_volatility_score(price_range_pct, atr_pct, volatility_std),
            "position_lifetime_minutes": position_lifetime_minutes,
            "trades_analyzed": len(recent_trades)
        }
    
    def _calculate_volatility_score(self, range_vol: float, atr_vol: float, std_vol: float) -> float:
        avg_volatility = (range_vol + atr_vol + std_vol) / 3
        score = min(100.0, avg_volatility * 10)
        return round(score, 1)
    
    def _calculate_adaptive_volume_metrics(self, symbol: str, trades: List[TradeEntry], now: float, 
                                         short_window: int, long_window: int) -> Dict[str, Any]:
        """Розрахунок метрик об'єму з адаптивними вікнами"""
        short_from = now - short_window
        long_from = now - long_window
        
        short_trades = [t for t in trades if t.ts >= short_from]
        long_trades = [t for t in trades if t.ts >= long_from]
        
        if len(short_trades) < self.cfg.default_min_trades:
            return self._get_default_volume_data(symbol, now, len(short_trades), len(long_trades))
        
        momentum_metrics = {}
        if self.cfg.enable_multi_timeframe_momentum:
            current_volatility = self._last_calculations.get(symbol, {}).get("recent_volatility", 0.1)
            momentum_metrics = self.calculate_adaptive_multi_timeframe_momentum(symbol, trades, now, current_volatility)
        else:
            momentum_score = self.enhanced_momentum(symbol, short_trades, short_window)
            momentum_metrics = {"momentum_score": momentum_score}
        
        tape_analysis = self.tape_analyzer.analyze_tape_patterns(symbol, short_trades, short_window)
        
        total_vol_short = sum(t.size * t.price for t in short_trades)
        total_vol_long = sum(t.size * t.price for t in long_trades)
        
        vwap = self.calculate_vwap(short_trades, total_vol_short)
        
        return {
            "symbol": symbol, 
            "vwap": round(vwap, 6), 
            "total_volume_short": total_vol_short, 
            "total_volume_long": total_vol_long,
            "buy_volume_short": sum(t.size * t.price for t in short_trades if t.side.lower() == "buy"),
            "sell_volume_short": sum(t.size * t.price for t in short_trades if t.side.lower() == "sell"),
            "short_trades_count": len(short_trades), 
            "long_trades_count": len(long_trades), 
            "timestamp": now, 
            "tape_analysis": tape_analysis,
            "trades": short_trades,
            **momentum_metrics
        }

    def calculate_adaptive_multi_timeframe_momentum(self, symbol: str, trades: List[TradeEntry], 
                                                  now: float, current_volatility: float) -> Dict[str, Any]:
        """Адаптивний багаточасовий моментум"""
        result = {}
        base_windows = self.cfg.momentum_windows
        weights = self.cfg.momentum_weights
        
        adaptive_windows = [
            self.get_adaptive_window(window, symbol, current_volatility) 
            for window in base_windows
        ]
        
        momentums = []
        weighted_sum = 0
        
        for i, window_sec in enumerate(adaptive_windows):
            window_start = now - window_sec
            window_trades = [t for t in trades if t.ts >= window_start]
            
            if len(window_trades) < 5:
                momentum = 0
            else:
                momentum = self.enhanced_momentum(symbol, window_trades, window_sec)
            
            momentums.append(momentum)
            weighted_sum += momentum * weights[i]
            
            result[f"momentum_{window_sec}s"] = momentum
        
        combined_momentum = weighted_sum
        result["momentum_score"] = combined_momentum
        result["momentum_breakdown"] = dict(zip([f"{w}s" for w in adaptive_windows], momentums))
        result["adaptive_momentum_windows"] = adaptive_windows
        
        if abs(combined_momentum) > 30:
            logger.info(f"[ADAPTIVE_MOMENTUM] {symbol}: combined={combined_momentum:.1f}% "
                       f"windows={adaptive_windows}")
        
        return result

    def enhanced_momentum(self, symbol: str, trades: List[TradeEntry], short_window: int) -> float:
        if not trades:
            return 0.0
        
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in trades:
            try:
                volume_usdt = trade.size * trade.price
                if trade.side.lower() == "buy":
                    buy_volume += volume_usdt
                elif trade.side.lower() == "sell":
                    sell_volume += volume_usdt
            except (TypeError, AttributeError) as e:
                logger.warning(f"[MOMENTUM] {symbol}: Error processing trade: {e}")
                continue
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
            
        momentum = (buy_volume - sell_volume) / total_volume * 100.0
        momentum = max(-100.0, min(100.0, momentum))
        
        return momentum
    
    def _analyze_trade_frequency(self, symbol: str, trades: List[TradeEntry], now: float) -> Dict[str, Any]:
        """O'HARA METHOD 3: Trade Frequency Analysis"""
        baseline_window = self.cfg.frequency_baseline_window_sec
        
        baseline_start = now - baseline_window
        baseline_trades = [t for t in trades if baseline_start <= t.ts < now]
        
        if len(baseline_trades) < 10:
            return self._empty_frequency_data()
        
        baseline_rate = len(baseline_trades) / (baseline_window / 60.0)
        
        if symbol not in self._trade_frequency_baseline:
            self._trade_frequency_baseline[symbol] = deque(maxlen=20)
        self._trade_frequency_baseline[symbol].append(baseline_rate)
        
        avg_baseline = statistics.mean(self._trade_frequency_baseline[symbol]) if self._trade_frequency_baseline[symbol] else baseline_rate
        
        current_window = 30
        current_start = now - current_window
        current_trades = [t for t in trades if t.ts >= current_start]
        current_rate = len(current_trades) / (current_window / 60.0)
        
        if avg_baseline > 0:
            ratio = current_rate / avg_baseline
        else:
            ratio = 1.0
        
        if ratio >= self.cfg.frequency_very_high_multiplier:
            activity_level = "VERY_HIGH"
            risk_signal = "AVOID"
        elif ratio >= self.cfg.frequency_high_multiplier:
            activity_level = "HIGH"
            risk_signal = "CAUTION"
        elif ratio <= self.cfg.frequency_very_low_multiplier:
            activity_level = "VERY_LOW"
            risk_signal = "LOW_LIQUIDITY"
        else:
            activity_level = "NORMAL"
            risk_signal = "OK"
        
        return {
            'current_rate': round(current_rate, 2),
            'baseline_rate': round(avg_baseline, 2),
            'ratio': round(ratio, 2),
            'activity_level': activity_level,
            'risk_signal': risk_signal,
            'current_trades': len(current_trades),
            'baseline_trades': len(baseline_trades)
        }

    def _analyze_volume_confirmation_adaptive(self, symbol: str, trades: List[TradeEntry], 
                                            now: float, volume_metrics: Dict, 
                                            adaptive_analysis: Dict) -> Dict[str, Any]:
        """O'HARA METHOD 5: Volume Confirmation (АДАПТИВНА ВЕРСІЯ)"""
        current_volume = volume_metrics.get('total_volume_short', 0)
        
        zscore = adaptive_analysis['zscore']
        volume_class = adaptive_analysis['classification']
        
        if len(trades) >= 10:
            recent_trades = trades[-10:]
            first_price = recent_trades[0].price
            last_price = recent_trades[-1].price
            price_change_pct = (last_price - first_price) / first_price * 100 if first_price > 0 else 0
        else:
            price_change_pct = 0
        
        if abs(price_change_pct) > 1.0:
            if zscore > self.cfg.volume_confirmation_zscore:
                confirmation = "CONFIRMED"
                strength = "STRONG"
            elif zscore > 0:
                confirmation = "MODERATE"
                strength = "MEDIUM"
            elif zscore < self.cfg.volume_weak_zscore:
                confirmation = "WEAK"
                strength = "WEAK"
            else:
                confirmation = "NEUTRAL"
                strength = "MEDIUM"
        else:
            confirmation = "NEUTRAL"
            strength = "WEAK"
        
        logger.debug(f"[VOLUME_CONFIRM_ADAPTIVE] {symbol}: zscore={zscore:.2f}, "
                    f"price_change={price_change_pct:.2f}%, confirm={confirmation}")
        
        return {
            'current_volume': round(current_volume, 2),
            'volume_zscore': round(zscore, 2),
            'volume_class': volume_class,
            'price_change_pct': round(price_change_pct, 2),
            'confirmation': confirmation,
            'strength': strength,
            'method': 'adaptive_zscore'
        }

    def _create_large_order_data_from_adaptive(self, large_order_flow: Dict) -> Dict[str, Any]:
        """Створюємо large_order_data з адаптивного аналізу"""
        direction = large_order_flow['direction']
        buy_volume = large_order_flow['buy_volume']
        sell_volume = large_order_flow['sell_volume']
        count = large_order_flow['count']
        
        return {
            'large_buy_count': count if 'BUY' in direction else 0,
            'large_sell_count': count if 'SELL' in direction else 0,
            'large_net': count if 'BUY' in direction else -count if 'SELL' in direction else 0,
            'informed_direction': direction,
            'large_buy_volume': buy_volume,
            'large_sell_volume': sell_volume,
            'method': 'adaptive_zscore'
        }

    def _empty_frequency_data(self) -> Dict[str, Any]:
        return {
            'current_rate': 0.0,
            'baseline_rate': 0.0,
            'ratio': 1.0,
            'activity_level': 'UNKNOWN',
            'risk_signal': 'UNKNOWN',
            'current_trades': 0,
            'baseline_trades': 0
        }

    def _empty_volume_confirmation(self) -> Dict[str, Any]:
        return {
            'current_volume': 0.0,
            'avg_volume': 0.0,
            'volume_ratio': 1.0,
            'price_change_pct': 0.0,
            'confirmation': 'UNKNOWN',
            'strength': 'UNKNOWN'
        }

    def _get_default_volume_data(self, symbol: str, timestamp: float, short_count: int = 0, long_count: int = 0) -> Dict[str, Any]:
        """Повертає дані за замовчуванням"""
        return {
            "symbol": symbol, "vwap": 0.0, "total_volume_short": 0.0, "total_volume_long": 0.0,
            "buy_volume_short": 0.0, "sell_volume_short": 0.0, "momentum_score": 0.0, "spike": False,
            "short_trades_count": short_count, "long_trades_count": long_count, "timestamp": timestamp,
            "tape_analysis": {
                "large_buys_count": 0, "large_sells_count": 0, "large_net": 0, "volume_acceleration": 0.0,
                "absorption_bullish": False, "absorption_bearish": False, "buy_volume": 0.0, "sell_volume": 0.0,
                "total_trades": 0, "trade_sequences": {}
            },
            "range_position_lifetime": 0.1, "atr_position_lifetime": 0.05, "recent_volatility": 0.1, 
            "volatility_score": 10.0, "trades_analyzed": short_count,
            "adaptive_windows": {
                "short_sec": self.cfg.short_window_sec,
                "long_sec": self.cfg.long_window_sec,
                "volatility": 0.1
            },
            "frequency_data": self._empty_frequency_data(),
            "volume_confirmation": self._empty_volume_confirmation(),
            "large_order_data": {
                'large_buy_count': 0, 'large_sell_count': 0, 'large_net': 0,
                'informed_direction': 'NEUTRAL', 'large_buy_volume': 0.0, 'large_sell_volume': 0.0
            },
            "adaptive_volume_analysis": {'classification': 'INSUFFICIENT_DATA', 'zscore': 0.0, 'percentile': 50.0, 'ema_ratio': 1.0},
            "adaptive_statistics": {},
            "mtf_analysis": {}  # 🆕 Додаємо пусті MTF дані
        }