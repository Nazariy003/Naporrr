# analysis/volume.py
import time
import math
import statistics
from typing import Dict, Any, List, Tuple
from collections import deque
from config.settings import settings
from data.storage import DataStorage, TradeEntry
from utils.logger import logger

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

class VolumeAnalyzer:
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.cfg = settings.volume
        self.adaptive_cfg = settings.adaptive
        self.tape_analyzer = TapeAnalyzer(large_trade_threshold_usdt=5000)
        self._last_calculations = {}
        self._adaptive_windows_cache = {}

    def get_adaptive_window(self, base_window: int, symbol: str, current_volatility: float) -> int:
        """Адаптивне розширення/звуження вікна на основі волатильності"""
        if not self.adaptive_cfg.enable_adaptive_windows:
            return base_window

        cache_key = f"{symbol}_{base_window}"
        if cache_key in self._adaptive_windows_cache:
            cached_data = self._adaptive_windows_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 30:  # Кеш на 30 сек
                return cached_data['window']

        # Розрахунок адаптивного множника
        base_vol = self.adaptive_cfg.base_volatility_threshold
        if current_volatility <= base_vol * 0.7:
            # Дуже низька волатильність - розширюємо вікно
            multiplier = self.adaptive_cfg.low_volatility_multiplier
        elif current_volatility >= base_vol * 2.0:
            # Дуже висока волатильність - звужуємо вікно
            multiplier = self.adaptive_cfg.high_volatility_multiplier
        else:
            # Нормальна волатильність - незмінне вікно
            multiplier = 1.0

        adaptive_window = int(base_window * multiplier)
        
        # Обмеження розміру вікна
        max_window = int(base_window * self.adaptive_cfg.max_window_expansion)
        min_window = int(base_window * self.adaptive_cfg.min_window_reduction)
        adaptive_window = max(min_window, min(adaptive_window, max_window))

        # Кешуємо результат
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
        trades = self.storage.get_trades(symbol)
        now = time.time()
        
        if len(trades) < 5:
            if symbol in self._last_calculations:
                return self._last_calculations[symbol]
            return self._get_default_volume_data(symbol, now)
        
        # Спочатку обчислюємо волатильність з базовими вікнами
        volatility_metrics = self._calculate_volatility_metrics(symbol, trades, now)
        current_volatility = volatility_metrics.get("recent_volatility", 0.1)
        
        # Адаптивні вікна на основі поточної волатильності
        adaptive_short_window = self.get_adaptive_window(self.cfg.short_window_sec, symbol, current_volatility)
        adaptive_long_window = self.get_adaptive_window(self.cfg.long_window_sec, symbol, current_volatility)
        
        # Перераховуємо метрики з адаптивними вікнами
        volume_metrics = self._calculate_adaptive_volume_metrics(symbol, trades, now, adaptive_short_window, adaptive_long_window)
        
        result = {**volume_metrics, **volatility_metrics}
        result["volatility"] = result["range_position_lifetime"]
        result["adaptive_windows"] = {
            "short_sec": adaptive_short_window,
            "long_sec": adaptive_long_window,
            "volatility": current_volatility
        }
        
        self._last_calculations[symbol] = result
        
        return result

    def _calculate_volatility_metrics(self, symbol: str, trades: List[TradeEntry], now: float) -> Dict[str, Any]:
        """Правильний розрахунок волатильності з реальними даними"""
        position_lifetime_minutes = settings.risk.position_lifetime_minutes
        window_seconds = position_lifetime_minutes * 60
        
        # Отримуємо трейди за останній період
        recent_trades = [t for t in trades if t.ts >= now - window_seconds]
        
        if len(recent_trades) < 10:
            # Недостатньо даних для точного розрахунку
            return {
                "range_position_lifetime": 0.1,  # Мінімальна волатильність
                "atr_position_lifetime": 0.05,
                "recent_volatility": 0.1,
                "volatility_score": 10.0,
                "position_lifetime_minutes": position_lifetime_minutes
            }
        
        # Сортуємо трейди за часом
        recent_trades.sort(key=lambda x: x.ts)
        
        # Розраховуємо справжню волатильність
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
        
        # True Range та ATR
        true_ranges = []
        for i in range(1, len(recent_trades)):
            current_high = max(recent_trades[i].price, recent_trades[i-1].price)
            current_low = min(recent_trades[i].price, recent_trades[i-1].price)
            true_range = current_high - current_low
            true_ranges.append(true_range)
        
        atr = statistics.mean(true_ranges) if true_ranges else 0
        atr_pct = (atr / avg_price) * 100 if avg_price > 0 else 0.05
        
        # Price Range
        price_range_pct = ((high_price - low_price) / avg_price) * 100
        
        # Стандартне відхилення
        if len(prices) >= 2:
            price_std = statistics.stdev(prices)
            volatility_std = (price_std / avg_price) * 100
        else:
            volatility_std = 0.1
        
        # Комбінована волатильність
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
        """Розрахунок оцінки волатильності"""
        avg_volatility = (range_vol + atr_vol + std_vol) / 3
        # Нормалізуємо до шкали 0-100
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
        
        # Багаточасовий моментум з адаптивними вікнами
        momentum_metrics = {}
        if self.cfg.enable_multi_timeframe_momentum:
            # Отримуємо поточну волатильність з уже розрахованих даних
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
    
    def _get_default_volume_data(self, symbol: str, timestamp: float, short_count: int = 0, long_count: int = 0) -> Dict[str, Any]:
        """Повертає дані за замовчуванням при відсутності достатньої кількості трейдів"""
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
            }
        }