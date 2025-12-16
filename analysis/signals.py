# analysis/signals.py
import time
import math
import numpy as np
from typing import Dict, Any, Tuple, List
from collections import deque, defaultdict
from config.settings import settings
from utils.logger import logger
from utils.signal_logger import signal_logger

class MultiTimeframeAnalyzer:
    """Багатотаймфреймовий аналізатор для конвергенції сигналів"""
    
    def __init__(self):
        self.timeframes = [60, 300, 1800]  # 1, 5, 30 хвилин
        self.timeframe_weights = [0.4, 0.35, 0.25]  # Ваги для комбінації
        self.historical_data = defaultdict(lambda: defaultdict(deque))
        
    def add_data_point(self, symbol: str, data_type: str, value: float, timestamp: float):
        """Додавання точки даних для всіх таймфреймів"""
        for tf in self.timeframes:
            key = f"{data_type}_{tf}"
            if symbol not in self.historical_data:
                self.historical_data[symbol] = defaultdict(lambda: deque(maxlen=1000))
            
            self.historical_data[symbol][key].append({
                'timestamp': timestamp,
                'value': value
            })
    
    def get_timeframe_data(self, symbol: str, data_type: str, timeframe: int, 
                          lookback_points: int = None) -> List[float]:
        """Отримання даних для конкретного таймфрейму"""
        key = f"{data_type}_{timeframe}"
        if symbol not in self.historical_data or key not in self.historical_data[symbol]:
            return []
        
        data = self.historical_data[symbol][key]
        current_time = time.time()
        
        if lookback_points:
            return [d['value'] for d in list(data)[-lookback_points:]]
        
        # Автоматичний відбір за часом
        timeframe_data = []
        cutoff = current_time - timeframe
        for item in reversed(data):
            if item['timestamp'] >= cutoff:
                timeframe_data.append(item['value'])
            else:
                break
                
        return list(reversed(timeframe_data))
    
    def calculate_convergence(self, symbol: str, data_type: str) -> Dict[str, Any]:
        """Розрахунок конвергенції між таймфреймами"""
        results = {}
        all_values = []
        
        for tf, weight in zip(self.timeframes, self.timeframe_weights):
            tf_data = self.get_timeframe_data(symbol, data_type, tf)
            if not tf_data:
                continue
                
            tf_key = f"tf_{tf}"
            results[tf_key] = {
                'value': np.mean(tf_data[-20:]) if len(tf_data) >= 20 else np.mean(tf_data),
                'trend': self._calculate_trend(tf_data),
                'volatility': np.std(tf_data) if len(tf_data) > 1 else 0,
                'weight': weight
            }
            all_values.extend(tf_data[-5:])  # Беремо останні 5 точок
        
        if not results:
            return {'convergence_score': 0, 'confirmed': False, 'details': {}}
        
        # Перевірка конвергенції
        convergence_score = self._calculate_convergence_score(results)
        confirmed = self._check_signal_confirmation(results)
        
        return {
            'convergence_score': convergence_score,
            'confirmed': confirmed,
            'details': results,
            'composite_value': np.mean(all_values) if all_values else 0
        }
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Розрахунок тренду лінійної регресії"""
        if len(data) < 5:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope / np.mean(y) * 100 if np.mean(y) != 0 else 0
        except:
            return 0.0
    
    def _calculate_convergence_score(self, results: Dict) -> float:
        """Розрахунок оцінки конвергенції"""
        if len(results) < 2:
            return 0.0
        
        # Перевіряємо напрямок на всіх таймфреймах
        directions = []
        for tf_data in results.values():
            direction = 1 if tf_data['trend'] > 0.1 else (-1 if tf_data['trend'] < -0.1 else 0)
            directions.append(direction)
        
        # Якщо всі таймфрейми в одному напрямку - сильна конвергенція
        if all(d == 1 for d in directions):
            return 1.0
        elif all(d == -1 for d in directions):
            return 1.0
        elif sum(directions) >= 2:  # Більшість в одному напрямку
            return 0.7
        else:
            return 0.3
    
    def _check_signal_confirmation(self, results: Dict) -> bool:
        """Перевірка підтвердження сигналу на всіх таймфреймах"""
        if len(results) < 2:
            return False
        
        # Логіка: якщо короткий та середній таймфрейми підтверджують довгий
        tf_keys = list(results.keys())
        if len(tf_keys) >= 3:
            short = results[tf_keys[0]]['trend']
            medium = results[tf_keys[1]]['trend']
            long = results[tf_keys[2]]['trend']
            
            # Конвергенція: всі тренди в одному напрямку
            if (short > 0 and medium > 0 and long > 0) or (short < 0 and medium < 0 and long < 0):
                return True
            
            # Алігмент: короткий і середній підтверджують довгий
            if abs(long) > 0.2 and ((short > 0 and medium > 0) or (short < 0 and medium < 0)):
                return True
        
        return False

class SignalQualityMonitor:
    def __init__(self):
        self.signal_history = []
        self.performance_metrics = {
            "total_signals": 0,
            "correct_predictions": 0,
            "strong_correct": 0,
            "total_strong": 0
        }
    
    def track_signal_quality(self, signal: Dict, actual_price_movement: float):
        """Відстеження ефективності сигналів"""
        quality_metric = {
            "timestamp": time.time(),
            "symbol": signal.get("symbol", "UNKNOWN"),
            "signal_strength": signal.get("strength", 0),
            "predicted_direction": signal.get("action", "HOLD"),
            "actual_movement": actual_price_movement,
            "correct_prediction": (
                (signal.get("action") == "BUY" and actual_price_movement > 0) or
                (signal.get("action") == "SELL" and actual_price_movement < 0)
            )
        }
        
        self.signal_history.append(quality_metric)
        self.performance_metrics["total_signals"] += 1
        
        if quality_metric["correct_prediction"]:
            self.performance_metrics["correct_predictions"] += 1
            
            if signal.get("strength", 0) >= 3:
                self.performance_metrics["strong_correct"] += 1
                self.performance_metrics["total_strong"] += 1
        elif signal.get("strength", 0) >= 3:
            self.performance_metrics["total_strong"] += 1
        
        if len(self.signal_history) % 20 == 0:
            self.log_performance_stats()
    
    def log_performance_stats(self):
        """Логування статистики ефективності сигналів"""
        if self.performance_metrics["total_signals"] > 0:
            accuracy = (self.performance_metrics["correct_predictions"] / 
                       self.performance_metrics["total_signals"]) * 100
            
            strong_accuracy = 0
            if self.performance_metrics["total_strong"] > 0:
                strong_accuracy = (self.performance_metrics["strong_correct"] / 
                                 self.performance_metrics["total_strong"]) * 100
            
            logger.info(f"[QUALITY] Signal Accuracy: {accuracy:.1f}% "
                       f"({self.performance_metrics['correct_predictions']}/{self.performance_metrics['total_signals']}) | "
                       f"Strong Signals: {strong_accuracy:.1f}%")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Отримання звіту про ефективність"""
        if self.performance_metrics["total_signals"] == 0:
            return {"accuracy": 0, "strong_accuracy": 0}
        
        accuracy = (self.performance_metrics["correct_predictions"] / 
                   self.performance_metrics["total_signals"]) * 100
        
        strong_accuracy = 0
        if self.performance_metrics["total_strong"] > 0:
            strong_accuracy = (self.performance_metrics["strong_correct"] / 
                             self.performance_metrics["total_strong"]) * 100
        
        return {
            "total_signals": self.performance_metrics["total_signals"],
            "accuracy_percent": round(accuracy, 1),
            "strong_signals_accuracy": round(strong_accuracy, 1),
            "strong_signals_count": self.performance_metrics["total_strong"]
        }

class SpreadMonitor:
    """O'HARA METHOD 7: Spread as Risk Measure"""
    
    def __init__(self):
        self.cfg = settings.spread
        self._spread_history = {}
        self._spread_baseline = {}
    
    def update(self, symbol: str, bid: float, ask: float):
        """Оновлення spread даних"""
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        
        spread_abs = ask - bid
        spread_bps = (spread_abs / bid) * 10000
        
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=self.cfg.spread_history_size)
        
        self._spread_history[symbol].append({
            'timestamp': time.time(),
            'spread_bps': spread_bps
        })
        
        if len(self._spread_history[symbol]) >= 10:
            avg_spread = sum(s['spread_bps'] for s in self._spread_history[symbol]) / len(self._spread_history[symbol])
            self._spread_baseline[symbol] = avg_spread
    
    def get_risk_level(self, symbol: str, current_spread_bps: float) -> Dict[str, Any]:
        """Визначення рівня ризику на основі spread"""
        if symbol not in self._spread_baseline or len(self._spread_history.get(symbol, [])) < 10:
            return {
                'risk_level': 'UNKNOWN',
                'current_spread_bps': round(current_spread_bps, 2),
                'avg_spread_bps': 0,
                'spread_ratio': 1.0,
                'should_avoid': False,
                'should_reduce_size': False
            }
        
        avg_spread = self._spread_baseline[symbol]
        
        if avg_spread == 0:
            spread_ratio = 1.0
        else:
            spread_ratio = current_spread_bps / avg_spread
        
        if spread_ratio >= self.cfg.very_high_risk_spread_multiplier:
            risk_level = "VERY_HIGH_RISK"
            should_avoid = self.cfg.avoid_trading_on_very_high_spread
            should_reduce_size = True
        elif spread_ratio >= self.cfg.high_risk_spread_multiplier:
            risk_level = "HIGH_RISK"
            should_avoid = False
            should_reduce_size = self.cfg.reduce_size_on_high_spread
        elif current_spread_bps > self.cfg.max_spread_threshold_bps:
            risk_level = "ELEVATED"
            should_avoid = False
            should_reduce_size = False
        else:
            risk_level = "NORMAL"
            should_avoid = False
            should_reduce_size = False
        
        return {
            'risk_level': risk_level,
            'current_spread_bps': round(current_spread_bps, 2),
            'avg_spread_bps': round(avg_spread, 2),
            'spread_ratio': round(spread_ratio, 2),
            'should_avoid': should_avoid,
            'should_reduce_size': should_reduce_size,
            'size_reduction_pct': self.cfg.high_spread_size_reduction_pct if should_reduce_size else 0
        }

class SignalGenerator:
    def __init__(self):
        self.cfg = settings.signals
        self.ohara_cfg = settings.ohara
        
        # 🆕 Додаємо відсутні атрибути для сумісності
        self._ensure_mtf_attributes()
        
        self._state = {}
        self.quality_monitor = SignalQualityMonitor()
        self.spread_monitor = SpreadMonitor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()  # 🆕 Багатотаймфреймовий аналізатор

    def _ensure_mtf_attributes(self):
        """Забезпечуємо наявність необхідних MTF атрибутів"""
        # MTF фільтри
        if not hasattr(self.cfg, 'enable_mtf_filter'):
            self.cfg.enable_mtf_filter = True
        if not hasattr(self.cfg, 'mtf_convergence_threshold'):
            self.cfg.mtf_convergence_threshold = 0.3
        if not hasattr(self.cfg, 'min_mtf_timeframes_confirmed'):
            self.cfg.min_mtf_timeframes_confirmed = 2
        if not hasattr(self.cfg, 'mtf_require_confirmation_for_entry'):
            self.cfg.mtf_require_confirmation_for_entry = True
        if not hasattr(self.cfg, 'mtf_allow_override_on_strong_signal'):
            self.cfg.mtf_allow_override_on_strong_signal = True
        if not hasattr(self.cfg, 'mtf_override_strength_threshold'):
            self.cfg.mtf_override_strength_threshold = 4
        if not hasattr(self.cfg, 'enable_multi_timeframe_analysis'):
            self.cfg.enable_multi_timeframe_analysis = True
        if not hasattr(self.cfg, 'mtf_confirmation_boost'):
            self.cfg.mtf_confirmation_boost = 1.2
        if not hasattr(self.cfg, 'mtf_weight_1min'):
            self.cfg.mtf_weight_1min = 0.4
        if not hasattr(self.cfg, 'mtf_weight_5min'):
            self.cfg.mtf_weight_5min = 0.35
        if not hasattr(self.cfg, 'mtf_weight_30min'):
            self.cfg.mtf_weight_30min = 0.25

    def _init_symbol(self, symbol: str):
        if symbol not in self._state:
            self._state[symbol] = {
                "ema_score": 0.0,
                "last_action": "HOLD",
                "last_strength": 0,
                "cooldown_until": 0.0,
                "last_update": 0.0,
                "mtf_scores": {60: 0.0, 300: 0.0, 1800: 0.0},  # 🆕 MTF scores
                "mtf_confirmation": False
            }

    def generate(self, symbol: str, imbalance_data: Dict, volume_data: Dict, spread_bps: float = None) -> Dict[str, Any]:
        self._init_symbol(symbol)
        st = self._state[symbol]

        # 🆕 Збір даних для мультитаймфреймового аналізу
        current_time = time.time()
        
        # Імбаланс на різних таймфреймах
        imb_score = imbalance_data.get("effective_imbalance", imbalance_data.get("imbalance_score", 0))
        self.mtf_analyzer.add_data_point(symbol, "imbalance", imb_score, current_time)
        
        # Моментум на різних таймфреймах
        mom_score = volume_data.get("momentum_score", 0)
        self.mtf_analyzer.add_data_point(symbol, "momentum", mom_score, current_time)
        
        # Волатильність на різних таймфреймах
        volatility = volume_data.get("volatility", 0)
        self.mtf_analyzer.add_data_point(symbol, "volatility", volatility, current_time)
        
        # 🆕 Отримання конвергенції сигналів
        imb_convergence = self.mtf_analyzer.calculate_convergence(symbol, "imbalance")
        mom_convergence = self.mtf_analyzer.calculate_convergence(symbol, "momentum")
        
        # Логування MTF інформації
        logger.debug(f"[MTF_ANALYSIS] {symbol}: "
                    f"Imbalance Conv={imb_convergence['convergence_score']:.2f}, "
                    f"Momentum Conv={mom_convergence['convergence_score']:.2f}")
        
        # Розрахунок MTF-зваженого сигналу
        mtf_imbalance = self._calculate_mtf_weighted_imbalance(symbol, imb_convergence)
        mtf_momentum = self._calculate_mtf_weighted_momentum(symbol, mom_convergence)
        
        # Оригінальні розрахунки
        tape_analysis = volume_data.get("tape_analysis", {})
        depth_analysis = imbalance_data.get("depth_analysis", {})
        cluster_analysis = imbalance_data.get("cluster_analysis", {})
        
        bayesian_data = imbalance_data.get("bayesian_data", {})
        trade_imbalance = imbalance_data.get("trade_imbalance", {})
        frequency_data = volume_data.get("frequency_data", {})
        volume_confirm = volume_data.get("volume_confirmation", {})
        large_order_data = volume_data.get("large_order_data", {})
        
        adaptive_volume = volume_data.get("adaptive_volume_analysis", {})
        adaptive_stats = volume_data.get("adaptive_statistics", {})
        
        # 🆕 Заміна оригінальних значень на MTF-зважені
        factors = self._calculate_all_factors(
            mtf_imbalance,  # Використовуємо MTF імбаланс
            mtf_momentum,   # Використовуємо MTF моментум
            tape_analysis, depth_analysis, cluster_analysis, 
            spread_bps, volatility,
            bayesian_data, trade_imbalance, frequency_data, 
            volume_confirm, large_order_data
        )
        
        # 🆕 Додаємо MTF конвергенцію як додатковий фактор
        factors["mtf_convergence"] = (imb_convergence['convergence_score'] + 
                                     mom_convergence['convergence_score']) / 2
        
        composite_score = self._calculate_composite_score(factors)
        
        # 🆕 Застосування MTF-згладжування
        ema_score = self._apply_mtf_smoothing(symbol, composite_score, imb_convergence, mom_convergence)
        
        action, strength, reason = self._generate_action_strength(symbol, ema_score, volume_data, factors)
        
        if self._is_in_cooldown(symbol, action, strength):
            return self._create_cooldown_response(symbol, action, strength, ema_score, factors)
        
        self._update_state(symbol, action, strength)
        
        result = self._create_signal_response(
            symbol, action, strength, ema_score, composite_score, factors, reason
        )
        
        # 🆕 Додаємо MTF дані до результату
        result["mtf_data"] = {
            "imbalance_convergence": imb_convergence,
            "momentum_convergence": mom_convergence,
            "mtf_imbalance": mtf_imbalance,
            "mtf_momentum": mtf_momentum,
            "confirmed": imb_convergence['confirmed'] and mom_convergence['confirmed']
        }
        
        result["adaptive_volume"] = adaptive_volume
        result["adaptive_stats"] = adaptive_stats
        
        try:
            raw_values = factors.get("raw_values", {})
            signal_logger.log_signal(
                symbol=symbol,
                action=result["action"],
                strength=result["strength"],
                composite=composite_score,
                ema=ema_score,
                imbalance=raw_values.get("imbalance_score", 0.0),
                momentum=raw_values.get("momentum_score", 0.0),
                bayesian=raw_values.get("bayesian_signal", "NEUTRAL"),
                large_orders=raw_values.get("informed_direction", "NEUTRAL"),
                frequency=raw_values.get("activity_level", "UNKNOWN"),
                vol_confirm=raw_values.get("vol_confirmation", "UNKNOWN"),
                ohara_score=result.get("ohara_score", 0),
                mtf_convergence=factors.get("mtf_convergence", 0),  # 🆕
                reason=result.get("reason", "ok"),
                accepted=(result["strength"] >= self.cfg.min_strength_for_action)
            )
        except Exception as e:
            logger.warning(f"[SIGNAL_LOGGER] Failed to log signal for {symbol}: {e}")
        
        self._log_signal_generation(symbol, result, factors)
        return result

    def debug_signal_calculation(self, symbol: str, imb_data: Dict, vol_data: Dict, spread_bps: float = None):
        self._init_symbol(symbol)

        imb_score = imb_data.get("effective_imbalance", 0)
        mom_score = vol_data.get("momentum_score", 0)
        volatility = vol_data.get("volatility", 0)
        
        tape_analysis = vol_data.get("tape_analysis", {})
        depth_analysis = imb_data.get("depth_analysis", {})
        cluster_analysis = imb_data.get("cluster_analysis", {})
        
        bayesian_data = imb_data.get("bayesian_data", {})
        trade_imbalance = imb_data.get("trade_imbalance", {})
        frequency_data = vol_data.get("frequency_data", {})
        volume_confirm = vol_data.get("volume_confirmation", {})
        large_order_data = vol_data.get("large_order_data", {})
        
        factors = self._calculate_all_factors(
            imb_score, mom_score, tape_analysis, depth_analysis,
            cluster_analysis, spread_bps, volatility,
            bayesian_data, trade_imbalance, frequency_data,
            volume_confirm, large_order_data
        )
        
        composite_score = self._calculate_composite_score(factors)
        ema_score = self._apply_smoothing(symbol, composite_score)
        
        logger.info(f"🔍 [SIGNAL_DEBUG] {symbol}: ")
        logger.info(f"   - Imbalance: {imb_score:.1f} -> factor: {factors['imbalance']:.3f}")
        logger.info(f"   - Momentum: {mom_score:.1f} -> factor: {factors['momentum']:.3f}") 
        logger.info(f"   - Bayesian: {bayesian_data.get('signal')} -> factor: {factors['ohara_bayesian']:.3f}")
        logger.info(f"   - Large Orders: {large_order_data.get('informed_direction')} -> factor: {factors['ohara_large_orders']:.3f}")
        logger.info(f"   - Frequency: {frequency_data.get('activity_level')} -> factor: {factors['ohara_frequency']:.3f}")
        logger.info(f"   - Vol Confirm: {volume_confirm.get('confirmation')} -> factor: {factors['ohara_volume_confirm']:.3f}")
        logger.info(f"   - Composite: {composite_score:.3f}, EMA: {ema_score:.3f}")
        
        action, strength, reason = self._generate_action_strength(symbol, ema_score, vol_data, factors)
        logger.info(f"   - Result: {action}{strength}, reason: {reason}")
        
        return action, strength

    def _calculate_mtf_weighted_imbalance(self, symbol: str, convergence_data: Dict) -> float:
        """Розрахунок MTF-зваженого імбалансу"""
        details = convergence_data.get('details', {})
        if not details:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tf_key, tf_data in details.items():
            weight = tf_data.get('weight', self.cfg.mtf_weight_1min)
            value = tf_data.get('value', 0.0)
            trend = tf_data.get('trend', 0.0)
            
            # Вага збільшується, якщо висока конвергенція
            convergence_bonus = 1.0 + (convergence_data['convergence_score'] * 0.5)
            adjusted_weight = weight * convergence_bonus
            
            # Комбінуємо значення та тренд
            combined_value = value * 0.7 + trend * 0.3
            
            weighted_sum += combined_value * adjusted_weight
            total_weight += adjusted_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_mtf_weighted_momentum(self, symbol: str, convergence_data: Dict) -> float:
        """Розрахунок MTF-зваженого моментуму"""
        details = convergence_data.get('details', {})
        if not details:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for tf_key, tf_data in details.items():
            weight = tf_data.get('weight', self.cfg.mtf_weight_1min)
            value = tf_data.get('value', 0.0)
            trend = tf_data.get('trend', 0.0)
            volatility = tf_data.get('volatility', 1.0)
            
            # Корекція на волатильність
            volatility_factor = 1.0 / (1.0 + volatility * 0.1)
            adjusted_weight = weight * volatility_factor
            
            # Комбінуємо значення з трендом
            combined_value = value * 0.6 + trend * 0.4
            
            weighted_sum += combined_value * adjusted_weight
            total_weight += adjusted_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _apply_mtf_smoothing(self, symbol: str, current_score: float, 
                           imb_convergence: Dict, mom_convergence: Dict) -> float:
        """MTF-згладжування з урахуванням конвергенції"""
        st = self._state[symbol]
        
        # Базова EMA
        ema_prev = st.get("ema_score", 0.0)
        base_alpha = self.cfg.smoothing_alpha
        
        # Корекція альфи на основі конвергенції
        convergence_score = (imb_convergence['convergence_score'] + 
                           mom_convergence['convergence_score']) / 2
        
        if convergence_score > 0.8:
            # Висока конвергенція - швидше оновлення
            adaptive_alpha = min(base_alpha * 1.5, 0.9)
        elif convergence_score > 0.5:
            adaptive_alpha = base_alpha
        else:
            # Низька конвергенція - повільніше оновлення
            adaptive_alpha = max(base_alpha * 0.5, 0.1)
        
        ema_new = ema_prev * (1 - adaptive_alpha) + current_score * adaptive_alpha
        st["ema_score"] = ema_new
        
        # Оновлюємо MTF scores
        for tf in [60, 300, 1800]:
            mtf_key = f"mtf_score_{tf}"
            if mtf_key not in st:
                st[mtf_key] = 0.0
            st[mtf_key] = st[mtf_key] * 0.9 + current_score * 0.1
        
        return ema_new

    def _apply_smoothing(self, symbol: str, current_score: float) -> float:
        """Зворотна сумісність оригінального методу"""
        st = self._state[symbol]
        ema_prev = st.get("ema_score", 0.0)
        ema_new = ema_prev * (1 - self.cfg.smoothing_alpha) + current_score * self.cfg.smoothing_alpha
        st["ema_score"] = ema_new
        return ema_new

    def _calculate_all_factors(self, imb_score, mom_score, tape_analysis, 
                             depth_analysis, cluster_analysis, spread_bps, volatility,
                             bayesian_data, trade_imbalance, frequency_data,
                             volume_confirm, large_order_data):
        # Нормалізація
        imb_norm = imb_score / 100.0
        mom_norm = mom_score / 100.0
        
        tape_factor = self._calculate_tape_factor(tape_analysis)
        depth_factor = self._calculate_depth_factor(depth_analysis)
        cluster_factor = self._calculate_cluster_factor(cluster_analysis)
        spread_factor = self._calculate_spread_factor(spread_bps)
        volatility_factor = self._calculate_volatility_factor(volatility)
        
        bayesian_factor = self._calculate_bayesian_factor(bayesian_data)
        large_order_factor = self._calculate_large_order_factor(large_order_data)
        frequency_factor = self._calculate_frequency_factor(frequency_data)
        volume_confirm_factor = self._calculate_volume_confirm_factor(volume_confirm)
        
        return {
            "imbalance": imb_norm,
            "momentum": mom_norm,
            "tape": tape_factor,
            "depth": depth_factor,
            "cluster": cluster_factor,
            "spread": spread_factor,
            "volatility": volatility_factor,
            "ohara_bayesian": bayesian_factor,
            "ohara_large_orders": large_order_factor,
            "ohara_frequency": frequency_factor,
            "ohara_volume_confirm": volume_confirm_factor,
            "large_orders": large_order_data,
            "raw_values": {
                "imbalance_score": imb_score,
                "momentum_score": mom_score,
                "large_net": tape_analysis.get("large_net", 0),
                "volume_acceleration": tape_analysis.get("volume_acceleration", 0),
                "support_ratio": depth_analysis.get("support_resistance_ratio", 0.5),
                "poc_distance": cluster_analysis.get("poc_distance_pct", 0),
                "bayesian_signal": bayesian_data.get("signal", "NEUTRAL"),
                "informed_direction": large_order_data.get("informed_direction", "NEUTRAL"),
                "activity_level": frequency_data.get("activity_level", "UNKNOWN"),
                "vol_confirmation": volume_confirm.get("confirmation", "UNKNOWN")
            }
        }

    def _calculate_bayesian_factor(self, bayesian_data: Dict) -> float:
        """O'HARA METHOD 1: Bayesian factor"""
        signal = bayesian_data.get("signal", "NEUTRAL")
        confidence = bayesian_data.get("confidence", 0)
        
        if signal == "BULLISH":
            return confidence * 0.5
        elif signal == "BEARISH":
            return -confidence * 0.5
        else:
            return 0.0

    def _calculate_large_order_factor(self, large_order_data: Dict) -> float:
        """O'HARA METHOD 2: Large orders factor"""
        direction = large_order_data.get("informed_direction", "NEUTRAL")
        count = large_order_data.get("count", 0)
        
        base_strong = 0.5
        base_medium = 0.25
        
        count_bonus = 0
        if count > self.cfg.large_order_count_bonus_threshold:
            count_bonus = min(
                self.cfg.large_order_count_bonus_max, 
                count * self.cfg.large_order_count_bonus_per_order
            )
        
        if direction == "STRONG_BUY":
            return base_strong + count_bonus
        elif direction == "MEDIUM_BUY":
            return base_medium + count_bonus * 0.5
        elif direction == "STRONG_SELL":
            return -(base_strong + count_bonus)
        elif direction == "MEDIUM_SELL":
            return -(base_medium + count_bonus * 0.5)
        else:
            return 0.0

    def _calculate_frequency_factor(self, frequency_data: Dict) -> float:
        """O'HARA METHOD 3: Trade frequency factor"""
        activity_level = frequency_data.get("activity_level", "UNKNOWN")
        risk_signal = frequency_data.get("risk_signal", "OK")
        
        if risk_signal == "AVOID":
            return -0.5
        elif risk_signal == "CAUTION":
            return -0.2
        elif risk_signal == "LOW_LIQUIDITY":
            return -0.3
        else:
            return 0.0

    def _calculate_volume_confirm_factor(self, volume_confirm: Dict) -> float:
        """O'HARA METHOD 5: Volume confirmation factor"""
        confirmation = volume_confirm.get("confirmation", "UNKNOWN")
        strength = volume_confirm.get("strength", "WEAK")
        
        if confirmation == "CONFIRMED" and strength == "STRONG":
            return 0.3
        elif confirmation == "MODERATE":
            return 0.1
        elif confirmation == "WEAK":
            return -0.2
        else:
            return 0.0

    def _calculate_tape_factor(self, tape_analysis: Dict) -> float:
        large_net = tape_analysis.get("large_net", 0)
        volume_acceleration = tape_analysis.get("volume_acceleration", 0)
        absorption_bullish = tape_analysis.get("absorption_bullish", False)
        absorption_bearish = tape_analysis.get("absorption_bearish", False)
        
        factor = 0.0
        
        if abs(large_net) > 0:
            factor += large_net / 5.0 * 0.4
        
        if abs(volume_acceleration) > 10:
            factor += volume_acceleration / 100.0 * 0.3
        
        if absorption_bullish:
            factor += 0.2
        if absorption_bearish:
            factor -= 0.2
        
        return max(-1.0, min(1.0, factor))

    def _calculate_depth_factor(self, depth_analysis: Dict) -> float:
        support_ratio = depth_analysis.get("support_resistance_ratio", 0.5)
        liquidity_imb = depth_analysis.get("liquidity_imbalance", 0)
        
        depth_factor = (support_ratio - 0.5) * 2
        liquidity_factor = liquidity_imb / 100.0 * 0.5
        
        return max(-1.0, min(1.0, depth_factor + liquidity_factor))

    def _calculate_cluster_factor(self, cluster_analysis: Dict) -> float:
        poc_distance = cluster_analysis.get("poc_distance_pct", 0)
        
        if poc_distance > 1.0:
            return -0.3
        elif poc_distance < -1.0:
            return 0.3
        else:
            return 0.0

    def _calculate_spread_factor(self, spread_bps: float) -> float:
        """O'HARA METHOD 7: Spread factor"""
        if spread_bps is None:
            return 0.0
        
        max_spread = settings.spread.max_spread_threshold_bps
        if spread_bps > max_spread * 2:
            return -0.5
        elif spread_bps > max_spread:
            return -0.2
        else:
            return 0.0

    def _calculate_volatility_factor(self, volatility: float) -> float:
        if volatility > 5.0:
            return -0.3
        elif volatility > 2.0:
            return -0.1
        else:
            return 0.0

    def _calculate_composite_score(self, factors: Dict) -> float:
        """Розрахунок composite score з O'Hara факторами"""
        score = (
            factors["imbalance"] * self.cfg.weight_imbalance +
            factors["momentum"] * self.cfg.weight_momentum +
            factors["ohara_bayesian"] * self.cfg.weight_ohara_bayesian +
            factors["ohara_large_orders"] * self.cfg.weight_ohara_large_orders +
            factors["ohara_frequency"] * self.cfg.weight_ohara_frequency +
            factors["ohara_volume_confirm"] * self.cfg.weight_ohara_volume_confirm +
            factors["tape"] * 0.15 +
            factors["depth"] * 0.1 +
            factors["cluster"] * 0.05 +
            factors["spread"] * 0.075 +
            factors["volatility"] * 0.05
        )
        
        # 🆕 Додаємо MTF конвергенцію до score
        if "mtf_convergence" in factors:
            mtf_bonus = factors["mtf_convergence"] * 0.1
            score += mtf_bonus
        
        if factors["raw_values"]["volume_acceleration"] > 50:
            score += self.cfg.spike_bonus
        
        return max(-1.0, min(1.0, score))

    def _get_adaptive_threshold(self, symbol: str, volume_data: Dict, factors: Dict) -> float:
        """Calculate adaptive threshold based on market conditions"""
        if not self.cfg.enable_adaptive_threshold:
            return self.cfg.composite_thresholds["strength_3"]
        
        base = self.cfg.base_threshold
        adjustment = 0.0
        
        volatility = volume_data.get("volatility", 1.0)
        if volatility >= self.cfg.volatility_high_level:
            adjustment -= self.cfg.high_volatility_threshold_reduction
        elif volatility <= self.cfg.volatility_low_level:
            adjustment += self.cfg.low_volatility_threshold_increase
        
        adaptive_volume = volume_data.get("adaptive_volume_analysis", {})
        vol_zscore = adaptive_volume.get("zscore", 0)
        
        if vol_zscore > 1.0:
            adjustment -= self.cfg.high_liquidity_threshold_reduction
        elif vol_zscore < -0.5:
            adjustment += self.cfg.low_liquidity_threshold_increase
        
        ohara_score = self._calculate_ohara_score(factors)
        if ohara_score >= self.cfg.ohara_strong_score_threshold:
            adjustment -= self.cfg.ohara_threshold_reduction
        
        # 🆕 MTF корекція порогу
        mtf_convergence = factors.get("mtf_convergence", 0)
        if mtf_convergence > 0.7:
            adjustment -= 0.05  # Знижуємо поріг при високій конвергенції
        elif mtf_convergence < 0.3:
            adjustment += 0.03  # Підвищуємо поріг при низькій конвергенції
        
        final_threshold = max(self.cfg.min_threshold, min(self.cfg.max_threshold, base + adjustment))
        
        return final_threshold

    def _check_whale_entry_conditions(
        self, 
        symbol: str, 
        direction: str,
        momentum_pct: float,
        imbalance_pct: float,
        ohara_score: int,
        large_order_data: Dict
    ) -> Tuple[bool, str]:
        """
        🐋 Перевірка умов входу - ПОМ'ЯКШЕНА ВЕРСІЯ
        """
        
        # 1.Momentum перевірка
        min_mom = getattr(self.cfg, 'min_momentum_for_entry', 45.0)  # було 60
        max_mom = getattr(self.cfg, 'max_momentum_for_entry', 92.0)  # було 88
        
        if abs(momentum_pct) < min_mom:
            return False, f"momentum_too_low ({abs(momentum_pct):.0f}% < {min_mom}%)"
        
        if abs(momentum_pct) > max_mom:
            return False, f"momentum_too_high ({abs(momentum_pct):.0f}% > {max_mom}%)"
        
        # 2.Imbalance перевірка
        min_imb = getattr(self.cfg, 'min_imbalance_for_entry', 8.0)  # було 18! 
        if abs(imbalance_pct) < min_imb:
            return False, f"imbalance_too_low ({abs(imbalance_pct):.0f}% < {min_imb}%)"
        
        # 3.O'Hara score перевірка
        min_ohara = getattr(self.cfg, 'min_ohara_for_entry', 4)  # було 5
        if ohara_score < min_ohara:
            return False, f"ohara_too_low ({ohara_score} < {min_ohara})"
        
        # 4.Large Orders - тільки перевірка напрямку (кількість опціонально)
        lo_direction = large_order_data.get("informed_direction", "NEUTRAL")
        
        # НЕ йдемо проти великих ордерів
        if direction in ["BUY", "LONG"]:
            if lo_direction in ["STRONG_SELL"]:  # тільки STRONG, не MEDIUM
                return False, f"large_orders_against_buy ({lo_direction})"
        else:
            if lo_direction in ["STRONG_BUY"]:
                return False, f"large_orders_against_sell ({lo_direction})"
        
        # ✅ Пройшли всі перевірки
        logger.info(
            f"🐋 [WHALE_OK] {symbol} {direction}: "
            f"mom={momentum_pct:.0f}%, imb={imbalance_pct:.0f}%, ohara={ohara_score}"
        )
        return True, "whale_confirmed"

    def _generate_action_strength(
        self, 
        symbol: str, 
        ema_score: float, 
        volume_data: Dict, 
        factors: Dict
    ) -> Tuple[str, int, str]:
        abs_score = abs(ema_score)
        direction = "BUY" if ema_score > 0 else "SELL"
        raw_values = factors.get("raw_values", {})
        
        # 🆕 MTF ФІЛЬТРАЦІЯ
        mtf_convergence = factors.get("mtf_convergence", 0)
        mtf_confirmed = factors.get("mtf_data", {}).get("confirmed", False)
        
        # Перевірка MTG фільтру
        if self.cfg.enable_mtf_filter and not mtf_confirmed:
            # Перевіряємо чи достатньо сильний сигнал для override
            if abs_score < self.cfg.composite_thresholds["strength_4"]:
                logger.debug(
                    f"[MTF_FILTER] {symbol}: No MTF confirmation, "
                    f"signal too weak (score={abs_score:.3f})"
                )
                return "HOLD", 0, "no_mtf_confirmation"
            elif not self.cfg.mtf_allow_override_on_strong_signal:
                logger.debug(
                    f"[MTF_FILTER] {symbol}: No MTF confirmation, "
                    f"override not allowed"
                )
                return "HOLD", 0, "no_mtf_confirmation"
            elif self.cfg.mtf_allow_override_on_strong_signal and abs_score >= self.cfg.composite_thresholds["strength_5"]:
                logger.info(
                    f"[MTF_OVERRIDE] {symbol}: Very strong signal ({abs_score:.3f}), "
                    f"overriding MTF filter"
                )
                # Дозволяємо вхід при дуже сильному сигналі
        
        if self.cfg.enable_volume_validation:
            activity_level = raw_values.get("activity_level", "UNKNOWN")
            
            if activity_level == "VERY_HIGH":
                if abs_score < 0.50:
                    logger.debug(
                        f"[OHARA_FILTER] {symbol}: VERY_HIGH activity detected - "
                        f"avoiding trade (score={abs_score:.3f})"
                    )
                    return "HOLD", 0, "very_high_activity"
                else:
                    logger.info(
                        f"[OHARA_OVERRIDE] {symbol}: VERY_HIGH activity BUT strong signal "
                        f"(score={abs_score:.2f}) - allowing trade"
                    )
            
            elif activity_level == "VERY_LOW":
                logger.debug(
                    f"[OHARA_FILTER] {symbol}: VERY_LOW activity - "
                    f"low liquidity, avoiding trade"
                )
                return "HOLD", 0, "very_low_activity"
        
        action = "HOLD"
        strength = 0
        reason = "weak_signal"
        
        min_threshold = self._get_adaptive_threshold(symbol, volume_data, factors)
        
        if abs_score >= self.cfg.composite_thresholds["strength_5"]:
            strength = 5
        elif abs_score >= self.cfg.composite_thresholds["strength_4"]:
            strength = 4
        elif abs_score >= self.cfg.composite_thresholds["strength_3"]:
            strength = 3
        elif abs_score >= self.cfg.composite_thresholds["strength_2"]:
            strength = 2
        elif abs_score >= self.cfg.composite_thresholds["strength_1"]:
            strength = 1
        
        # 🆕 MTF підвищення сили при конвергенції
        if mtf_confirmed and mtf_convergence > 0.7:
            strength = min(5, strength + 1)  # +1 до сили
            logger.debug(f"[MTF_BOOST] {symbol}: Strength boosted to {strength} due to MTF convergence")
        
        if abs_score >= min_threshold and strength >= self.cfg.min_strength_for_action:
            # =====================================================
            # 🐋 WHALE STRATEGY: Перевірка перед входом
            # =====================================================
            momentum_pct = raw_values.get("momentum_score", 0)
            imbalance_pct = raw_values.get("imbalance_score", 0)
            ohara_score = self._calculate_ohara_score(factors)
            large_order_data = factors.get("large_orders", {})
            
            can_enter, whale_reason = self._check_whale_entry_conditions(
                symbol=symbol,
                direction=direction,
                momentum_pct=momentum_pct,
                imbalance_pct=imbalance_pct,
                ohara_score=ohara_score,
                large_order_data=large_order_data
            )
            
            if can_enter:
                action = direction
                reason = whale_reason
                logger.info(
                    f"🐋 [WHALE_SIGNAL] {symbol} {action}{strength}: "
                    f"composite={abs_score:.3f}, MTF={mtf_convergence:.2f}, {whale_reason}"
                )
            else:
                # Блокуємо вхід - не пройшли whale перевірку
                action = "HOLD"
                strength = 0
                reason = whale_reason
                logger.info(
                    f"🚫 [WHALE_BLOCKED] {symbol}: {whale_reason} "
                    f"(composite={abs_score:.3f} passed, but whale check failed)"
                )
        else:
            if abs_score < min_threshold:
                reason = "below_threshold"
            elif strength < self.cfg.min_strength_for_action:
                reason = "weak_signal"
        
        # Late entry check
        if action != "HOLD":
            momentum_pct = abs(raw_values.get("momentum_score", 0))
            ohara_score = self._calculate_ohara_score(factors)
            
            if momentum_pct > self.cfg.late_entry_momentum_threshold:
                logger.warning(
                    f"[LATE_ENTRY] {symbol}: Extreme momentum {momentum_pct:.1f}% - rejecting signal"
                )
                action = "HOLD"
                strength = 0
                reason = "late_entry"
            elif momentum_pct > self.cfg.late_entry_high_momentum_threshold:
                if self.cfg.late_entry_allow_strong_trend and ohara_score >= self.cfg.late_entry_min_ohara_for_override:
                    logger.info(
                        f"[LATE_ENTRY_ALLOWED] {symbol}: momentum={momentum_pct:.1f}% but O'Hara={ohara_score}/10 - "
                        f"allowing with {self.cfg.late_entry_position_size_reduction*100:.0f}% position size"
                    )
                    reason = "late_entry_reduced"
                else:
                    logger.warning(
                        f"[LATE_ENTRY] {symbol}: High momentum {momentum_pct:.1f}% (O'Hara={ohara_score}/10 too low) - rejecting"
                    )
                    action = "HOLD"
                    strength = 0
                    reason = "late_entry"
        
        if action != "HOLD":
            logger.info(
                f"[ACTION] {symbol} → {action}{strength} "
                f"(composite={abs_score:.3f}, MTF={mtf_convergence:.2f}, reason={reason})"
            )
        
        return action, strength, reason

    def _validate_signal_consistency(self, action: str, strength: int, raw_values: Dict) -> str:
        """Перевірка узгодженості сигналів"""
        imbalance_score = raw_values.get("imbalance_score", 0)
        
        max_contradiction = self.cfg.max_imbalance_contradiction
        if action == "BUY" and imbalance_score < -max_contradiction:
            return "contradictory_imbalance_buy"
        if action == "SELL" and imbalance_score > max_contradiction:
            return "contradictory_imbalance_sell"
        
        return "ok"

    def _validate_signal_quality(self, symbol: str, action: str, factors: Dict, volume_data: Dict) -> str:
        """Валідація якості сигналу"""
        raw_values = factors.get("raw_values", {})
        
        volatility = volume_data.get("volatility", 0)
        if volatility < self.cfg.volatility_filter_threshold:
            return "low_volatility"
        
        if not self.cfg.allow_override_contradictory_orders:
            informed_direction = raw_values.get("informed_direction", "NEUTRAL")
            if action == "BUY" and informed_direction in ["STRONG_SELL", "MEDIUM_SELL"]:
                return "contradictory_large_orders"
            if action == "SELL" and informed_direction in ["STRONG_BUY", "MEDIUM_BUY"]:
                return "contradictory_large_orders"
        
        vol_confirmation = raw_values.get("vol_confirmation", "UNKNOWN")
        if vol_confirmation == "CONTRADICTORY":
            return "contradictory_volume"
        
        return "ok"

    def _is_in_cooldown(self, symbol: str, action: str, strength: int) -> bool:
        st = self._state[symbol]
        now = time.time()
        
        if now < st["cooldown_until"]:
            if action != "HOLD" and self.cfg.allow_reversal_during_cooldown:
                prev = st["last_action"]
                if prev in ("BUY", "SELL") and prev != action:
                    if strength >= st["last_strength"]:
                        return False
            return True
        return False

    def _create_hold_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Створення сигналу HOLD"""
        now = time.time()
        st = self._state.get(symbol, {})
        
        result = {
            "symbol": symbol,
            "action": "HOLD",
            "strength": 0,
            "score_smoothed": 0.0,
            "score_raw": 0.0,
            "imbalance_score": 0.0,
            "momentum_score": 0.0,
            "spike": False,
            "spoof_filtered_volume": 0.0,
            "spread_bps": None,
            "volatility": 0.0,
            "cooldown_until": st.get("cooldown_until", 0.0),
            "cooldown_active": now < st.get("cooldown_until", 0.0),
            "reason": reason,
            "factors": {},
            "ohara_score": 0,
            "mtf_data": {}  # 🆕 Додаємо пусті MTF дані
        }
        
        try:
            signal_logger.log_signal(
                symbol=symbol,
                action="HOLD",
                strength=0,
                composite=0.0,
                ema=0.0,
                imbalance=0.0,
                momentum=0.0,
                bayesian="NEUTRAL",
                large_orders="NEUTRAL",
                frequency="UNKNOWN",
                vol_confirm="UNKNOWN",
                ohara_score=0,
                reason=reason,
                accepted=False
            )
        except Exception as e:
            logger.warning(f"[SIGNAL_LOGGER] Failed to log HOLD signal for {symbol}: {e}")
        
        return result

    def _create_cooldown_response(self, symbol: str, action: str, strength: int, 
                                ema_score: float, factors: Dict) -> Dict[str, Any]:
        st = self._state[symbol]
        return {
            "symbol": symbol,
            "action": "HOLD",
            "planned_action": action,
            "strength": 0,
            "raw_strength": strength,
            "score_smoothed": round(ema_score, 5),
            "score_raw": round(ema_score, 5),
            "imbalance_score": factors["raw_values"]["imbalance_score"],
            "momentum_score": factors["raw_values"]["momentum_score"],
            "spike": factors["raw_values"]["volume_acceleration"] > 50,
            "spoof_filtered_volume": 0.0,
            "spread_bps": None,
            "volatility": factors.get("volatility", 0.0),
            "cooldown_until": st["cooldown_until"],
            "cooldown_active": True,
            "reason": "cooldown",
            "factors": factors,
            "ohara_score": self._calculate_ohara_score(factors),
            "mtf_data": factors.get("mtf_data", {})  # 🆕 Додаємо MTF дані
        }

    def _update_state(self, symbol: str, action: str, strength: int):
        st = self._state[symbol]
        now = time.time()
        
        if action in ("BUY", "SELL") and strength >= self.cfg.strong_cooldown_level:
            st["cooldown_until"] = now + self.cfg.cooldown_seconds
        
        if action != "HOLD":
            st["last_action"] = action
            st["last_strength"] = strength
        
        st["last_update"] = now

    def _create_signal_response(self, symbol: str, action: str, strength: int, 
                              ema_score: float, composite_score: float, 
                              factors: Dict, reason: str) -> Dict[str, Any]:
        st = self._state[symbol]
        now = time.time()
        
        ohara_score = self._calculate_ohara_score(factors)
        
        return {
            "symbol": symbol,
            "action": action,
            "strength": strength,
            "score_smoothed": round(ema_score, 5),
            "score_raw": round(composite_score, 5),
            "imbalance_score": factors["raw_values"]["imbalance_score"],
            "momentum_score": factors["raw_values"]["momentum_score"],
            "spike": factors["raw_values"]["volume_acceleration"] > 50,
            "spoof_filtered_volume": 0.0,
            "spread_bps": None,
            "volatility": factors.get("volatility", 0.0),
            "cooldown_until": st["cooldown_until"],
            "cooldown_active": now < st["cooldown_until"],
            "reason": reason,
            "factors": factors,
            "ohara_score": ohara_score,
            "mtf_data": factors.get("mtf_data", {})  # 🆕 Додаємо MTF дані
        }

    def _calculate_ohara_score(self, factors: Dict) -> int:
        """Розрахунок комбінованого O'Hara score (0-10 балів)"""
        score = 0
        raw_values = factors.get("raw_values", {})
        
        bayesian_signal = raw_values.get("bayesian_signal", "NEUTRAL")
        if bayesian_signal in ["BULLISH", "BEARISH"]:
            score += 2
        elif bayesian_signal != "NEUTRAL":
            score += 1
        
        informed_dir = raw_values.get("informed_direction", "NEUTRAL")
        if informed_dir in ["STRONG_BUY", "STRONG_SELL"]:
            score += 3
        elif informed_dir in ["MEDIUM_BUY", "MEDIUM_SELL"]:
            score += 2
        
        activity = raw_values.get("activity_level", "UNKNOWN")
        if activity == "NORMAL":
            score += 2
        elif activity in ["HIGH", "LOW"]:
            score += 1
        
        vol_conf = raw_values.get("vol_confirmation", "UNKNOWN")
        if vol_conf == "CONFIRMED":
            score += 2
        elif vol_conf == "MODERATE":
            score += 1
        
        # 🆕 Додаємо бонус за MTF конвергенцію
        mtf_convergence = factors.get("mtf_convergence", 0)
        if mtf_convergence > 0.8:
            score += 2
        elif mtf_convergence > 0.6:
            score += 1
        
        return min(10, score)

    def _log_signal_generation(self, symbol: str, signal: Dict, factors: Dict):
        raw_values = factors["raw_values"]
        
        if signal["strength"] >= 3:
            ohara_score = signal.get("ohara_score", 0)
            mtf_convergence = factors.get("mtf_convergence", 0)
            
            logger.info(
                f"🎯 [STRONG_SIGNAL] {symbol}: {signal['action']}{signal['strength']} "
                f"score={signal['score_smoothed']:.3f} "
                f"ohara={ohara_score}/10 "
                f"MTF={mtf_convergence:.2f} "
                f"imb={raw_values['imbalance_score']:.0f}, "
                f"mom={raw_values['momentum_score']:.0f}, "
                f"bayesian={raw_values.get('bayesian_signal', 'N/A')}, "
                f"large_orders={raw_values.get('informed_direction', 'N/A')}, "
                f"vol_conf={raw_values.get('vol_confirmation', 'N/A')}, "
                f"reason={signal.get('reason', 'ok')}"
            )
            
            if abs(raw_values['momentum_score']) > 90:
                logger.warning(f"⚠️ [EXTREME_MOMENTUM] {symbol}: momentum={raw_values['momentum_score']:.0f}")
                
        elif signal["strength"] >= 1:
            mtf_convergence = factors.get("mtf_convergence", 0)
            logger.debug(
                f"[SIGNAL_DEBUG] {symbol}: {signal['action']}{signal['strength']} "
                f"score={signal['score_smoothed']:.3f}, MTF={mtf_convergence:.2f}, "
                f"reason={signal.get('reason', '')}"
            )

    def get_signal_state(self, symbol: str) -> Dict[str, Any]:
        """Отримати поточний стан сигналів для символу"""
        return self._state.get(symbol, {})

    def reset_cooldown(self, symbol: str):
        """Скинути cooldown для символу"""
        if symbol in self._state:
            self._state[symbol]["cooldown_until"] = 0.0
            logger.info(f"[SIGNAL] Cooldown reset for {symbol}")

    def get_quality_report(self) -> Dict[str, Any]:
        """Отримати звіт про якість сигналів"""
        return self.quality_monitor.get_performance_report()

    def track_signal_performance(self, signal: Dict, price_movement: float):
        """Відстежити результати сигналу для покращення якості"""
        self.quality_monitor.track_signal_quality(signal, price_movement)