# analysis/signals.py
import time
import math
from typing import Dict, Any, Tuple
from collections import deque
from config.settings import settings
from utils.logger import logger

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
    """🆕 O'HARA METHOD 7: Spread as Risk Measure"""
    
    def __init__(self):
        self.cfg = settings.spread
        self._spread_history = {}  # {symbol: deque of spreads}
        self._spread_baseline = {}  # {symbol: avg_spread}
    
    def update(self, symbol: str, bid: float, ask: float):
        """Оновлення spread даних"""
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        
        spread_abs = ask - bid
        spread_bps = (spread_abs / bid) * 10000  # basis points
        
        # Ініціалізація історії
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=self.cfg.spread_history_size)
        
        # Додаємо до історії
        self._spread_history[symbol].append({
            'timestamp': time.time(),
            'spread_bps': spread_bps
        })
        
        # Оновлюємо baseline (середній spread)
        if len(self._spread_history[symbol]) >= 10:
            avg_spread = sum(s['spread_bps'] for s in self._spread_history[symbol]) / len(self._spread_history[symbol])
            self._spread_baseline[symbol] = avg_spread
    
    def get_risk_level(self, symbol: str, current_spread_bps: float) -> Dict[str, Any]:
        """
        Визначення рівня ризику на основі spread згідно O'Hara
        """
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
        
        # Визначаємо рівень ризику
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
        self._state = {}
        self.quality_monitor = SignalQualityMonitor()
        self.spread_monitor = SpreadMonitor()  # 🆕 O'Hara Method 7

    def _init_symbol(self, symbol: str):
        if symbol not in self._state:
            self._state[symbol] = {
                "ema_score": 0.0,
                "last_action": "HOLD",
                "last_strength": 0,
                "cooldown_until": 0.0,
                "last_update": 0.0
            }

    def generate(self, symbol: str, imbalance_data: Dict, volume_data: Dict, spread_bps: float = None) -> Dict[str, Any]:
        self._init_symbol(symbol)

        mom_score = volume_data.get("momentum_score", 0)
        if abs(mom_score) > 50:
            self.debug_signal_calculation(symbol, imbalance_data, volume_data, spread_bps)

        st = self._state[symbol]

        volatility = volume_data.get("volatility", 0)
        trades_count = volume_data.get("short_trades_count", 0)
        if volatility == 0.1 and trades_count < 10:
            logger.warning(f"⚠️ [DATA_QUALITY] {symbol}: Suspected fake volatility data")
            return self._create_hold_signal(symbol, "suspicious_volatility_data")

        imb_score = imbalance_data.get("effective_imbalance", imbalance_data.get("imbalance_score", 0))
        mom_score = volume_data.get("momentum_score", 0)
        volatility = volume_data.get("volatility", 0)

        tape_analysis = volume_data.get("tape_analysis", {})
        depth_analysis = imbalance_data.get("depth_analysis", {})
        cluster_analysis = imbalance_data.get("cluster_analysis", {})

        # 🆕 O'HARA DATA
        bayesian_data = imbalance_data.get("bayesian_data", {})
        trade_imbalance = imbalance_data.get("trade_imbalance", {})
        frequency_data = volume_data.get("frequency_data", {})
        volume_confirm = volume_data.get("volume_confirmation", {})
        large_order_data = volume_data.get("large_order_data", {})

        logger.debug(f"[SIGNAL_DEBUG] {symbol}: imb={imb_score}, mom={mom_score}, vol={volatility}")
        logger.debug(f"[SIGNAL_DEBUG] {symbol}: bayesian={bayesian_data.get('signal')}, "
                     f"freq={frequency_data.get('activity_level')}, "
                     f"vol_confirm={volume_confirm.get('confirmation')}")

        factors = self._calculate_all_factors(
            imb_score, mom_score, tape_analysis, depth_analysis,
            cluster_analysis, spread_bps, volatility,
            bayesian_data, trade_imbalance, frequency_data, 
            volume_confirm, large_order_data
        )

        composite_score = self._calculate_composite_score(factors)
        ema_score = self._apply_smoothing(symbol, composite_score)
        action, strength, reason = self._generate_action_strength(symbol, ema_score, volume_data, factors)

        if self._is_in_cooldown(symbol, action, strength):
            return self._create_cooldown_response(symbol, action, strength, ema_score, factors)

        self._update_state(symbol, action, strength)

        result = self._create_signal_response(
            symbol, action, strength, ema_score, composite_score, factors, reason
        )
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

    def _calculate_all_factors(self, imb_score, mom_score, tape_analysis, 
                             depth_analysis, cluster_analysis, spread_bps, volatility,
                             bayesian_data, trade_imbalance, frequency_data,
                             volume_confirm, large_order_data):
        imb_norm = imb_score / 100.0
        mom_norm = mom_score / 100.0
        
        tape_factor = self._calculate_tape_factor(tape_analysis)
        depth_factor = self._calculate_depth_factor(depth_analysis)
        cluster_factor = self._calculate_cluster_factor(cluster_analysis)
        spread_factor = self._calculate_spread_factor(spread_bps)
        volatility_factor = self._calculate_volatility_factor(volatility)
        
        # 🆕 O'HARA FACTORS
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
        """🆕 O'HARA METHOD 1: Bayesian factor"""
        signal = bayesian_data.get("signal", "NEUTRAL")
        confidence = bayesian_data.get("confidence", 0)
        
        if signal == "BULLISH":
            return confidence * 0.5  # Позитивний фактор
        elif signal == "BEARISH":
            return -confidence * 0.5  # Негативний фактор
        else:
            return 0.0

    def _calculate_large_order_factor(self, large_order_data: Dict) -> float:
        """🆕 O'HARA METHOD 2: Large orders factor"""
        direction = large_order_data.get("informed_direction", "NEUTRAL")
        
        if direction == "STRONG_BUY":
            return 0.8
        elif direction == "MEDIUM_BUY":
            return 0.4
        elif direction == "STRONG_SELL":
            return -0.8
        elif direction == "MEDIUM_SELL":
            return -0.4
        else:
            return 0.0

    def _calculate_frequency_factor(self, frequency_data: Dict) -> float:
        """🆕 O'HARA METHOD 3: Trade frequency factor"""
        activity_level = frequency_data.get("activity_level", "UNKNOWN")
        risk_signal = frequency_data.get("risk_signal", "OK")
        
        # Штраф за аномальну активність
        if risk_signal == "AVOID":
            return -0.5  # Дуже висока активність - не торгувати
        elif risk_signal == "CAUTION":
            return -0.2  # Висока активність - обережно
        elif risk_signal == "LOW_LIQUIDITY":
            return -0.3  # Дуже низька активність - тихо перед бурею
        else:
            return 0.0

    def _calculate_volume_confirm_factor(self, volume_confirm: Dict) -> float:
        """🆕 O'HARA METHOD 5: Volume confirmation factor"""
        confirmation = volume_confirm.get("confirmation", "UNKNOWN")
        strength = volume_confirm.get("strength", "WEAK")
        
        if confirmation == "CONFIRMED" and strength == "STRONG":
            return 0.3  # Обсяг підтверджує рух
        elif confirmation == "MODERATE":
            return 0.1
        elif confirmation == "WEAK":
            return -0.2  # Фейковий рух
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
        """🆕 O'HARA METHOD 7: Spread factor"""
        if spread_bps is None:
            return 0.0
        
        # Отримуємо рівень ризику від SpreadMonitor
        # Тут spread_bps вже передано, тож просто перевіряємо поріг
        max_spread = settings.spread.max_spread_threshold_bps
        if spread_bps > max_spread * 2:
            return -0.5  # Дуже широкий spread
        elif spread_bps > max_spread:
            return -0.2  # Широкий spread
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
        
        if factors["raw_values"]["volume_acceleration"] > 50:
            score += self.cfg.spike_bonus
        
        return max(-1.0, min(1.0, score))

    def _apply_smoothing(self, symbol: str, current_score: float) -> float:
        st = self._state[symbol]
        ema_prev = st.get("ema_score", 0.0)
        ema_new = ema_prev * (1 - self.cfg.smoothing_alpha) + current_score * self.cfg.smoothing_alpha
        st["ema_score"] = ema_new
        return ema_new

    def _generate_action_strength(
        self, symbol: str, ema_score: float, volume_data: Dict, factors: Dict
    ) -> Tuple[str, int, str]:
        """Генерація дії та сили з O'Hara фільтрами"""
        abs_score = abs(ema_score)
        raw_values = factors["raw_values"]

        # 🆕 O'HARA FILTER: Trade Frequency
        if self.cfg.enable_volume_validation:
            activity_level = raw_values.get("activity_level", "UNKNOWN")
            if activity_level == "VERY_HIGH":
                logger.debug(f"[OHARA_FILTER] {symbol}: VERY_HIGH activity - avoiding trade")
                return "HOLD", 0, "very_high_activity"
            elif activity_level == "VERY_LOW":
                logger.debug(f"[OHARA_FILTER] {symbol}: VERY_LOW activity - low liquidity")
                return "HOLD", 0, "very_low_activity"

        # Фільтр волатильності
        volatility_percent = volume_data.get("volatility", 0)
        if volatility_percent < self.cfg.volatility_filter_threshold:
            logger.debug(f"[VOL_FILTER] {symbol} filtered: {volatility_percent:.3f}% < {self.cfg.volatility_filter_threshold}%")
            return "HOLD", 0, "low_volatility"

        # Перевірка мінімального обсягу
        if self.cfg.enable_volume_validation:
            min_volume = self.cfg.min_short_volume_for_signal
            current_volume = volume_data.get("total_volume_short", 0)
            trades_count = volume_data.get("short_trades_count", 0)
            
            if current_volume < min_volume and trades_count < self.cfg.min_trades_for_signal:
                return "HOLD", 0, "low_volume"

        # Основний поріг
        if abs_score < self.cfg.hold_threshold:
            return "HOLD", 0, "below_threshold"

        action = "BUY" if ema_score > 0 else "SELL"

        # Оптимізована система визначення сили
        strength = 0
        composite_thresholds = self.cfg.composite_thresholds
        
        if abs_score >= composite_thresholds.get("strength_5", 0.80):
            strength = 5
        elif abs_score >= composite_thresholds.get("strength_4", 0.65):
            strength = 4
        elif abs_score >= composite_thresholds.get("strength_3", 0.45):
            strength = 3
        elif abs_score >= composite_thresholds.get("strength_2", 0.30):
            strength = 2
        elif abs_score >= composite_thresholds.get("strength_1", 0.15):
            strength = 1

        if strength < self.cfg.min_strength_for_action:
            return "HOLD", 0, "weak_signal"

        if self.cfg.require_signal_consistency:
            reason = self._validate_signal_consistency(action, strength, raw_values)
            if reason != "ok":
                return "HOLD", 0, reason

        quality_reason = self._validate_signal_quality(action, strength, raw_values)
        if quality_reason != "ok":
            return "HOLD", 0, quality_reason

        return action, strength, "ok"

    def _validate_signal_consistency(self, action: str, strength: int, raw_values: Dict) -> str:
        """Перевірка узгодженості сигналів"""
        imbalance_score = raw_values.get("imbalance_score", 0)
        
        max_contradiction = self.cfg.max_imbalance_contradiction
        if action == "BUY" and imbalance_score < -max_contradiction:
            return "contradictory_imbalance_buy"
        if action == "SELL" and imbalance_score > max_contradiction:
            return "contradictory_imbalance_sell"
        
        return "ok"

    def _validate_signal_quality(self, action: str, strength: int, raw_values: Dict) -> str:
        """✅ ВИПРАВЛЕНО: Перевірка якості сигналів з O'Hara + фільтр пізнього входу"""
        large_net = raw_values.get("large_net", 0)
        support_ratio = raw_values.get("support_ratio", 0.5)
        momentum_score = raw_values.get("momentum_score", 0)
        imbalance_score = raw_values.get("imbalance_score", 0)
        
        # 🆕 ВИПРАВЛЕННЯ 1: Фільтр пізнього входу (Late Entry Filter)
        if self.cfg.enable_exhaustion_filter:
            # Якщо моментум екстремальний але імбаланс слабкий = пропустили рух
            if abs(momentum_score) > self.cfg.max_momentum_for_entry:
                if abs(imbalance_score) < self.cfg.min_imbalance_for_high_momentum:
                    logger.debug(f"[LATE_ENTRY] {action}: mom={momentum_score:.0f}, imb={imbalance_score:.0f}")
                    return "late_entry"
            
            # 🆕 ВИПРАВЛЕННЯ 2: Суперечність між моментумом і імбалансом
            if action == "BUY":
                if momentum_score > 60 and imbalance_score < -5:
                    logger.debug(f"[CONTRADICTORY] BUY: mom={momentum_score:.0f}, imb={imbalance_score:.0f}")
                    return "contradictory_momentum_imbalance"
            elif action == "SELL":
                if momentum_score < -60 and imbalance_score > 5:
                    logger.debug(f"[CONTRADICTORY] SELL: mom={momentum_score:0f}, imb={imbalance_score:.0f}")
                    return "contradictory_momentum_imbalance"
        
        # 🆕 O'HARA VALIDATION: Large orders consistency
        informed_direction = raw_values.get("informed_direction", "NEUTRAL")
        if action == "BUY" and informed_direction in ["STRONG_SELL", "MEDIUM_SELL"]:
            return "contradictory_large_orders"
        if action == "SELL" and informed_direction in ["STRONG_BUY", "MEDIUM_BUY"]:
            return "contradictory_large_orders"
        
        # 🆕 O'HARA VALIDATION: Volume confirmation
        vol_confirmation = raw_values.get("vol_confirmation", "UNKNOWN")
        if vol_confirmation == "WEAK" and strength >= 4:
            return "weak_volume_confirmation"
        
        if abs(momentum_score) > 95 and strength >= 3:
            return "extreme_momentum"
        
        if action == "BUY" and imbalance_score < -20 and momentum_score > 80:
            return "contradictory_signals"
        if action == "SELL" and imbalance_score > 20 and momentum_score < -80:
            return "contradictory_signals"
        
        if action == "BUY" and support_ratio < 0.3:
            return "weak_support"
        if action == "SELL" and support_ratio > 0.7:
            return "weak_resistance"
        
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
        
        return {
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
            "ohara_score": 0
        }

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
            "ohara_score": self._calculate_ohara_score(factors)
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
            "ohara_score": ohara_score  # 🆕 Combined O'Hara score
        }

    def _calculate_ohara_score(self, factors: Dict) -> int:
        """
        🆕 Розрахунок комбінованого O'Hara score (0-10 балів)
        """
        score = 0
        raw_values = factors.get("raw_values", {})
        
        # Bayesian (0-2 бали)
        bayesian_signal = raw_values.get("bayesian_signal", "NEUTRAL")
        if bayesian_signal in ["BULLISH", "BEARISH"]:
            score += 2
        elif bayesian_signal != "NEUTRAL":
            score += 1
        
        # Large Orders (0-3 бали)
        informed_dir = raw_values.get("informed_direction", "NEUTRAL")
        if informed_dir in ["STRONG_BUY", "STRONG_SELL"]:
            score += 3
        elif informed_dir in ["MEDIUM_BUY", "MEDIUM_SELL"]:
            score += 2
        
        # Frequency (0-2 бали - штраф за аномалії)
        activity = raw_values.get("activity_level", "UNKNOWN")
        if activity == "NORMAL":
            score += 2
        elif activity in ["HIGH", "LOW"]:
            score += 1
        # VERY_HIGH, VERY_LOW = 0 балів
        
        # Volume Confirmation (0-2 бали)
        vol_conf = raw_values.get("vol_confirmation", "UNKNOWN")
        if vol_conf == "CONFIRMED":
            score += 2
        elif vol_conf == "MODERATE":
            score += 1
        
        # Trade Imbalance (0-1 бал) - вже врахований в imbalance
        # Spread (додатковий штраф вже в spread_factor)
        
        return min(10, score)

    def _log_signal_generation(self, symbol: str, signal: Dict, factors: Dict):
        raw_values = factors["raw_values"]
        
        if signal["strength"] >= 3:
            ohara_score = signal.get("ohara_score", 0)
            logger.info(
                f"🎯 [STRONG_SIGNAL] {symbol}: {signal['action']}{signal['strength']} "
                f"score={signal['score_smoothed']:.3f} "
                f"ohara={ohara_score}/10 "
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
            logger.debug(
                f"[SIGNAL_DEBUG] {symbol}: {signal['action']}{signal['strength']} "
                f"score={signal['score_smoothed']:.3f}, reason={signal.get('reason', '')}"
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