#\analysis\signals.py
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

class SignalGenerator:
    def __init__(self):
        self.cfg = settings.signals
        self._state = {}
        self.quality_monitor = SignalQualityMonitor()

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
        # ІНІЦІАЛІЗАЦІЯ ДО ВСЬОГО
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

        logger.debug(f"[SIGNAL_DEBUG] {symbol}: imb={imb_score}, mom={mom_score}, vol={volatility}")
        logger.debug(f"[SIGNAL_DEBUG] {symbol}: tape={tape_analysis.get('large_net', 0)}, "
                     f"depth_ratio={depth_analysis.get('support_resistance_ratio', 0.5)}, "
                     f"poc_dist={cluster_analysis.get('poc_distance_pct', 0)}")

        factors = self._calculate_all_factors(
            imb_score, mom_score, tape_analysis, depth_analysis,
            cluster_analysis, spread_bps, volatility
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
        # Додаткова гарантія ініціалізації
        self._init_symbol(symbol)

        imb_score = imb_data.get("effective_imbalance", 0)
        mom_score = vol_data.get("momentum_score", 0)
        volatility = vol_data.get("volatility", 0)
        
        tape_analysis = vol_data.get("tape_analysis", {})
        depth_analysis = imb_data.get("depth_analysis", {})
        cluster_analysis = imb_data.get("cluster_analysis", {})
        
        factors = self._calculate_all_factors(
            imb_score, mom_score, tape_analysis, depth_analysis,
            cluster_analysis, spread_bps, volatility
        )
        
        composite_score = self._calculate_composite_score(factors)
        ema_score = self._apply_smoothing(symbol, composite_score)
        
        logger.info(f"🔍 [SIGNAL_DEBUG] {symbol}: ")
        logger.info(f"   - Imbalance: {imb_score:.1f} -> factor: {factors['imbalance']:.3f}")
        logger.info(f"   - Momentum: {mom_score:.1f} -> factor: {factors['momentum']:.3f}") 
        logger.info(f"   - Volatility: {volatility:.3f}% -> factor: {factors['volatility']:.3f}")
        logger.info(f"   - Tape: large_net={tape_analysis.get('large_net', 0)} -> factor: {factors['tape']:.3f}")
        logger.info(f"   - Depth: ratio={depth_analysis.get('support_resistance_ratio', 0):.2f} -> factor: {factors['depth']:.3f}")
        logger.info(f"   - Composite: {composite_score:.3f}, EMA: {ema_score:.3f}")
        
        action, strength, reason = self._generate_action_strength(symbol, ema_score, vol_data, factors)
        logger.info(f"   - Result: {action}{strength}, reason: {reason}")
        
        return action, strength

    def _calculate_all_factors(self, imb_score, mom_score, tape_analysis, 
                             depth_analysis, cluster_analysis, spread_bps, volatility):
        imb_norm = imb_score / 100.0
        mom_norm = mom_score / 100.0
        
        tape_factor = self._calculate_tape_factor(tape_analysis)
        depth_factor = self._calculate_depth_factor(depth_analysis)
        cluster_factor = self._calculate_cluster_factor(cluster_analysis)
        spread_factor = self._calculate_spread_factor(spread_bps)
        volatility_factor = self._calculate_volatility_factor(volatility)
        
        return {
            "imbalance": imb_norm,
            "momentum": mom_norm,
            "tape": tape_factor,
            "depth": depth_factor,
            "cluster": cluster_factor,
            "spread": spread_factor,
            "volatility": volatility_factor,
            "raw_values": {
                "imbalance_score": imb_score,
                "momentum_score": mom_score,
                "large_net": tape_analysis.get("large_net", 0),
                "volume_acceleration": tape_analysis.get("volume_acceleration", 0),
                "support_ratio": depth_analysis.get("support_resistance_ratio", 0.5),
                "poc_distance": cluster_analysis.get("poc_distance_pct", 0)
            }
        }

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
        if spread_bps is None:
            return 0.0
        
        max_spread = settings.spread.max_spread_threshold_bps
        if spread_bps > max_spread:
            return -0.5
        elif spread_bps > max_spread * 0.7:
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
        score = (
            factors["imbalance"] * self.cfg.weight_imbalance +
            factors["momentum"] * self.cfg.weight_momentum +
            factors["tape"] * 0.3 +
            factors["depth"] * 0.2 +
            factors["cluster"] * 0.1 +
            factors["spread"] * 0.15 +
            factors["volatility"] * 0.1
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
        """Генерація дії та сили з оптимізованими параметрами"""
        abs_score = abs(ema_score)
        raw_values = factors["raw_values"]

        # Фільтр волатильності з оптимізованим порогом
        volatility_percent = volume_data.get("volatility", 0)
        if volatility_percent < self.cfg.volatility_filter_threshold:  # 0.25% оптимізовано
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
        
        if abs_score >= composite_thresholds.get("strength_5", 0.75):
            strength = 5
        elif abs_score >= composite_thresholds.get("strength_4", 0.60):
            strength = 4
        elif abs_score >= composite_thresholds.get("strength_3", 0.40):
            strength = 3
        elif abs_score >= composite_thresholds.get("strength_2", 0.25):
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
        """Перевірка якості сигналів"""
        large_net = raw_values.get("large_net", 0)
        support_ratio = raw_values.get("support_ratio", 0.5)
        momentum_score = raw_values.get("momentum_score", 0)
        imbalance_score = raw_values.get("imbalance_score", 0)
        
        if abs(momentum_score) > 95 and strength >= 3:
            return "extreme_momentum"
        
        if action == "BUY" and imbalance_score < -20 and momentum_score > 80:
            return "contradictory_signals"
        if action == "SELL" and imbalance_score > 20 and momentum_score < -80:
            return "contradictory_signals"
        
        if action == "BUY" and large_net < -3:
            return "contradictory_large_trades"
        if action == "SELL" and large_net > 3:
            return "contradictory_large_trades"
        
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
            "factors": {}
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
            "factors": factors
        }

    def _update_state(self, symbol: str, action: str, strength: int):
        st = self._state[symbol]
        now = time.time()
        
        if action in ("BUY", "SELL") and strength >= self.cfg.strong_cooldown_level:
            st["cooldown_until"] = now + self.cfg.cooldown_seconds  # 3 хвилини оптимізовано
        
        if action != "HOLD":
            st["last_action"] = action
            st["last_strength"] = strength
        
        st["last_update"] = now

    def _create_signal_response(self, symbol: str, action: str, strength: int, 
                              ema_score: float, composite_score: float, 
                              factors: Dict, reason: str) -> Dict[str, Any]:
        st = self._state[symbol]
        now = time.time()
        
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
            "factors": factors
        }

    def _log_signal_generation(self, symbol: str, signal: Dict, factors: Dict):
        raw_values = factors["raw_values"]
        
        if signal["strength"] >= 3:
            logger.info(
                f"🎯 [STRONG_SIGNAL] {symbol}: {signal['action']}{signal['strength']} "
                f"score={signal['score_smoothed']:.3f} "
                f"imb={raw_values['imbalance_score']:.0f}, "
                f"mom={raw_values['momentum_score']:.0f}, "
                f"large_net={raw_values['large_net']:+d}, "
                f"depth_ratio={raw_values['support_ratio']:.2f}, "
                f"vol={factors.get('volatility', 0):.1f}%, "
                f"reason={signal.get('reason', 'ok')}"
            )
            
            if abs(raw_values['momentum_score']) > 90:
                logger.warning(f"⚠️  [EXTREME_MOMENTUM] {symbol}: momentum={raw_values['momentum_score']:.0f}")
                
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