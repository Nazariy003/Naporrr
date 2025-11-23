# utils/backtest/optimizer.py
import itertools
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from config.settings import settings
from utils.logger import logger

@dataclass
class ParameterRange:
    """Діапазон параметра для оптимізації"""
    name: str
    min_value: float
    max_value: float
    step: float
    current_value: float
    
    def get_test_values(self) -> List[float]:
        """Генерація значень для тестування"""
        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(round(current, 4))
            current += self.step
        return values

class ParameterOptimizer:
    """
    🎯 Grid Search оптимізація параметрів
    
    Оптимізує:
    - Сигнали: weights, thresholds, smoothing
    - Ризик: TP/SL multipliers, lifetime multipliers
    - Аналізатори: вікна, ваги momentum
    """
    
    def __init__(self):
        self.optimization_space = self._define_optimization_space()
        
    def _define_optimization_space(self) -> Dict[str, ParameterRange]:
        """Визначення простору оптимізації"""
        space = {
            # === SIGNALS ===
            'weight_imbalance': ParameterRange(
                name='signals.weight_imbalance',
                min_value=0.2,
                max_value=0.6,
                step=0.1,
                current_value=settings.signals.weight_imbalance
            ),
            'weight_momentum': ParameterRange(
                name='signals.weight_momentum',
                min_value=0.2,
                max_value=0.6,
                step=0.1,
                current_value=settings.signals.weight_momentum
            ),
            'smoothing_alpha': ParameterRange(
                name='signals.smoothing_alpha',
                min_value=0.2,
                max_value=0.6,
                step=0.1,
                current_value=settings.signals.smoothing_alpha
            ),
            'hold_threshold': ParameterRange(
                name='signals.hold_threshold',
                min_value=0.08,
                max_value=0.20,
                step=0.04,
                current_value=settings.signals.hold_threshold
            ),
            
            # === RISK ===
            'sl_vol_multiplier': ParameterRange(
                name='risk.sl_vol_multiplier',
                min_value=1.0,
                max_value=2.5,
                step=0.5,
                current_value=settings.risk.sl_vol_multiplier
            ),
            'tp_vol_multiplier': ParameterRange(
                name='risk.tp_vol_multiplier',
                min_value=2.0,
                max_value=4.0,
                step=0.5,
                current_value=settings.risk.tp_vol_multiplier
            ),
            'low_volatility_lifetime_multiplier': ParameterRange(
                name='risk.low_volatility_lifetime_multiplier',
                min_value=1.2,
                max_value=2.0,
                step=0.2,
                current_value=settings.risk.low_volatility_lifetime_multiplier
            ),
            'high_volatility_lifetime_multiplier': ParameterRange(
                name='risk.high_volatility_lifetime_multiplier',
                min_value=0.5,
                max_value=0.9,
                step=0.1,
                current_value=settings.risk.high_volatility_lifetime_multiplier
            ),
            
            # === IMBALANCE ===
            'smoothing_factor': ParameterRange(
                name='imbalance.smoothing_factor',
                min_value=0.2,
                max_value=0.5,
                step=0.1,
                current_value=settings.imbalance.smoothing_factor
            ),
            
            # === VOLUME ===
            # momentum_weights - окремо через складність
        }
        
        return space
    
    def optimize(self, 
                replay_engine,
                start_date,
                end_date,
                symbols: List[str],
                max_combinations: int = None) -> Tuple[Dict, Dict]:
        """
        Оптимізація параметрів
        
        Args:
            replay_engine: ReplayEngine instance
            start_date: Початок періоду
            end_date: Кінець періоду
            symbols: Список символів
            max_combinations: Максимум комбінацій (None = всі)
        
        Returns:
            (best_params, all_results)
        """
        logger.info("🔍 [OPTIMIZER] Starting parameter optimization...")
        
        # Генерація сітки параметрів
        param_grid = self._generate_parameter_grid(max_combinations)
        
        logger.info(f"📊 [OPTIMIZER] Testing {len(param_grid)} combinations")
        
        # Тестування кожної комбінації
        results = []
        start_time = time.time()
        
        for idx, params in enumerate(param_grid, 1):
            try:
                # Replay з цими параметрами
                result = replay_engine.replay_period(
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    test_params=params
                )
                
                if not result:
                    continue
                
                # Розрахунок objective score
                score = self._calculate_objective_score(result['metrics'])
                
                results.append({
                    'params': params,
                    'metrics': result['metrics'],
                    'score': score,
                    'trades': result.get('trades', [])
                })
                
                # Прогрес
                if idx % 10 == 0 or idx == len(param_grid):
                    elapsed = time.time() - start_time
                    eta = (elapsed / idx) * (len(param_grid) - idx)
                    logger.info(f"⏳ [OPTIMIZER] Progress: {idx}/{len(param_grid)} "
                              f"(ETA: {eta/60:.1f}min)")
                    
            except Exception as e:
                logger.error(f"❌ [OPTIMIZER] Error testing params {idx}: {e}")
        
        # Сортування за score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        best_result = results[0] if results else None
        
        if best_result:
            logger.info(f"🏆 [OPTIMIZER] Best score: {best_result['score']:.4f}")
            logger.info(f"📈 [OPTIMIZER] Best params: {best_result['params']}")
        else:
            logger.error("❌ [OPTIMIZER] No valid results")
        
        return best_result, results
    
    def _generate_parameter_grid(self, max_combinations: int = None) -> List[Dict]:
        """Генерація сітки параметрів"""
        # Отримуємо всі можливі значення для кожного параметра
        param_values = {}
        for param_name, param_range in self.optimization_space.items():
            param_values[param_name] = param_range.get_test_values()
        
        # Генеруємо всі комбінації
        param_names = list(param_values.keys())
        all_combinations = list(itertools.product(*[param_values[name] for name in param_names]))
        
        # Обмежуємо кількість якщо потрібно
        if max_combinations and len(all_combinations) > max_combinations:
            # Випадкова вибірка
            import random
            all_combinations = random.sample(all_combinations, max_combinations)
            logger.info(f"⚠️ [OPTIMIZER] Limited to {max_combinations} random combinations")
        
        # Конвертуємо в список словників
        param_grid = []
        for combination in all_combinations:
            params = dict(zip(param_names, combination))
            param_grid.append(params)
        
        return param_grid
    
    def _calculate_objective_score(self, metrics: Dict) -> float:
        """
        Розрахунок objective score для ранжування
        
        Formula: weighted combination of key metrics
        """
        if not metrics:
            return -999999
        
        # Витягуємо метрики
        win_rate = metrics.get('win_rate', 0)
        total_pnl = metrics.get('total_pnl', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Перевірки валідності
        if total_trades < 10:
            return -999999  # Недостатньо трейдів
        
        if win_rate <= 0 or profit_factor <= 0:
            return -999999
        
        # Нормалізація метрик (0-1)
        win_rate_norm = min(win_rate / 100, 1.0)
        sharpe_norm = min(max(sharpe, 0) / 3.0, 1.0)  # Sharpe > 3 = excellent
        pf_norm = min(profit_factor / 3.0, 1.0)  # PF > 3 = excellent
        dd_penalty = max(0, 1 - (max_dd / 20))  # Penalty for DD > 20%
        
        # Ваги метрик
        weights = {
            'win_rate': 0.2,
            'sharpe': 0.3,
            'profit_factor': 0.3,
            'total_pnl': 0.1,
            'drawdown': 0.1
        }
        
        # Розрахунок score
        score = (
            weights['win_rate'] * win_rate_norm +
            weights['sharpe'] * sharpe_norm +
            weights['profit_factor'] * pf_norm +
            weights['total_pnl'] * (1 if total_pnl > 0 else 0) +
            weights['drawdown'] * dd_penalty
        )
        
        return score
    
    def compare_with_current(self, best_params: Dict, best_metrics: Dict) -> Dict:
        """
        Порівняння знайдених параметрів з поточними
        
        Returns:
            Dict з рекомендаціями
        """
        current_performance = self._get_current_performance()
        
        # Порівняння метрик
        improvement = {}
        
        for metric_name in ['win_rate', 'profit_factor', 'sharpe_ratio', 'total_pnl']:
            current = current_performance.get(metric_name, 0)
            new = best_metrics.get(metric_name, 0)
            
            if current > 0:
                change_pct = ((new - current) / current) * 100
            else:
                change_pct = 0
            
            improvement[metric_name] = {
                'current': current,
                'new': new,
                'change_pct': change_pct,
                'improved': new > current
            }
        
        # Рішення про застосування
        should_apply = self._should_apply_params(improvement)
        
        return {
            'improvement': improvement,
            'should_apply': should_apply,
            'reason': self._get_apply_reason(improvement, should_apply)
        }
    
    def _get_current_performance(self) -> Dict:
        """Отримання поточної performance з логів"""
        try:
            import pandas as pd
            from pathlib import Path
            
            # ✅ ВИПРАВЛЕННЯ: перевірка існування файлу
            trades_file = Path("logs/trades.csv")
            if not trades_file.exists():
                logger.warning("⚠️  [CURRENT_PERF] No logs/trades.csv found, using defaults")
                return {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'total_pnl': 0
                }
            
            trades_df = pd.read_csv(trades_file)
            
            # Останні 100 трейдів
            recent_trades = trades_df.tail(100)
            
            # Розрахунок метрик
            metrics = self._calculate_metrics_from_trades(recent_trades)
            return metrics
            
        except Exception as e:
            logger.error(f"❌ [CURRENT_PERF] Error: {e}")
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'total_pnl': 0
            }
    
    def _calculate_metrics_from_trades(self, trades_df) -> Dict:
        """Розрахунок метрик з трейдів"""
        # Placeholder - детальна реалізація в metrics.py
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'total_pnl': 0
        }
    
    def _should_apply_params(self, improvement: Dict) -> bool:
        """Чи варто застосовувати нові параметри?"""
        # Мінімальні пороги покращення
        min_improvement = settings.backtest.min_improvement_threshold_pct
        
        # Перевіряємо ключові метрики
        key_metrics = ['win_rate', 'profit_factor', 'sharpe_ratio']
        improved_count = sum(
            1 for metric in key_metrics 
            if improvement.get(metric, {}).get('change_pct', -999) >= min_improvement
        )
        
        # Потрібно покращення мінімум 2 з 3 ключових метрик
        return improved_count >= 2
    
    def _get_apply_reason(self, improvement: Dict, should_apply: bool) -> str:
        """Пояснення рішення"""
        if should_apply:
            improved = [
                f"{k}: +{v['change_pct']:.1f}%" 
                for k, v in improvement.items() 
                if v.get('improved', False)
            ]
            return f"Significant improvement detected: {', '.join(improved)}"
        else:
            return "Insufficient improvement to justify parameter change"