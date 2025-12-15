#!/usr/bin/env python3
"""
🎯 MARKET-CONTEXT OPTIMIZATION ENGINE v4.0
===========================================

НОВИЙ АЛГОРИТМ:
1. 📊 Аналіз ринкового контексту (всі пари, 48h даних)
2. 🔬 Глибокий аналіз паттернів (що працює/не працює)
3. ⚙️ Автоматична оптимізація параметрів
4. 🧪 Тестування 1000+ комбінацій параметрів
5. 📈 Вибір найкращої стратегії для поточних умов
6. 🔄 Ітераційне покращення до досягнення цільових результатів

ЦІЛЬ: Win Rate > 55%, Profit Factor > 1.5, Total PnL > $50
"""

import os
import sys
import csv
import json
import argparse
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Додаємо кореневу директорію до path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

# =============================================================================
# 📊 КОНСТАНТИ ТА НАЛАШТУВАННЯ ОПТИМІЗАЦІЇ
# =============================================================================

# Цільові показники
TARGET_WIN_RATE = 55.0      # Мінімум 55% виграшних угод
TARGET_PROFIT_FACTOR = 1.5  # Мінімум 1.5 profit factor
TARGET_TOTAL_PNL = 50.0     # Мінімум $50 загального PnL

# Комісії
BYBIT_FEE = 0.0001  # 0.01%
MIN_TP_AFTER_FEES = 0.002  # 0.2% мінімум

# Максимальний час оптимізації (хвилин)
MAX_OPTIMIZATION_MINUTES = 10

# =============================================================================
# 📊 DATA CLASSES ДЛЯ ОПТІМІЗАЦІЇ
# =============================================================================

@dataclass
class StrategyConfig:
    """Конфігурація стратегії для оптимізації"""
    # Фільтри вхідних сигналів
    min_composite: float = 0.35
    min_strength: int = 3
    min_imbalance: float = 20.0
    min_momentum: float = 40.0
    max_momentum: float = 85.0
    min_ohara: int = 5
    
    # Ризик-менеджмент
    tp_pct: float = 0.005      # 0.5%
    sl_pct: float = 0.003      # 0.3%
    max_hold_minutes: int = 45
    
    # Позиціонування
    position_size_usd: float = 100.0
    
    # Адаптивність
    adapt_to_market: bool = True
    use_dynamic_tpsl: bool = True
    
    def __post_init__(self):
        # Перевірка мінімального TP після комісій
        if self.tp_pct < MIN_TP_AFTER_FEES:
            self.tp_pct = MIN_TP_AFTER_FEES
        
        # Перевірка RR ratio
        if self.tp_pct / self.sl_pct < 1.2:
            self.sl_pct = self.tp_pct / 1.5
    
    def get_id(self) -> str:
        """Унікальний ідентифікатор конфігурації"""
        return (f"C{self.min_composite:.2f}_S{self.min_strength}_"
                f"I{self.min_imbalance:.0f}_M{self.min_momentum:.0f}-{self.max_momentum:.0f}_"
                f"O{self.min_ohara}_TP{self.tp_pct:.4f}_SL{self.sl_pct:.4f}")

@dataclass
class OptimizationResult:
    """Результат оптимізації однієї конфігурації"""
    config: StrategyConfig
    trades_count: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_net_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_duration_min: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Детальна статистика
    by_symbol: Dict[str, Dict] = field(default_factory=dict)
    by_exit_reason: Dict[str, Dict] = field(default_factory=dict)
    best_trades: List[Dict] = field(default_factory=list)
    worst_trades: List[Dict] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """Оцінка стратегії (більше = краще)"""
        if self.trades_count < 20:  # Мінімум 20 угод
            return -1000.0
        
        # Базові бали
        score = 0.0
        
        # Win Rate (вага 40%)
        win_rate_score = (self.win_rate - 50.0) * 2.0  # Нормалізація
        score += win_rate_score * 0.4
        
        # Profit Factor (вага 30%)
        pf_score = (self.profit_factor - 1.0) * 10.0
        score += pf_score * 0.3
        
        # Total PnL (вага 20%)
        pnl_score = self.total_net_pnl / 10.0
        score += pnl_score * 0.2
        
        # Стабільність (вага 10%)
        stability_score = 0.0
        if self.trades_count > 0:
            # Менше TIME_EXIT = краще
            time_exit_ratio = self.by_exit_reason.get("TIME_EXIT", {}).get("count", 0) / self.trades_count
            stability_score = (1.0 - time_exit_ratio) * 10.0
        score += stability_score * 0.1
        
        # Штраф за мало угод
        if self.trades_count < 50:
            score *= (self.trades_count / 50.0)
        
        return score
    
    @property
    def meets_targets(self) -> bool:
        """Перевіряє, чи задовольняє стратегія цілям"""
        return (self.win_rate >= TARGET_WIN_RATE and
                self.profit_factor >= TARGET_PROFIT_FACTOR and
                self.total_net_pnl >= TARGET_TOTAL_PNL and
                self.trades_count >= 30)

@dataclass 
class MarketRegime:
    """Ринковий режим для адаптивної оптимізації"""
    name: str  # SIDEWAYS, UPTREND, DOWNTREND, HIGH_VOL, LOW_VOL
    detected_at: datetime
    strength: float  # 0-1
    characteristics: Dict[str, Any]
    
    # Оптимальні параметри для цього режиму
    optimal_params: Optional[Dict[str, Any]] = None

# =============================================================================
# 📊 ДАНІ ТА ЗАВАНТАЖУВАЧ
# =============================================================================

class OptimizationDataLoader:
    """Завантажувач даних для оптимізації"""
    
    def __init__(self, hours_back: int = 48):
        self.hours_back = hours_back
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'OptimizationEngine/4.0'})
    
    def load_all_data(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Завантажує всі дані для оптимізації"""
        print(f"\n📊 ЗАВАНТАЖЕННЯ {self.hours_back}h ДАНИХ ДЛЯ ОПТИМІЗАЦІЇ")
        print("=" * 70)
        
        all_data = {}
        total_candles = 0
        
        for symbol in symbols:
            print(f"  📥 {symbol}...", end=" ", flush=True)
            
            try:
                candles = self._load_symbol_data(symbol)
                if candles and len(candles) >= self.hours_back * 30:
                    all_data[symbol] = candles
                    total_candles += len(candles)
                    print(f"✅ {len(candles)} свічок")
                else:
                    print(f"❌ недостатньо даних")
            except Exception as e:
                print(f"❌ помилка: {e}")
        
        print(f"\n📈 ЗАВАНТАЖЕНО: {len(all_data)} пар, {total_candles:,} свічок")
        return all_data
    
    def _load_symbol_data(self, symbol: str) -> List[Dict]:
        """Завантажує дані для одного символу"""
        candles = []
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = end_time - (self.hours_back * 3600 * 1000)
        
        current_end = end_time
        
        while current_end > start_time:
            try:
                response = self.session.get(
                    f"{self.base_url}/v5/market/kline",
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "interval": "1",
                        "end": current_end,
                        "limit": 1000
                    },
                    timeout=10
                )
                
                data = response.json()
                if data.get("retCode") != 0:
                    break
                
                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break
                
                for k in klines:
                    ts = datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc)
                    ts = ts.replace(tzinfo=None)
                    
                    if ts.timestamp() * 1000 < start_time:
                        continue
                    
                    candle = {
                        'timestamp': ts,
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'turnover': float(k[6])
                    }
                    candles.append(candle)
                
                # Оновлюємо для наступного запиту
                oldest_ts = int(klines[-1][0])
                current_end = oldest_ts - 1
                
                if len(klines) < 1000:
                    break
                    
            except Exception as e:
                print(f"⚠️  помилка завантаження {symbol}: {e}")
                break
        
        # Сортуємо за часом
        candles.sort(key=lambda x: x['timestamp'])
        return candles[-self.hours_back * 60:]  # Обрізаємо до потрібної кількості

# =============================================================================
# 🔬 АНАЛІЗ РИНКУ ТА ВИЯВЛЕННЯ ПАТТЕРНІВ
# =============================================================================

class MarketPatternAnalyzer:
    """Аналізує ринкові паттерни та визначає оптимальні параметри"""
    
    def __init__(self, market_data: Dict[str, List[Dict]]):
        self.market_data = market_data
        self.regimes = []
        self.patterns = {}
    
    def analyze_market_regimes(self) -> List[MarketRegime]:
        """Аналізує ринкові режими"""
        print(f"\n🔍 АНАЛІЗ РИНКОВИХ РЕЖИМІВ")
        print("-" * 50)
        
        regimes = []
        
        # Аналіз загального ринку
        overall_trend = self._analyze_overall_trend()
        volatility_regime = self._analyze_volatility_regime()
        momentum_regime = self._analyze_momentum_regime()
        
        # Визначення головного режиму
        if volatility_regime["regime"] == "HIGH" and abs(overall_trend["strength"]) < 0.3:
            main_regime = "HIGH_VOL_SIDEWAYS"
        elif overall_trend["strength"] > 0.5:
            main_regime = "UPTREND"
        elif overall_trend["strength"] < -0.5:
            main_regime = "DOWNTREND"
        elif volatility_regime["regime"] == "LOW":
            main_regime = "LOW_VOL_SIDEWAYS"
        else:
            main_regime = "SIDEWAYS"
        
        # Створюємо об'єкт режиму
        regime = MarketRegime(
            name=main_regime,
            detected_at=datetime.now(),
            strength=max(abs(overall_trend["strength"]), volatility_regime["score"]),
            characteristics={
                "overall_trend": overall_trend,
                "volatility": volatility_regime,
                "momentum": momentum_regime,
                "avg_price_change": self._calculate_avg_price_change(),
                "volume_profile": self._analyze_volume_profile()
            }
        )
        
        regimes.append(regime)
        
        # Виводимо результати
        print(f"  🎯 ГОЛОВНИЙ РЕЖИМ: {main_regime}")
        print(f"     • Тренд: {overall_trend['direction']} (сила: {overall_trend['strength']:.2f})")
        print(f"     • Волатильність: {volatility_regime['regime']} (показник: {volatility_regime['score']:.1f})")
        print(f"     • Моментум: {momentum_regime['regime']}")
        
        # Додаємо рекомендації
        self._add_regime_recommendations(regime)
        
        self.regimes = regimes
        return regimes
    
    def _analyze_overall_trend(self) -> Dict[str, Any]:
        """Аналізує загальний тренд ринку"""
        all_changes = []
        
        for symbol, candles in self.market_data.items():
            if len(candles) >= 100:
                price_change = (candles[-1]['close'] - candles[0]['close']) / candles[0]['close'] * 100
                all_changes.append(price_change)
        
        if not all_changes:
            return {"direction": "NEUTRAL", "strength": 0.0}
        
        avg_change = np.mean(all_changes)
        
        if abs(avg_change) < 1.0:
            direction = "NEUTRAL"
            strength = 0.0
        elif avg_change > 0:
            direction = "UP"
            strength = min(abs(avg_change) / 5.0, 1.0)
        else:
            direction = "DOWN"
            strength = min(abs(avg_change) / 5.0, 1.0)
        
        return {"direction": direction, "strength": strength}
    
    def _analyze_volatility_regime(self) -> Dict[str, Any]:
        """Аналізує режим волатильності"""
        volatilities = []
        
        for symbol, candles in self.market_data.items():
            if len(candles) >= 50:
                returns = []
                for i in range(1, min(50, len(candles))):
                    if candles[i-1]['close'] > 0:
                        ret = abs((candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close'])
                        returns.append(ret)
                
                if returns:
                    vol = np.std(returns) * 100  # Волатильність у відсотках
                    volatilities.append(vol)
        
        if not volatilities:
            return {"regime": "MEDIUM", "score": 50.0}
        
        avg_vol = np.mean(volatilities)
        
        if avg_vol < 0.5:
            regime = "VERY_LOW"
            score = 20.0
        elif avg_vol < 1.0:
            regime = "LOW"
            score = 35.0
        elif avg_vol < 2.0:
            regime = "MEDIUM"
            score = 50.0
        elif avg_vol < 4.0:
            regime = "HIGH"
            score = 70.0
        else:
            regime = "EXTREME"
            score = 90.0
        
        return {"regime": regime, "score": score}
    
    def _analyze_momentum_regime(self) -> Dict[str, Any]:
        """Аналізує режим моментуму"""
        # Спрощений аналіз моментуму
        price_changes = []
        
        for symbol, candles in self.market_data.items():
            if len(candles) >= 20:
                # Швидкість зміни ціни
                short_change = (candles[-1]['close'] - candles[-5]['close']) / candles[-5]['close'] * 100
                medium_change = (candles[-1]['close'] - candles[-20]['close']) / candles[-20]['close'] * 100
                
                # Моментум = короткострокова зміна / довгострокова зміна
                if abs(medium_change) > 0.1:
                    momentum = short_change / medium_change
                    price_changes.append(momentum)
        
        if not price_changes:
            return {"regime": "NEUTRAL"}
        
        avg_momentum = np.mean(price_changes)
        
        if avg_momentum > 1.5:
            return {"regime": "ACCELERATING"}
        elif avg_momentum > 1.0:
            return {"regime": "STRONG"}
        elif avg_momentum > 0.5:
            return {"regime": "MODERATE"}
        elif avg_momentum > 0:
            return {"regime": "WEAK"}
        else:
            return {"regime": "DECELERATING"}
    
    def _calculate_avg_price_change(self) -> float:
        """Розраховує середню зміну ціни"""
        changes = []
        
        for symbol, candles in self.market_data.items():
            if len(candles) >= 10:
                for i in range(1, min(10, len(candles))):
                    if candles[i-1]['close'] > 0:
                        change = abs(candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close'] * 100
                        changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _analyze_volume_profile(self) -> Dict[str, float]:
        """Аналізує профіль об'ємів"""
        volumes = []
        
        for symbol, candles in self.market_data.items():
            if candles:
                avg_volume = np.mean([c['volume'] for c in candles[-100:]]) if len(candles) >= 100 else np.mean([c['volume'] for c in candles])
                volumes.append(avg_volume)
        
        if not volumes:
            return {"avg_volume": 0.0}
        
        return {
            "avg_volume": np.mean(volumes),
            "volume_trend": self._analyze_volume_trend()
        }
    
    def _analyze_volume_trend(self) -> str:
        """Аналізує тренд об'ємів"""
        volume_changes = []
        
        for symbol, candles in self.market_data.items():
            if len(candles) >= 20:
                recent_vol = np.mean([c['volume'] for c in candles[-10:]])
                older_vol = np.mean([c['volume'] for c in candles[-20:-10]])
                
                if older_vol > 0:
                    change = (recent_vol - older_vol) / older_vol
                    volume_changes.append(change)
        
        if not volume_changes:
            return "STABLE"
        
        avg_change = np.mean(volume_changes)
        
        if avg_change > 0.3:
            return "INCREASING"
        elif avg_change < -0.3:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _add_regime_recommendations(self, regime: MarketRegime):
        """Додає рекомендації для режиму"""
        recommendations = {
            "SIDEWAYS": {
                "min_tp_pct": 0.002,  # 0.2%
                "max_tp_pct": 0.005,  # 0.5%
                "min_sl_pct": 0.0015, # 0.15%
                "max_hold_minutes": 30,
                "min_imbalance": 35.0,
                "min_momentum": 60.0
            },
            "UPTREND": {
                "min_tp_pct": 0.008,  # 0.8%
                "max_tp_pct": 0.015,  # 1.5%
                "min_sl_pct": 0.005,  # 0.5%
                "max_hold_minutes": 60,
                "min_imbalance": 25.0,
                "min_momentum": 40.0
            },
            "DOWNTREND": {
                "min_tp_pct": 0.008,
                "max_tp_pct": 0.015,
                "min_sl_pct": 0.005,
                "max_hold_minutes": 60,
                "min_imbalance": 25.0,
                "min_momentum": 40.0
            },
            "HIGH_VOL_SIDEWAYS": {
                "min_tp_pct": 0.005,  # 0.5%
                "max_tp_pct": 0.010,  # 1.0%
                "min_sl_pct": 0.003,  # 0.3%
                "max_hold_minutes": 20,
                "min_imbalance": 40.0,
                "min_momentum": 70.0
            },
            "LOW_VOL_SIDEWAYS": {
                "min_tp_pct": 0.0015, # 0.15%
                "max_tp_pct": 0.003,  # 0.3%
                "min_sl_pct": 0.001,  # 0.1%
                "max_hold_minutes": 40,
                "min_imbalance": 30.0,
                "min_momentum": 50.0
            }
        }
        
        regime.optimal_params = recommendations.get(regime.name, recommendations["SIDEWAYS"])
        
        print(f"\n  💡 РЕКОМЕНДАЦІЇ ДЛЯ '{regime.name}':")
        for key, value in regime.optimal_params.items():
            if "pct" in key:
                print(f"     • {key}: {value*100:.2f}%")
            elif "imbalance" in key or "momentum" in key:
                print(f"     • {key}: {value}")
            elif "hold" in key:
                print(f"     • {key}: {value} хв")

# =============================================================================
# ⚙️ ОПТИМІЗАЦІЙНИЙ ДВИГУН
# =============================================================================

class StrategyOptimizer:
    """Двигун оптимізації стратегії"""
    
    def __init__(self, market_data: Dict[str, List[Dict]], signals: List[Dict]):
        self.market_data = market_data
        self.signals = signals
        self.best_results = []
        self.optimization_history = []
    
    def run_comprehensive_optimization(self, regime: MarketRegime, max_configs: int = 1000) -> List[OptimizationResult]:
        """Запускає комплексну оптимізацію"""
        print(f"\n⚙️  ЗАПУСК КОМПЛЕКСНОЇ ОПТИМІЗАЦІЇ")
        print(f"   • Режим: {regime.name}")
        print(f"   • Макс. конфігурацій: {max_configs}")
        print(f"   • Цілі: WR ≥ {TARGET_WIN_RATE}%, PF ≥ {TARGET_PROFIT_FACTOR}, PnL ≥ ${TARGET_TOTAL_PNL}")
        print("-" * 70)
        
        # Генеруємо конфігурації на основі рекомендацій режиму
        configs = self._generate_configurations(regime, max_configs)
        
        print(f"🔧 ЗГЕНЕРОВАНО {len(configs)} КОНФІГУРАЦІЙ ДЛЯ ТЕСТУ")
        
        # Запускаємо паралельне тестування
        results = self._test_configurations_parallel(configs, max_workers=4)
        
        # Сортуємо за результатами
        results.sort(key=lambda r: r.score, reverse=True)
        
        # Вибираємо кращі
        top_results = results[:20]
        
        print(f"\n🏆 ТОП-20 КОНФІГУРАЦІЙ:")
        print("-" * 80)
        
        for i, result in enumerate(top_results[:10], 1):
            print(f"  {i:2d}. Score: {result.score:6.2f} | "
                  f"WR: {result.win_rate:5.1f}% | "
                  f"PF: {result.profit_factor:5.2f} | "
                  f"PnL: ${result.total_net_pnl:6.2f} | "
                  f"Trades: {result.trades_count:3d}")
        
        self.best_results = top_results
        return top_results
    
    def _generate_configurations(self, regime: MarketRegime, max_configs: int) -> List[StrategyConfig]:
        """Генерує конфігурації для оптимізації"""
        configs = []
        
        # Базові параметри з рекомендацій режиму
        base_params = regime.optimal_params
        
        # Діапазони для оптимізації
        param_ranges = {
            'min_composite': [0.35, 0.40, 0.45, 0.50, 0.55],
            'min_strength': [3, 4, 5],
            'min_imbalance': [20, 25, 30, 35, 40, 45, 50],
            'min_momentum': [30, 40, 50, 60, 70],
            'max_momentum': [70, 75, 80, 85, 90],
            'min_ohara': [4, 5, 6, 7],
            'tp_pct': self._generate_tp_range(base_params['min_tp_pct'], base_params['max_tp_pct']),
            'sl_pct': self._generate_sl_range(base_params['min_sl_pct']),
            'max_hold_minutes': [20, 30, 40, 50, 60]
        }
        
        # Обмежуємо кількість комбінацій
        sampled_configs = self._sample_parameter_combinations(param_ranges, max_configs)
        
        for params in sampled_configs:
            config = StrategyConfig(**params)
            configs.append(config)
        
        return configs
    
    def _generate_tp_range(self, min_tp: float, max_tp: float) -> List[float]:
        """Генерує діапазон TP значень"""
        steps = 5
        step = (max_tp - min_tp) / (steps - 1) if steps > 1 else 0
        return [min_tp + i * step for i in range(steps)]
    
    def _generate_sl_range(self, min_sl: float) -> List[float]:
        """Генерує діапазон SL значень"""
        return [min_sl * 0.8, min_sl, min_sl * 1.2, min_sl * 1.5]
    
    def _sample_parameter_combinations(self, param_ranges: Dict, max_samples: int) -> List[Dict]:
        """Вибірка комбінацій параметрів"""
        # Генеруємо всі можливі комбінації
        all_combinations = list(itertools.product(*param_ranges.values()))
        
        # Обмежуємо кількість
        if len(all_combinations) <= max_samples:
            indices = range(len(all_combinations))
        else:
            # Випадкова вибірка
            indices = np.random.choice(len(all_combinations), max_samples, replace=False)
        
        # Конвертуємо в список словників
        samples = []
        param_keys = list(param_ranges.keys())
        
        for idx in indices:
            params = {}
            for i, key in enumerate(param_keys):
                params[key] = all_combinations[idx][i]
            samples.append(params)
        
        return samples
    
    def _test_configurations_parallel(self, configs: List[StrategyConfig], max_workers: int = 4) -> List[OptimizationResult]:
        """Тестує конфігурації паралельно"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаємо тестування
            future_to_config = {
                executor.submit(self._test_single_configuration, config): config 
                for config in configs
            }
            
            completed = 0
            total = len(configs)
            
            for future in as_completed(future_to_config):
                completed += 1
                
                if completed % 10 == 0:
                    print(f"  🔄 Протестовано {completed}/{total} конфігурацій...")
                
                try:
                    result = future.result()
                    if result and result.trades_count >= 10:  # Мінімум 10 угод
                        results.append(result)
                except Exception as e:
                    print(f"⚠️  Помилка тестування: {e}")
        
        return results
    
    def _test_single_configuration(self, config: StrategyConfig) -> Optional[OptimizationResult]:
        """Тестує одну конфігурацію стратегії"""
        try:
            # Фільтруємо сигнали за конфігурацією
            filtered_signals = self._filter_signals(config)
            
            if len(filtered_signals) < 10:  # Мінімум 10 сигналів
                return None
            
            # Симулюємо угоди
            trades = self._simulate_trades(filtered_signals, config)
            
            if len(trades) < 10:  # Мінімум 10 угод
                return None
            
            # Аналізуємо результати
            result = self._analyze_trading_results(trades, config)
            
            return result
            
        except Exception as e:
            print(f"⚠️  Помилка тестування {config.get_id()}: {e}")
            return None
    
    def _filter_signals(self, config: StrategyConfig) -> List[Dict]:
        """Фільтрує сигнали за параметрами конфігурації"""
        filtered = []
        
        for signal in self.signals:
            # Перевіряємо основні критерії
            if (signal.get('composite', 0) >= config.min_composite and
                signal.get('strength', 0) >= config.min_strength and
                abs(signal.get('imbalance', 0)) >= config.min_imbalance and
                abs(signal.get('momentum', 0)) >= config.min_momentum and
                abs(signal.get('momentum', 0)) <= config.max_momentum and
                signal.get('ohara_score', 0) >= config.min_ohara):
                
                filtered.append(signal)
        
        return filtered
    
    def _simulate_trades(self, signals: List[Dict], config: StrategyConfig) -> List[Dict]:
        """Симулює угоди на основі сигналів"""
        trades = []
        
        for signal in signals:
            symbol = signal.get('symbol')
            if symbol not in self.market_data:
                continue
            
            candles = self.market_data[symbol]
            if len(candles) < 100:
                continue
            
            # Знаходимо свічку для входу
            signal_time = signal.get('timestamp')
            entry_candle = None
            entry_idx = -1
            
            for i, candle in enumerate(candles):
                if abs((candle['timestamp'] - signal_time).total_seconds()) < 300:  # 5 хвилин
                    entry_candle = candle
                    entry_idx = i
                    break
            
            if not entry_candle:
                continue
            
            # Симулюємо угоду
            trade = self._simulate_single_trade(signal, entry_candle, entry_idx, candles, config)
            if trade:
                trades.append(trade)
        
        return trades
    
    def _simulate_single_trade(self, signal: Dict, entry_candle: Dict, 
                              entry_idx: int, candles: List[Dict], 
                              config: StrategyConfig) -> Optional[Dict]:
        """Симулює одну угоду"""
        try:
            entry_price = entry_candle['close']
            is_long = signal.get('action') == 'BUY'
            
            # Розраховуємо TP/SL
            if is_long:
                tp_price = entry_price * (1 + config.tp_pct)
                sl_price = entry_price * (1 - config.sl_pct)
            else:
                tp_price = entry_price * (1 - config.tp_pct)
                sl_price = entry_price * (1 + config.sl_pct)
            
            # Симулюємо утримання
            exit_price = None
            exit_reason = "TIME_EXIT"
            exit_idx = entry_idx
            
            max_candles = min(config.max_hold_minutes, len(candles) - entry_idx - 1)
            
            for i in range(entry_idx + 1, entry_idx + max_candles + 1):
                candle = candles[i]
                
                if is_long:
                    if candle['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL_HIT"
                        exit_idx = i
                        break
                    elif candle['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP_HIT"
                        exit_idx = i
                        break
                else:
                    if candle['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL_HIT"
                        exit_idx = i
                        break
                    elif candle['low'] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP_HIT"
                        exit_idx = i
                        break
            
            # TIME EXIT
            if not exit_price:
                exit_idx = min(entry_idx + max_candles, len(candles) - 1)
                exit_candle = candles[exit_idx]
                exit_price = exit_candle['close']
            
            # Розраховуємо PnL
            if is_long:
                gross_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                gross_pnl_pct = (entry_price - exit_price) / entry_price
            
            # Комісії
            fees = (entry_price + exit_price) * config.position_size_usd / entry_price * BYBIT_FEE
            net_pnl_pct = gross_pnl_pct - (fees / config.position_size_usd)
            
            # Створюємо об'єкт угоди
            trade = {
                'symbol': signal.get('symbol'),
                'direction': 'LONG' if is_long else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_candle['timestamp'],
                'exit_time': candles[exit_idx]['timestamp'],
                'duration_minutes': (candles[exit_idx]['timestamp'] - entry_candle['timestamp']).total_seconds() / 60,
                'gross_pnl_pct': gross_pnl_pct,
                'fees': fees,
                'net_pnl_pct': net_pnl_pct,
                'net_pnl_usd': net_pnl_pct * config.position_size_usd,
                'exit_reason': exit_reason,
                'signal': signal
            }
            
            return trade
            
        except Exception as e:
            return None
    
    def _analyze_trading_results(self, trades: List[Dict], config: StrategyConfig) -> OptimizationResult:
        """Аналізує результати торгівлі"""
        if not trades:
            return None
        
        # Основна статистика
        winners = [t for t in trades if t['net_pnl_usd'] > 0]
        losers = [t for t in trades if t['net_pnl_usd'] <= 0]
        
        total_gross_pnl = sum(t['gross_pnl_pct'] * config.position_size_usd for t in trades)
        total_fees = sum(t['fees'] for t in trades)
        total_net_pnl = sum(t['net_pnl_usd'] for t in trades)
        
        win_rate = len(winners) / len(trades) * 100 if trades else 0
        
        # Profit Factor
        total_winner_pnl = sum(t['net_pnl_usd'] for t in winners)
        total_loser_pnl = abs(sum(t['net_pnl_usd'] for t in losers))
        profit_factor = total_winner_pnl / total_loser_pnl if total_loser_pnl > 0 else 0
        
        # Статистика по символах
        by_symbol = defaultdict(list)
        for t in trades:
            by_symbol[t['symbol']].append(t)
        
        symbol_stats = {}
        for symbol, symbol_trades in by_symbol.items():
            symbol_winners = [t for t in symbol_trades if t['net_pnl_usd'] > 0]
            symbol_net_pnl = sum(t['net_pnl_usd'] for t in symbol_trades)
            symbol_stats[symbol] = {
                'trades': len(symbol_trades),
                'winners': len(symbol_winners),
                'win_rate': len(symbol_winners) / len(symbol_trades) * 100 if symbol_trades else 0,
                'total_net_pnl': symbol_net_pnl
            }
        
        # Статистика по причинах виходу
        by_exit = defaultdict(list)
        for t in trades:
            by_exit[t['exit_reason']].append(t)
        
        exit_stats = {}
        for reason, reason_trades in by_exit.items():
            reason_winners = [t for t in reason_trades if t['net_pnl_usd'] > 0]
            exit_stats[reason] = {
                'count': len(reason_trades),
                'win_rate': len(reason_winners) / len(reason_trades) * 100 if reason_trades else 0,
                'avg_pnl': sum(t['net_pnl_usd'] for t in reason_trades) / len(reason_trades) if reason_trades else 0
            }
        
        # Найкращі/найгірші угоди
        best_trades = sorted(trades, key=lambda x: x['net_pnl_usd'], reverse=True)[:5]
        worst_trades = sorted(trades, key=lambda x: x['net_pnl_usd'])[:5]
        
        # Максимальні поспіль результати
        max_consecutive_wins = self._calculate_max_consecutive([t['net_pnl_usd'] > 0 for t in trades])
        max_consecutive_losses = self._calculate_max_consecutive([t['net_pnl_usd'] <= 0 for t in trades])
        
        # Створюємо результат
        result = OptimizationResult(
            config=config,
            trades_count=len(trades),
            winners=len(winners),
            losers=len(losers),
            win_rate=win_rate,
            total_gross_pnl=total_gross_pnl,
            total_fees=total_fees,
            total_net_pnl=total_net_pnl,
            profit_factor=profit_factor,
            avg_duration_min=sum(t['duration_minutes'] for t in trades) / len(trades) if trades else 0,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            by_symbol=dict(symbol_stats),
            by_exit_reason=exit_stats,
            best_trades=[{
                'symbol': t['symbol'],
                'direction': t['direction'],
                'net_pnl_usd': t['net_pnl_usd'],
                'net_pnl_pct': t['net_pnl_pct'] * 100,
                'duration_min': t['duration_minutes'],
                'exit_reason': t['exit_reason']
            } for t in best_trades],
            worst_trades=[{
                'symbol': t['symbol'],
                'direction': t['direction'],
                'net_pnl_usd': t['net_pnl_usd'],
                'net_pnl_pct': t['net_pnl_pct'] * 100,
                'duration_min': t['duration_minutes'],
                'exit_reason': t['exit_reason']
            } for t in worst_trades]
        )
        
        return result
    
    def _calculate_max_consecutive(self, results: List[bool]) -> int:
        """Розраховує максимальну кількість поспіль True або False"""
        max_count = 0
        current_count = 0
        current_value = None
        
        for value in results:
            if current_value is None:
                current_value = value
                current_count = 1
            elif value == current_value:
                current_count += 1
            else:
                current_value = value
                current_count = 1
            
            max_count = max(max_count, current_count)
        
        return max_count

# =============================================================================
# 📈 ВИВЕДЕННЯ РЕЗУЛЬТАТІВ ТА ЗБЕРЕЖЕННЯ
# =============================================================================

class OptimizationReporter:
    """Генерує звіти про оптимізацію"""
    
    @staticmethod
    def print_optimization_summary(best_results: List[OptimizationResult], regime: MarketRegime):
        """Друкує підсумок оптимізації"""
        print(f"\n" + "=" * 100)
        print(f"📊 ПІДСУМОК ОПТИМІЗАЦІЇ ДЛЯ РЕЖИМУ: {regime.name}")
        print("=" * 100)
        
        if not best_results:
            print("❌ Немає результатів для аналізу")
            return
        
        # Топ-3 стратегії
        top_3 = best_results[:3]
        
        print(f"\n🏆 ТОП-3 СТРАТЕГІЇ:")
        
        for i, result in enumerate(top_3, 1):
            print(f"\n  {i}️⃣  СТРАТЕГІЯ #{i} (Score: {result.score:.2f}):")
            print(f"     {'─' * 60}")
            
            # Основні показники
            print(f"     📊 РЕЗУЛЬТАТИ:")
            print(f"       • Угод: {result.trades_count}")
            print(f"       • Win Rate: {result.win_rate:.1f}%")
            print(f"       • Total PnL: ${result.total_net_pnl:.2f}")
            print(f"       • Profit Factor: {result.profit_factor:.2f}")
            print(f"       • Комісії: ${result.total_fees:.2f}")
            print(f"       • Середня тривалість: {result.avg_duration_min:.1f} хв")
            
            # Параметри стратегії
            print(f"\n     ⚙️  ПАРАМЕТРИ:")
            config = result.config
            print(f"       • Min Composite: {config.min_composite:.2f}")
            print(f"       • Min Strength: {config.min_strength}")
            print(f"       • Min Imbalance: {config.min_imbalance:.0f}")
            print(f"       • Min Momentum: {config.min_momentum:.0f}")
            print(f"       • Max Momentum: {config.max_momentum:.0f}")
            print(f"       • Min O'Hara: {config.min_ohara}")
            print(f"       • TP: {config.tp_pct*100:.2f}%")
            print(f"       • SL: {config.sl_pct*100:.2f}%")
            print(f"       • Max Hold: {config.max_hold_minutes} хв")
            print(f"       • RR Ratio: {config.tp_pct/config.sl_pct:.2f}")
            
            # Статистика по причинах виходу
            print(f"\n     📉 ВИХІД З ПОЗИЦІЙ:")
            for reason, stats in result.by_exit_reason.items():
                print(f"       • {reason}: {stats['count']} угод, "
                      f"WR: {stats['win_rate']:.1f}%, "
                      f"Avg PnL: ${stats['avg_pnl']:.3f}")
        
        # Перевіряємо, чи є стратегії, що відповідають цілям
        meeting_targets = [r for r in best_results if r.meets_targets]
        
        if meeting_targets:
            print(f"\n✅ ЗНАЙДЕНО {len(meeting_targets)} СТРАТЕГІЙ, ЩО ВІДПОВІДАЮТЬ ЦІЛЯМ!")
            best = meeting_targets[0]
            print(f"\n🎯 НАЙКРАЩА СТРАТЕГІЯ ДЛЯ ВПРОВАДЖЕННЯ:")
            print(f"   • Win Rate: {best.win_rate:.1f}% (ціль: ≥{TARGET_WIN_RATE}%)")
            print(f"   • Profit Factor: {best.profit_factor:.2f} (ціль: ≥{TARGET_PROFIT_FACTOR})")
            print(f"   • Total PnL: ${best.total_net_pnl:.2f} (ціль: ≥${TARGET_TOTAL_PNL})")
            print(f"   • Конфігурація ID: {best.config.get_id()}")
        else:
            print(f"\n⚠️  НЕ ЗНАЙДЕНО СТРАТЕГІЙ, ЩО ВІДПОВІДАЮТЬ ЦІЛЯМ")
            print(f"   Найкраща доступна стратегія:")
            best = best_results[0]
            print(f"   • Win Rate: {best.win_rate:.1f}% (ціль: ≥{TARGET_WIN_RATE}%)")
            print(f"   • Profit Factor: {best.profit_factor:.2f} (ціль: ≥{TARGET_PROFIT_FACTOR})")
            print(f"   • Total PnL: ${best.total_net_pnl:.2f} (ціль: ≥${TARGET_TOTAL_PNL})")
            
            # Рекомендації для покращення
            print(f"\n💡 РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ:")
            if best.win_rate < TARGET_WIN_RATE:
                print(f"   • Підвищити min_composite з {best.config.min_composite:.2f} до ≥0.45")
                print(f"   • Підвищити min_strength з {best.config.min_strength} до ≥4")
                print(f"   • Підвищити min_imbalance з {best.config.min_imbalance:.0f} до ≥35")
            
            if best.profit_factor < TARGET_PROFIT_FACTOR:
                print(f"   • Зменшити TP з {best.config.tp_pct*100:.2f}% до 0.3-0.5%")
                print(f"   • Зменшити SL з {best.config.sl_pct*100:.2f}% до 0.15-0.25%")
                print(f"   • Зменшити max_hold_minutes з {best.config.max_hold_minutes} до 30 хв")
    
    @staticmethod
    def save_optimization_results(best_results: List[OptimizationResult], regime: MarketRegime):
        """Зберігає результати оптимізації"""
        try:
            os.makedirs("logs/optimization", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Зберігаємо топ-10 стратегій
            top_strategies = best_results[:10]
            
            report_data = {
                'meta': {
                    'timestamp': datetime.now().isoformat(),
                    'market_regime': regime.name,
                    'regime_strength': regime.strength,
                    'regime_characteristics': regime.characteristics,
                    'targets': {
                        'win_rate': TARGET_WIN_RATE,
                        'profit_factor': TARGET_PROFIT_FACTOR,
                        'total_pnl': TARGET_TOTAL_PNL
                    }
                },
                'best_strategies': []
            }
            
            for i, result in enumerate(top_strategies, 1):
                strategy_data = {
                    'rank': i,
                    'score': result.score,
                    'meets_targets': result.meets_targets,
                    'performance': {
                        'trades_count': result.trades_count,
                        'win_rate': result.win_rate,
                        'total_net_pnl': result.total_net_pnl,
                        'profit_factor': result.profit_factor,
                        'total_fees': result.total_fees,
                        'avg_duration_min': result.avg_duration_min
                    },
                    'parameters': {
                        'min_composite': result.config.min_composite,
                        'min_strength': result.config.min_strength,
                        'min_imbalance': result.config.min_imbalance,
                        'min_momentum': result.config.min_momentum,
                        'max_momentum': result.config.max_momentum,
                        'min_ohara': result.config.min_ohara,
                        'tp_pct': result.config.tp_pct,
                        'sl_pct': result.config.sl_pct,
                        'max_hold_minutes': result.config.max_hold_minutes,
                        'config_id': result.config.get_id()
                    },
                    'exit_statistics': result.by_exit_reason,
                    'symbol_statistics': result.by_symbol
                }
                report_data['best_strategies'].append(strategy_data)
            
            # Зберігаємо JSON
            json_file = f"logs/optimization/optimization_report_{regime.name}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Результати оптимізації збережено: {json_file}")
            
            # Створюємо файл для впровадження
            if top_strategies:
                best = top_strategies[0]
                implementation_file = f"logs/optimization/best_strategy_{timestamp}.py"
                
                with open(implementation_file, 'w', encoding='utf-8') as f:
                    f.write(f'''
# 🎯 НАЙКРАЩА СТРАТЕГІЯ ДЛЯ РЕЖИМУ: {regime.name}
# Оптимізовано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Результати: WR={best.win_rate:.1f}%, PF={best.profit_factor:.2f}, PnL=${best.total_net_pnl:.2f}

BEST_STRATEGY_CONFIG = {{
    # Фільтри сигналів
    "min_composite": {best.config.min_composite},
    "min_strength": {best.config.min_strength},
    "min_imbalance": {best.config.min_imbalance},
    "min_momentum": {best.config.min_momentum},
    "max_momentum": {best.config.max_momentum},
    "min_ohara": {best.config.min_ohara},
    
    # Ризик-менеджмент
    "tp_pct": {best.config.tp_pct},
    "sl_pct": {best.config.sl_pct},
    "max_hold_minutes": {best.config.max_hold_minutes},
    "position_size_usd": {best.config.position_size_usd},
    
    # Статистика
    "expected_win_rate": {best.win_rate},
    "expected_profit_factor": {best.profit_factor},
    "config_id": "{best.config.get_id()}"
}}

# Інструкція для впровадження:
# 1. Додайте ці параметри до config/settings.py
# 2. Оновіть клас SignalSettings та RiskSettings
# 3. Перезапустіть бота
''')
                
                print(f"💡 Файл для впровадження: {implementation_file}")
                
        except Exception as e:
            print(f"❌ Помилка збереження результатів: {e}")

# =============================================================================
# 📥 ЗАВАНТАЖЕННЯ СИГНАЛІВ
# =============================================================================

class SignalLoader:
    """Завантажує та підготовлює сигнали для оптимізації"""
    
    @staticmethod
    def load_signals(hours_back: int = 48) -> List[Dict]:
        """Завантажує сигнали з CSV"""
        signals_path = "logs/signals.csv"
        
        if not os.path.exists(signals_path):
            print(f"❌ Файл {signals_path} не знайдено!")
            return []
        
        signals = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        print(f"\n📜 ЗАВАНТАЖЕННЯ СИГНАЛІВ...")
        
        try:
            with open(signals_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                for row in reader:
                    if len(row) < 15:
                        continue
                    
                    try:
                        # Парсимо timestamp
                        ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                        
                        if ts < cutoff_time:
                            continue
                        
                        # Тільки прийняті сигнали
                        if row[14].upper() != "YES":
                            continue
                        
                        # Створюємо об'єкт сигналу
                        signal = {
                            'timestamp': ts,
                            'symbol': row[1],
                            'action': row[2],
                            'strength': int(row[3]),
                            'composite': float(row[4]),
                            'ema': float(row[5]),
                            'imbalance': float(row[6]),
                            'momentum': float(row[7]),
                            'bayesian': row[8],
                            'large_orders': row[9],
                            'frequency': row[10],
                            'vol_confirm': row[11],
                            'ohara_score': int(row[12]),
                            'reason': row[13],
                            'accepted': True
                        }
                        
                        signals.append(signal)
                        
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"❌ Помилка завантаження сигналів: {e}")
        
        print(f"✅ Завантажено {len(signals)} сигналів за {hours_back} годин")
        
        # Статистика
        if signals:
            symbols_count = defaultdict(int)
            for s in signals:
                symbols_count[s['symbol']] += 1
            
            print(f"📊 Розподіл по символах (топ-5):")
            for symbol, count in sorted(symbols_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {symbol}: {count} сигналів")
        
        return signals

# =============================================================================
# 🚀 ОСНОВНИЙ КЛАС ОПТИМІЗАЦІЇ
# =============================================================================

class MarketContextOptimizationEngine:
    """Головний двигун оптимізації з ринковим контекстом"""
    
    def __init__(self, hours_back: int = 48):
        self.hours_back = hours_back
        self.data_loader = OptimizationDataLoader(hours_back)
        self.signal_loader = SignalLoader()
    
    def run_optimization(self, symbols: Optional[List[str]] = None, max_configs: int = 500):
        """Запускає повний процес оптимізації"""
        print("\n" + "=" * 100)
        print("🎯 MARKET-CONTEXT OPTIMIZATION ENGINE v4.0")
        print("=" * 100)
        
        # 1. Завантаження даних
        if not symbols:
            symbols = settings.pairs.trade_pairs
        
        print(f"\n1️⃣ 📊 ЗАВАНТАЖЕННЯ ДАНИХ ТА СИГНАЛІВ")
        print("-" * 70)
        
        market_data = self.data_loader.load_all_data(symbols)
        signals = self.signal_loader.load_signals(self.hours_back)
        
        if not market_data or not signals:
            print("❌ Недостатньо даних для оптимізації!")
            return
        
        # 2. Аналіз ринкового контексту
        print(f"\n2️⃣ 🔍 АНАЛІЗ РИНКОВОГО КОНТЕКСТУ")
        print("-" * 70)
        
        analyzer = MarketPatternAnalyzer(market_data)
        regimes = analyzer.analyze_market_regimes()
        
        if not regimes:
            print("❌ Не вдалося визначити ринковий режим!")
            return
        
        regime = regimes[0]  # Використовуємо головний режим
        
        # 3. Оптимізація стратегії
        print(f"\n3️⃣ ⚙️  ОПТИМІЗАЦІЯ СТРАТЕГІЇ")
        print("-" * 70)
        
        optimizer = StrategyOptimizer(market_data, signals)
        best_results = optimizer.run_comprehensive_optimization(regime, max_configs)
        
        if not best_results:
            print("❌ Не вдалося знайти прийнятні стратегії!")
            return
        
        # 4. Виведення результатів
        print(f"\n4️⃣ 📊 РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ")
        print("-" * 70)
        
        reporter = OptimizationReporter()
        reporter.print_optimization_summary(best_results, regime)
        
        # 5. Збереження результатів
        print(f"\n5️⃣ 💾 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
        print("-" * 70)
        
        reporter.save_optimization_results(best_results, regime)
        
        print("\n" + "=" * 100)
        print("✅ ОПТИМІЗАЦІЯ УСПІШНО ЗАВЕРШЕНА!")
        print("=" * 100)
        
        # 6. Рекомендації для впровадження
        self._print_implementation_guide(best_results[0] if best_results else None)

    def _print_implementation_guide(self, best_result: Optional[OptimizationResult]):
        """Друкує інструкцію для впровадження"""
        if not best_result:
            return
        
        print(f"\n🔧 ІНСТРУКЦІЯ ДЛЯ ВПРОВАДЖЕННЯ:")
        print("-" * 70)
        
        config = best_result.config
        
        print(f"\n1️⃣  Оновіть config/settings.py:")
        print(f"""
# У класі SignalSettings:
min_imbalance_for_entry = {config.min_imbalance}  # було {settings.signals.min_imbalance_for_entry}
min_momentum_for_entry = {config.min_momentum}    # було {settings.signals.min_momentum_for_entry}
max_momentum_for_entry = {config.max_momentum}    # було {settings.signals.max_momentum_for_entry}
min_ohara_for_entry = {config.min_ohara}          # було {settings.signals.min_ohara_for_entry}

# У класі RiskSettings:
min_tp_pct = {config.tp_pct}      # було {settings.risk.min_tp_pct}
min_sl_pct = {config.sl_pct}      # було {settings.risk.min_sl_pct}
base_position_lifetime_minutes = {config.max_hold_minutes}  # було {settings.risk.base_position_lifetime_minutes}
""")
        
        print(f"\n2️⃣  Очікувані результати:")
        print(f"   • Win Rate: {best_result.win_rate:.1f}%")
        print(f"   • Profit Factor: {best_result.profit_factor:.2f}")
        print(f"   • Середній PnL за угоду: ${best_result.total_net_pnl/best_result.trades_count:.2f}")
        
        print(f"\n3️⃣  Рекомендації:")
        print(f"   • Запустіть бота з новими налаштуваннями на 1-2 години")
        print(f"   • Моніторьте результати в реальному часі")
        print(f"   • При необхідності коригуйте параметри")

# =============================================================================
# 🚀 MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="🎯 Market-Context Optimization Engine v4.0")
    parser.add_argument("--optimize", action="store_true", 
                       help="Запустити оптимізацію стратегії")
    parser.add_argument("--hours", type=int, default=48,
                       help="Годин даних для аналізу")
    parser.add_argument("--symbols", type=str,
                       help="Символи для аналізу (через кому)")
    parser.add_argument("--max-configs", type=int, default=500,
                       help="Максимальна кількість конфігурацій для тесту")
    parser.add_argument("--target-wr", type=float, default=55.0,
                       help="Цільовий Win Rate (%)")
    parser.add_argument("--target-pf", type=float, default=1.5,
                       help="Цільовий Profit Factor")
    parser.add_argument("--target-pnl", type=float, default=50.0,
                       help="Цільовий Total PnL ($)")
    
    args = parser.parse_args()
    
    # Оновлюємо цільові показники
    global TARGET_WIN_RATE, TARGET_PROFIT_FACTOR, TARGET_TOTAL_PNL
    TARGET_WIN_RATE = args.target_wr
    TARGET_PROFIT_FACTOR = args.target_pf
    TARGET_TOTAL_PNL = args.target_pnl
    
    print(f"\n⚙️  КОНФІГУРАЦІЯ ОПТИМІЗАЦІЙНОГО ДВИГУНА v4.0:")
    print(f"  • Період даних: {args.hours} годин")
    print(f"  • Макс. конфігурацій: {args.max_configs}")
    print(f"  • Цільові показники:")
    print(f"      Win Rate: ≥ {TARGET_WIN_RATE}%")
    print(f"      Profit Factor: ≥ {TARGET_PROFIT_FACTOR}")
    print(f"      Total PnL: ≥ ${TARGET_TOTAL_PNL}")
    
    # Парсимо символи
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = settings.pairs.trade_pairs
    
    print(f"  • Символи для аналізу: {len(symbols)} пар")
    
    # Запускаємо оптимізацію
    engine = MarketContextOptimizationEngine(hours_back=args.hours)
    engine.run_optimization(symbols=symbols, max_configs=args.max_configs)

if __name__ == "__main__":
    main()