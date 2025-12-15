# analysis/imbalance.py
import time
import math
import statistics
from typing import Dict, Any, List, Tuple
from collections import deque
from config.settings import settings
from data.storage import DataStorage
from utils.logger import logger


class BayesianUpdater:
    """Баєсівський оновлювач для передбачення напрямку"""
    
    def __init__(self):
        self.prior_bullish = 0.5
        self.prior_bearish = 0.5
        self.confidence_threshold = 0.6
        
    def update_beliefs(self, signal: str, strength: float):
        """Оновлення віри на основі сигналу"""
        if signal == "BUY":
            self.prior_bullish += strength * settings.ohara.bayesian_update_step
            self.prior_bearish *= settings.ohara.bayesian_decay_factor
        elif signal == "SELL":
            self.prior_bearish += strength * settings.ohara.bayesian_update_step
            self.prior_bullish *= settings.ohara.bayesian_decay_factor
        
        # Нормалізація
        total = self.prior_bullish + self.prior_bearish
        if total > 0:
            self.prior_bullish /= total
            self.prior_bearish /= total
        
        # Визначення сигналу
        if self.prior_bullish >= settings.ohara.bayesian_bullish_threshold:
            return "BULLISH", self.prior_bullish
        elif self.prior_bearish >= settings.ohara.bayesian_bearish_threshold:
            return "BEARISH", self.prior_bearish
        else:
            return "NEUTRAL", max(self.prior_bullish, self.prior_bearish)


class ClusterAnalyzer:
    """Аналіз кластерів ліквідності"""
    
    def analyze_clusters(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], 
                        current_price: float) -> Dict[str, Any]:
        """Аналіз кластерів на основі глибини"""
        if not bids or not asks:
            return {
                "poc_price": current_price,
                "poc_distance_pct": 0.0,
                "support_cluster_strength": 0.0,
                "resistance_cluster_strength": 0.0,
                "cluster_imbalance": 0.0
            }
        
        # Пошук POC (Point of Control) - рівень з найбільшою ліквідністю
        all_levels = bids + asks
        poc_level = max(all_levels, key=lambda x: x[1])
        poc_price = poc_level[0]
        poc_distance_pct = ((poc_price - current_price) / current_price) * 100
        
        # Аналіз кластерів підтримки/опору
        bid_clusters = self._find_clusters(bids, threshold_pct=0.5)
        ask_clusters = self._find_clusters(asks, threshold_pct=0.5)
        
        support_cluster = max(bid_clusters, key=lambda x: x['strength']) if bid_clusters else None
        resistance_cluster = max(ask_clusters, key=lambda x: x['strength']) if ask_clusters else None
        
        return {
            "poc_price": poc_price,
            "poc_distance_pct": poc_distance_pct,
            "support_cluster_strength": support_cluster['strength'] if support_cluster else 0.0,
            "resistance_cluster_strength": resistance_cluster['strength'] if resistance_cluster else 0.0,
            "cluster_imbalance": (support_cluster['strength'] if support_cluster else 0) - 
                               (resistance_cluster['strength'] if resistance_cluster else 0)
        }
    
    def _find_clusters(self, levels: List[Tuple[float, float]], threshold_pct: float) -> List[Dict]:
        """Пошук кластерів ліквідності"""
        if not levels:
            return []
        
        clusters = []
        current_cluster = {'prices': [levels[0][0]], 'sizes': [levels[0][1]]}
        
        for price, size in levels[1:]:
            last_price = current_cluster['prices'][-1]
            if abs(price - last_price) / last_price * 100 <= threshold_pct:
                current_cluster['prices'].append(price)
                current_cluster['sizes'].append(size)
            else:
                if len(current_cluster['prices']) > 1:
                    clusters.append({
                        'avg_price': sum(current_cluster['prices']) / len(current_cluster['prices']),
                        'total_size': sum(current_cluster['sizes']),
                        'strength': sum(current_cluster['sizes']) / len(current_cluster['prices'])
                    })
                current_cluster = {'prices': [price], 'sizes': [size]}
        
        if len(current_cluster['prices']) > 1:
            clusters.append({
                'avg_price': sum(current_cluster['prices']) / len(current_cluster['prices']),
                'total_size': sum(current_cluster['sizes']),
                'strength': sum(current_cluster['sizes']) / len(current_cluster['prices'])
            })
        
        return clusters


class DepthAnalyzer:
    """Аналіз глибини ринку"""
    
    def analyze_depth(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], 
                     current_price: float) -> Dict[str, Any]:
        """Аналіз підтримки/опору на основі глибини"""
        if not bids or not asks:
            return {
                "support_resistance_ratio": 0.5,
                "liquidity_imbalance": 0.0,
                "bid_ask_wall_ratio": 1.0,
                "depth_distribution": {"bids": 0, "asks": 0}
            }
        
        # Розрахунок загальної ліквідності
        total_bid_liquidity = sum(size for _, size in bids)
        total_ask_liquidity = sum(size for _, size in asks)
        
        # Співвідношення підтримки/опору
        support_resistance_ratio = 0.5
        if total_bid_liquidity + total_ask_liquidity > 0:
            support_resistance_ratio = total_bid_liquidity / (total_bid_liquidity + total_ask_liquidity)
        
        # Імбаланс ліквідності
        liquidity_imbalance = 0.0
        if total_bid_liquidity > 0 and total_ask_liquidity > 0:
            liquidity_imbalance = ((total_bid_liquidity - total_ask_liquidity) / 
                                 (total_bid_liquidity + total_ask_liquidity)) * 100
        
        # Аналіз "стіни" bid/ask
        best_bid_size = bids[0][1] if bids else 0
        best_ask_size = asks[0][1] if asks else 0
        bid_ask_wall_ratio = best_bid_size / best_ask_size if best_ask_size > 0 else 1.0
        
        return {
            "support_resistance_ratio": support_resistance_ratio,
            "liquidity_imbalance": liquidity_imbalance,
            "bid_ask_wall_ratio": bid_ask_wall_ratio,
            "depth_distribution": {
                "bids": total_bid_liquidity,
                "asks": total_ask_liquidity
            }
        }


class TradeImbalanceAnalyzer:
    """Аналіз імбалансу трейдів"""
    
    def __init__(self):
        self.trade_history: Dict[str, deque] = {}
        
    def update_trade_imbalance(self, symbol: str, trade_price: float, trade_size: float, 
                              trade_side: str, current_price: float) -> Dict[str, Any]:
        """Оновлення імбалансу на основі останніх трейдів"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=100)
        
        self.trade_history[symbol].append({
            'price': trade_price,
            'size': trade_size,
            'side': trade_side,
            'timestamp': time.time()
        })
        
        # Розрахунок імбалансу за останні N трейдів
        recent_trades = list(self.trade_history[symbol])[-50:]  # останні 50 трейдів
        
        if len(recent_trades) < 10:
            return {
                "trade_imbalance_score": 0.0,
                "buy_pressure": 0.0,
                "sell_pressure": 0.0,
                "aggression_imbalance": 0.0
            }
        
        buy_volume = sum(t['size'] for t in recent_trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['size'] for t in recent_trades if t['side'].lower() == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {
                "trade_imbalance_score": 0.0,
                "buy_pressure": 0.0,
                "sell_pressure": 0.0,
                "aggression_imbalance": 0.0
            }
        
        buy_ratio = buy_volume / total_volume
        trade_imbalance_score = (buy_ratio - 0.5) * 200  # -100 to +100
        
        # Аналіз агресії (відстань від mid price)
        aggressive_buys = sum(t['size'] for t in recent_trades 
                            if t['side'].lower() == 'buy' and t['price'] > current_price)
        aggressive_sells = sum(t['size'] for t in recent_trades 
                             if t['side'].lower() == 'sell' and t['price'] < current_price)
        
        aggression_imbalance = aggressive_buys - aggressive_sells
        
        return {
            "trade_imbalance_score": trade_imbalance_score,
            "buy_pressure": buy_volume,
            "sell_pressure": sell_volume,
            "aggression_imbalance": aggression_imbalance
        }


class MultiTimeframeImbalanceAnalyzer:
    """Мульти-таймфрейм аналіз імбалансу"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.timeframes = ['1m', '5m', '30m']
        self.imbalance_cache: Dict[str, Dict[str, float]] = {}
        
    def compute_multi_tf_imbalance(self, symbol: str) -> Dict[str, float]:
        """Розрахунок імбалансу для всіх таймфреймів"""
        tf_imbalances = {}
        
        for tf in self.timeframes:
            bars = self.storage.multi_tf.get_bars(symbol, tf, limit=20)
            if not bars:
                tf_imbalances[f"{tf}_imbalance"] = 0.0
                continue
            
            # РОЗРАХОВУЄМО ІМБАЛАНС ДИНАМІЧНО ДЛЯ КОЖНОГО БАРУ
            imbalances = []
            for bar in bars[-10:]:  # останні 10 барів
                total_vol = bar.buy_volume + bar.sell_volume
                if total_vol > 0:
                    imb = ((bar.buy_volume - bar.sell_volume) / total_vol) * 100
                else:
                    imb = 0.0
                imbalances.append(imb)
            
            avg_imbalance = sum(imbalances) / len(imbalances) if imbalances else 0.0
            tf_imbalances[f"{tf}_imbalance"] = avg_imbalance
            
            # Кешування для адаптації
            self.imbalance_cache[f"{symbol}_{tf}"] = avg_imbalance
        
        return tf_imbalances
    
    def get_adaptive_imbalance_weights(self, symbol: str) -> Dict[str, float]:
        """Адаптивні ваги імбалансу залежно від таймфреймів"""
        weights = {}
        base_weight = 1.0 / len(self.timeframes)
        
        for tf in self.timeframes:
            cache_key = f"{symbol}_{tf}"
            imbalance = self.imbalance_cache.get(cache_key, 0.0)
            
            # Збільшити вагу таймфрейму з сильнішим імбалансом
            strength_multiplier = 1.0 + abs(imbalance) / 50.0  # max +1.0 при |imbalance|=50
            weights[tf] = base_weight * strength_multiplier
        
        # Нормалізація
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights


class ImbalanceAnalyzer:
    """Мульти-таймфрейм аналізатор імбалансу з адаптацією"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.cfg = settings.imbalance
        self.bayesian = BayesianUpdater()
        self.cluster_analyzer = ClusterAnalyzer()
        self.depth_analyzer = DepthAnalyzer()
        self.trade_analyzer = TradeImbalanceAnalyzer()
        self.multi_tf = MultiTimeframeImbalanceAnalyzer(storage)
        
        # Історичні дані для адаптивного аналізу
        self.historical_imbalances: Dict[str, deque] = {}
        self.volatility_cache: Dict[str, float] = {}
        
    def update_volatility_cache(self, symbol: str, volume_data: Dict):
        """Оновлення кешу волатильності для адаптації"""
        self.volatility_cache[symbol] = volume_data.get("volatility", 0.1)
    
    def compute(self, symbol: str) -> Dict[str, Any]:
        """Головний розрахунок імбалансу з мульти-таймфрейм підтримкою"""
        order_book = self.storage.get_order_book(symbol)
        if not order_book:
            return self._get_empty_imbalance_data(symbol)
        
        current_price = (order_book.best_bid + order_book.best_ask) / 2
        
        # Базовий аналіз orderbook
        bids = [(level.price, level.size) for level in order_book.bids[:self.cfg.depth_limit_for_calc]]
        asks = [(level.price, level.size) for level in order_book.asks[:self.cfg.depth_limit_for_calc]]
        
        # Розрахунок базових метрик
        effective_imbalance = self._calculate_effective_imbalance(bids, asks)
        depth_analysis = self.depth_analyzer.analyze_depth(bids, asks, current_price)
        cluster_analysis = self.cluster_analyzer.analyze_clusters(bids, asks, current_price)
        
        # Аналіз трейдів
        trades = self.storage.get_trades(symbol)
        trade_imbalance = self.trade_analyzer.update_trade_imbalance(
            symbol, trades[-1].price if trades else current_price, 
            trades[-1].size if trades else 0, 
            trades[-1].side if trades else 'unknown', 
            current_price
        )
        
        # Баєсівський аналіз
        bayesian_data = self._calculate_bayesian_data(symbol, effective_imbalance)
        
        # Мульти-таймфрейм імбаланс
        multi_tf_imbalances = self.multi_tf.compute_multi_tf_imbalance(symbol)
        
        # Адаптивні ваги
        adaptive_weights = self._calculate_adaptive_weights(symbol, effective_imbalance, depth_analysis)
        
        # Об'єднання результатів
        result = {
            "effective_imbalance": effective_imbalance,
            "imbalance_score": effective_imbalance,
            "depth_analysis": depth_analysis,
            "cluster_analysis": cluster_analysis,
            "bayesian_data": bayesian_data,
            "trade_imbalance": trade_imbalance,
            "multi_tf_imbalances": multi_tf_imbalances,
            "adaptive_weights": adaptive_weights,
            "timestamp": time.time()
        }
        
        # Збереження в історію для адаптації
        self._update_historical_data(symbol, effective_imbalance)
        
        return result
    
    def _calculate_effective_imbalance(self, bids: List[Tuple[float, float]], 
                                     asks: List[Tuple[float, float]]) -> float:
        """Розрахунок ефективного імбалансу з адаптивними порогами"""
        if not bids or not asks:
            return 0.0
        
        # Розрахунок сирих обсягів
        bid_volumes = [size for _, size in bids]
        ask_volumes = [size for _, size in asks]
        
        # Адаптивні пороги великих ордерів
        if self.cfg.enable_adaptive_large_orders:
            large_bid_threshold = self._calculate_adaptive_threshold(bid_volumes)
            large_ask_threshold = self._calculate_adaptive_threshold(ask_volumes)
        else:
            total_bid = sum(bid_volumes)
            total_ask = sum(ask_volumes)
            large_bid_threshold = max(self.cfg.large_order_side_percent * total_bid, self.cfg.large_order_min_notional_abs)
            large_ask_threshold = max(self.cfg.large_order_side_percent * total_ask, self.cfg.large_order_min_notional_abs)
        
        # Розрахунок імбалансу з великих ордерів
        large_bids = sum(size for size in bid_volumes if size >= large_bid_threshold)
        large_asks = sum(size for size in ask_volumes if size >= large_ask_threshold)
        
        total_large = large_bids + large_asks
        if total_large == 0:
            return 0.0
        
        imbalance = ((large_bids - large_asks) / total_large) * 100
        
        # Капування імбалансу
        imbalance = max(-self.cfg.universal_imbalance_cap, min(self.cfg.universal_imbalance_cap, imbalance))
        
        return imbalance
    
    def _calculate_adaptive_threshold(self, volumes: List[float]) -> float:
        """Адаптивний розрахунок порогу великих ордерів (Z-Score)"""
        if len(volumes) < self.cfg.large_order_min_samples:
            return self.cfg.large_order_min_notional_abs
        
        mean_vol = sum(volumes) / len(volumes)
        std_vol = math.sqrt(sum((v - mean_vol) ** 2 for v in volumes) / len(volumes))
        
        if std_vol == 0:
            return mean_vol
        
        threshold = mean_vol + (std_vol * self.cfg.large_order_zscore_threshold)
        return max(threshold, self.cfg.large_order_min_notional_abs)
    
    def _calculate_bayesian_data(self, symbol: str, current_imbalance: float) -> Dict[str, Any]:
        """Баєсівський аналіз імбалансу"""
        signal = "NEUTRAL"
        confidence = 0.5
        
        if current_imbalance > 10:
            signal, confidence = self.bayesian.update_beliefs("BUY", abs(current_imbalance) / 100)
        elif current_imbalance < -10:
            signal, confidence = self.bayesian.update_beliefs("SELL", abs(current_imbalance) / 100)
        else:
            signal, confidence = "NEUTRAL", 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "prior_bullish": self.bayesian.prior_bullish,
            "prior_bearish": self.bayesian.prior_bearish
        }
    
    def _calculate_adaptive_weights(self, symbol: str, imbalance: float, 
                                  depth_analysis: Dict) -> Dict[str, Any]:
        """Розрахунок адаптивних ваг залежно від ринкових умов"""
        volatility = self.volatility_cache.get(symbol, 1.0)
        
        # Визначення режиму ринку
        if volatility > settings.adaptive.tf_adaptation_volatility_threshold:
            market_mode = "high_volatility"
        elif volatility < 0.5:
            market_mode = "low_volatility"
        elif abs(imbalance) > 30:
            market_mode = "strong_trend"
        else:
            market_mode = "sideways"
        
        # Отримання множників з налаштувань
        multipliers = settings.adaptive.adaptive_weight_multipliers.get(market_mode, {})
        
        # Мульти-таймфрейм ваги імбалансу
        tf_weights = self.multi_tf.get_adaptive_imbalance_weights(symbol)
        
        return {
            "market_mode": market_mode,
            "volatility": volatility,
            "imbalance_strength": abs(imbalance),
            "weight_multipliers": multipliers,
            "tf_weights": tf_weights
        }
    
    def _update_historical_data(self, symbol: str, imbalance: float):
        """Оновлення історичних даних для адаптації"""
        if symbol not in self.historical_imbalances:
            self.historical_imbalances[symbol] = deque(maxlen=1000)
        
        self.historical_imbalances[symbol].append({
            'imbalance': imbalance,
            'timestamp': time.time()
        })
    
    def _get_empty_imbalance_data(self, symbol: str) -> Dict[str, Any]:
        """Повернення порожніх даних імбалансу"""
        return {
            "effective_imbalance": 0.0,
            "imbalance_score": 0.0,
            "depth_analysis": {
                "support_resistance_ratio": 0.5,
                "liquidity_imbalance": 0.0,
                "bid_ask_wall_ratio": 1.0,
                "depth_distribution": {"bids": 0, "asks": 0}
            },
            "cluster_analysis": {
                "poc_price": 0.0,
                "poc_distance_pct": 0.0,
                "support_cluster_strength": 0.0,
                "resistance_cluster_strength": 0.0,
                "cluster_imbalance": 0.0
            },
            "bayesian_data": {
                "signal": "NEUTRAL",
                "confidence": 0.5,
                "prior_bullish": 0.5,
                "prior_bearish": 0.5
            },
            "trade_imbalance": {
                "trade_imbalance_score": 0.0,
                "buy_pressure": 0.0,
                "sell_pressure": 0.0,
                "aggression_imbalance": 0.0
            },
            "multi_tf_imbalances": {f"{tf}_imbalance": 0.0 for tf in ['1m', '5m', '30m']},
            "adaptive_weights": {
                "market_mode": "unknown",
                "volatility": 0.0,
                "imbalance_strength": 0.0,
                "weight_multipliers": {},
                "tf_weights": {}
            },
            "timestamp": time.time()
        }