# trading/executor.py
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from config.settings import settings
from utils.logger import logger
from data.storage import DataStorage, Position
from utils.notifications import notifier


class TradeExecutor:
    """Мульти-таймфрейм адаптивний виконавець трейдів"""

    def __init__(self, storage: DataStorage, api_manager):
        self.storage = storage
        self.api = api_manager
        self.tcfg = settings.risk
        self.pcfg = settings.pairs
        
        # Стан трейдів
        self.active_orders: Dict[str, Dict] = {}
        self.pending_positions: Dict[str, Dict] = {}
        self._stats = {
            "total_trades": 0,
            "opens": 0,
            "closes": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity_diff_vs_start": 0.0,
            "open_positions": 0,
            "positions_details": []
        }
        
        # Мульти-таймфрейм адаптація
        self._market_adaptation_cache: Dict[str, Dict] = {}
        self._position_size_multipliers: Dict[str, float] = {}
        
        # Історія трейдів для статистики
        self._trade_history: List[Dict] = []
        self._running = False
        self._task = None

    async def start(self):
        """Запуск виконавця"""
        if self._running:
            return
            
        self._running = True
        logger.info("⚡ [EXECUTOR] Starting Multi-Timeframe Adaptive Trade Executor...")
        
        # Синхронізуємо позиції при старті
        await self._sync_positions_on_startup()
        
        self._task = asyncio.create_task(self._monitor_positions())
        logger.info("✅ [EXECUTOR] Trade Executor started successfully")

    async def stop(self):
        """Зупинка виконавця"""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 [EXECUTOR] Trade Executor stopped")

    async def _sync_positions_on_startup(self):
        """Синхронізація позицій при запуску"""
        try:
            await self.storage.force_sync_positions(self.api)
            logger.info("✅ [EXECUTOR] Positions synchronized on startup")
        except Exception as e:
            logger.error(f"❌ [EXECUTOR] Failed to sync positions: {e}")

    async def _monitor_positions(self):
        """Моніторинг позицій з мульти-таймфрейм адаптацією"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Кожні 5 секунд
                
                # Оновлюємо кеш адаптації
                await self._update_market_adaptation_cache()
                
                # Моніторимо відкриті позиції
                open_positions = self.storage.get_open_positions()
                
                for symbol, position in open_positions.items():
                    await self._monitor_single_position(symbol, position)
                
                # Оновлюємо статистику
                self._update_stats()
                
            except Exception as e:
                logger.error(f"❌ [POSITION_MONITOR] Error: {e}")

    async def _update_market_adaptation_cache(self):
        """Оновлення кешу ринкових умов для адаптації"""
        for symbol in self.pcfg.trade_pairs:
            try:
                multi_tf_data = self.storage.get_multi_timeframe_data(symbol)
                vol_data = {}  # Можна додати більше даних
                
                # Визначаємо режим ринку
                vol_30m = multi_tf_data.get('30m', {}).get('volatility', 0)
                trend_5m = multi_tf_data.get('5m', {}).get('trend', 'SIDEWAYS')
                
                if vol_30m > settings.adaptive.tf_adaptation_volatility_threshold:
                    market_mode = "high_volatility"
                elif vol_30m < 0.5:
                    market_mode = "low_volatility"
                elif trend_5m in ['UP', 'DOWN']:
                    market_mode = "strong_trend"
                else:
                    market_mode = "sideways"
                
                # Розраховуємо множник розміру позиції
                size_multiplier = self._calculate_position_size_multiplier(symbol, market_mode, multi_tf_data)
                
                self._market_adaptation_cache[symbol] = {
                    "market_mode": market_mode,
                    "volatility_30m": vol_30m,
                    "trend_5m": trend_5m,
                    "size_multiplier": size_multiplier,
                    "last_update": time.time()
                }
                
            except Exception as e:
                logger.debug(f"[ADAPTATION_CACHE] Error for {symbol}: {e}")

    def _calculate_position_size_multiplier(self, symbol: str, market_mode: str, multi_tf_data: Dict) -> float:
        """Розрахунок множника розміру позиції на основі умов"""
        base_multiplier = 1.0
        
        # Зменшуємо розмір при високій волатильності
        vol_30m = multi_tf_data.get('30m', {}).get('volatility', 0)
        if vol_30m > 5:
            base_multiplier *= 0.7  # 70% від звичайного розміру
        elif vol_30m > 3:
            base_multiplier *= 0.85  # 85%
        
        # Зменшуємо при сильному тренді (щоб уникнути late entry)
        trend_5m = multi_tf_data.get('5m', {}).get('trend', 'SIDEWAYS')
        imbalance_30m = abs(multi_tf_data.get('30m', {}).get('imbalance', 0))
        
        if trend_5m in ['UP', 'DOWN'] and imbalance_30m > 30:
            base_multiplier *= 0.8  # 80% при сильному тренді
        
        # Збільшуємо при боковому русі (кращі умови для імбалансу)
        if market_mode == "sideways":
            base_multiplier *= 1.1  # 110%
        
        return max(0.5, min(1.5, base_multiplier))  # Обмежуємо від 50% до 150%

    async def _monitor_single_position(self, symbol: str, position: Position):
        """Моніторинг однієї позиції з адаптацією"""
        try:
            current_time = time.time()
            
            # Перевіряємо тайм-аут позиції
            if current_time - position.timestamp > position.max_lifetime_sec:
                logger.warning(f"⏰ [POSITION_TIMEOUT] {symbol}: Position timed out")
                await self.close_position_market(symbol, "TIMEOUT", 0)
                return
            
            # Оновлюємо P&L
            await self._update_position_pnl(symbol, position)
            
            # Моніторинг стоп-лосс та тейк-профіт (якщо встановлені)
            await self._check_stop_loss_take_profit(symbol, position)
            
            # Мульти-таймфрейм перевірка на закриття
            await self._check_multi_tf_close_conditions(symbol, position)
            
        except Exception as e:
            logger.error(f"❌ [POSITION_MONITOR] {symbol}: {e}")

    async def _update_position_pnl(self, symbol: str, position: Position):
        """Оновлення P&L позиції"""
        try:
            # Отримуємо поточну ціну
            ob = self.storage.get_order_book(symbol)
            if not ob:
                return
                
            current_price = (ob.best_bid + ob.best_ask) / 2
            
            # Розраховуємо P&L
            if position.side == "LONG":
                unrealized_pnl = (current_price - position.entry_price) * position.qty
            else:  # SHORT
                unrealized_pnl = (position.entry_price - current_price) * position.qty
            
            position.current_price = current_price
            position.unrealised_pnl = unrealized_pnl
            
            # Оновлюємо час останнього моніторингу
            position.last_update = time.time()
            position._position_updated = True
            
            # Викликаємо callbacks
            await self.storage._trigger_position_callbacks(position)
            
        except Exception as e:
            logger.debug(f"[PNL_UPDATE] Error for {symbol}: {e}")

    async def _check_stop_loss_take_profit(self, symbol: str, position: Position):
        """Перевірка стоп-лосс та тейк-профіт"""
        if not position.stop_loss and not position.take_profit:
            return
            
        ob = self.storage.get_order_book(symbol)
        if not ob:
            return
            
        current_price = (ob.best_bid + ob.best_ask) / 2
        
        close_reason = None
        
        if position.side == "LONG":
            if position.stop_loss and current_price <= position.stop_loss:
                close_reason = "STOP_LOSS"
            elif position.take_profit and current_price >= position.take_profit:
                close_reason = "TAKE_PROFIT"
        else:  # SHORT
            if position.stop_loss and current_price >= position.stop_loss:
                close_reason = "STOP_LOSS"
            elif position.take_profit and current_price <= position.take_profit:
                close_reason = "TAKE_PROFIT"
        
        if close_reason:
            logger.info(f"🎯 [{close_reason}] {symbol}: Triggered at {current_price:.6f}")
            await self.close_position_market(symbol, close_reason, current_price)

    async def _check_multi_tf_close_conditions(self, symbol: str, position: Position):
        """Мульти-таймфрейм перевірка умов закриття"""
        try:
            adaptation = self._market_adaptation_cache.get(symbol, {})
            multi_tf_data = self.storage.get_multi_timeframe_data(symbol)
            
            # Закриття при екстремальній волатильності
            vol_1m = multi_tf_data.get('1m', {}).get('volatility', 0)
            vol_5m = multi_tf_data.get('5m', {}).get('volatility', 0)
            
            if vol_1m > 8 or vol_5m > 6:
                logger.warning(f"🌪️ [EXTREME_VOL] {symbol}: Closing due to extreme volatility")
                await self.close_position_market(symbol, "EXTREME_VOLATILITY", 0)
                return
            
            # Закриття при зміні тренду на вищих таймфреймах
            trend_5m = multi_tf_data.get('5m', {}).get('trend', 'SIDEWAYS')
            trend_30m = multi_tf_data.get('30m', {}).get('trend', 'SIDEWAYS')
            
            if position.side == "LONG" and (trend_5m == "DOWN" or trend_30m == "DOWN"):
                imbalance_30m = multi_tf_data.get('30m', {}).get('imbalance', 0)
                if imbalance_30m < -20:  # Підтвердження негативного імбалансу
                    logger.info(f"📉 [TREND_CHANGE] {symbol}: LONG closed due to downtrend on higher TF")
                    await self.close_position_market(symbol, "MTF_TREND_CHANGE_DOWN", 0)
                    return
            
            if position.side == "SHORT" and (trend_5m == "UP" or trend_30m == "UP"):
                imbalance_30m = multi_tf_data.get('30m', {}).get('imbalance', 0)
                if imbalance_30m > 20:  # Підтвердження позитивного імбалансу
                    logger.info(f"📈 [TREND_CHANGE] {symbol}: SHORT closed due to uptrend on higher TF")
                    await self.close_position_market(symbol, "MTF_TREND_CHANGE_UP", 0)
                    return
            
        except Exception as e:
            logger.debug(f"[MTF_CLOSE_CHECK] Error for {symbol}: {e}")

    async def open_position_limit(self, symbol: str, direction: str, ref_price: float, 
                                best_bid: float, best_ask: float, is_reversed: bool = False,
                                double_size: bool = False, signal_info: str = "",
                                volatility_data: Dict = None):
        """Відкриття позиції з мульти-таймфрейм адаптацією розміру"""
        try:
            # Отримуємо адаптацію для символу
            adaptation = self._market_adaptation_cache.get(symbol, {})
            size_multiplier = adaptation.get("size_multiplier", 1.0)
            
            # Базовий розмір (можна налаштувати)
            base_qty = 0.001  # Базовий розмір, можна адаптувати
            
            # Застосовуємо множник
            adjusted_qty = base_qty * size_multiplier
            
            # Подвоюємо при реверсі якщо потрібно
            if double_size:
                adjusted_qty *= 2
            
            # Мінімальний розмір
            adjusted_qty = max(0.0001, adjusted_qty)
            
            logger.info(f"📊 [POSITION_SIZE] {symbol}: base={base_qty}, multiplier={size_multiplier:.2f}, "
                       f"adjusted={adjusted_qty:.4f}, reason={adaptation.get('market_mode', 'normal')}")
            
            # Визначаємо ціну відкриття
            if direction.upper() == "BUY":
                open_price = best_ask
                side = "LONG"
            else:
                open_price = best_bid
                side = "SHORT"
            
            # Створюємо позицію в сховищі
            position = Position(
                symbol=symbol,
                side=side,
                qty=adjusted_qty,
                entry_price=open_price,
                status="OPEN",
                meta_open=signal_info
            )
            
            self.storage.positions[symbol] = position
            self._stats["opens"] += 1
            self._stats["total_trades"] += 1
            
            # Логуємо трейд
            self._log_trade(symbol, "OPEN", side, adjusted_qty, open_price, signal_info)
            
            logger.info(f"✅ [OPEN] {symbol}: {side} {adjusted_qty:.4f} @ {open_price:.6f} ({signal_info})")
            
            # Викликаємо callbacks
            await self.storage._trigger_position_callbacks(position)
            
            # Надсилаємо повідомлення
            try:
                await notifier.send(f"🆕 OPEN {symbol}: {side} {adjusted_qty:.4f} @ {open_price:.6f}")
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"❌ [OPEN_ERROR] {symbol}: {e}")

    async def close_position_market(self, symbol: str, close_reason: str, current_price: float = 0):
        """Закриття позиції по ринку"""
        try:
            position = self.storage.get_position(symbol)
            if not position or position.status != "OPEN":
                return
            
            # Отримуємо поточну ціну якщо не передана
            if current_price == 0:
                ob = self.storage.get_order_book(symbol)
                if ob:
                    current_price = (ob.best_bid + ob.best_ask) / 2
                else:
                    current_price = position.current_price
            
            # Розраховуємо реалізований P&L
            if position.side == "LONG":
                realized_pnl = (current_price - position.entry_price) * position.qty
            else:
                realized_pnl = (position.entry_price - current_price) * position.qty
            
            # Оновлюємо позицію
            position.status = "CLOSED"
            position.close_reason = close_reason
            position.exit_price = current_price
            position.realised_pnl = realized_pnl
            position.closed_timestamp = time.time()
            position._position_updated = True
            
            # Оновлюємо статистику
            self._stats["closes"] += 1
            self._stats["realized_pnl"] += realized_pnl
            
            # Логуємо трейд
            self._log_trade(symbol, "CLOSE", position.side, position.qty, current_price, close_reason)
            
            logger.info(f"🔒 [CLOSE] {symbol}: {position.side} {position.qty:.4f} @ {current_price:.6f}, "
                       f"PnL: {realized_pnl:.2f}, Reason: {close_reason}")
            
            # Переміщуємо в історію
            self.storage._closed_positions_history[symbol] = position
            del self.storage.positions[symbol]
            
            # Викликаємо callbacks
            await self.storage._trigger_position_callbacks(position)
            
            # Надсилаємо повідомлення
            try:
                pnl_emoji = "🟢" if realized_pnl > 0 else "🔴" if realized_pnl < 0 else "🟡"
                await notifier.send(f"{pnl_emoji} CLOSE {symbol}: PnL {realized_pnl:.2f}, Reason: {close_reason}")
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"❌ [CLOSE_ERROR] {symbol}: {e}")

    def _log_trade(self, symbol: str, action: str, side: str, qty: float, price: float, info: str):
        """Логування трейду в історію"""
        trade = {
            "timestamp": time.time(),
            "symbol": symbol,
            "action": action,
            "side": side,
            "qty": qty,
            "price": price,
            "info": info
        }
        
        self._trade_history.append(trade)
        
        # Обмежуємо історію останніми 1000 трейдами
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-1000:]

    def _update_stats(self):
        """Оновлення статистики"""
        try:
            open_positions = self.storage.get_open_positions()
            self._stats["open_positions"] = len(open_positions)
            
            # Деталі позицій
            positions_details = []
            total_unrealized = 0.0
            
            for symbol, pos in open_positions.items():
                positions_details.append({
                    "symbol": symbol,
                    "side": pos.side,
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "upnl": pos.unrealised_pnl
                })
                total_unrealized += pos.unrealised_pnl
            
            self._stats["unrealized_pnl"] = total_unrealized
            self._stats["positions_details"] = positions_details
            
            # Розрахунок win rate (якщо є закриті позиції)
            closed_positions = self.storage.get_closed_positions_history()
            if closed_positions:
                winning_trades = sum(1 for pos in closed_positions.values() 
                                    if pos.realised_pnl > 0)
                total_closed = len(closed_positions)
                self._stats["win_rate"] = (winning_trades / total_closed) * 100 if total_closed > 0 else 0
            
        except Exception as e:
            logger.debug(f"[STATS_UPDATE] Error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Отримання статистики"""
        self._update_stats()  # Оновлюємо перед поверненням
        return self._stats.copy()

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Отримання історії трейдів"""
        return self._trade_history[-limit:]

    def get_market_adaptation_info(self, symbol: str) -> Dict[str, Any]:
        """Отримання інформації про адаптацію для символу"""
        return self._market_adaptation_cache.get(symbol, {})