# trading/orchestrator.py
import asyncio
import time
import json
from typing import Optional, Dict, Any  # Додано Any
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage, Position
from analysis.imbalance import ImbalanceAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.signals import SignalGenerator
from trading.executor import TradeExecutor

class TradingOrchestrator:
    """Оновлений Orchestrator з мульти-таймфрейм підтримкою та адаптацією"""

    def __init__(self, storage: DataStorage, imbalance_analyzer: ImbalanceAnalyzer,
                 volume_analyzer: VolumeAnalyzer, signal_generator: SignalGenerator, executor: TradeExecutor):
        self.storage = storage
        self.imb = imbalance_analyzer
        self.vol = volume_analyzer
        self.sig_gen = signal_generator
        self.executor = executor
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_open_ts: Dict[str, float] = {}
        self._last_close_ts: Dict[str, float] = {}
        self._last_signal: Dict[str, Dict] = {}
        self._last_trade_time: Dict[str, float] = {}
        self._reverse_pending: Dict[str, bool] = {}
        
        self._position_status_cache: Dict[str, Dict] = {}
        self._cache_ttl = 3.0
        
        # Мульти-таймфрейм адаптація
        self._market_condition_cache: Dict[str, Dict] = {}
        self._adaptation_cycle = 0

    async def start(self):
        """Запуск оркестратора з мульти-таймфрейм моніторингом"""
        if self._running:
            return
            
        self._running = True
        logger.info("🎼 [ORCHESTRATOR] Starting Multi-Timeframe Trading Orchestrator...")
        
        self._task = asyncio.create_task(self._main_loop())
        logger.info("✅ [ORCHESTRATOR] Multi-Timeframe Orchestrator started successfully")

    async def stop(self):
        """Зупинка оркестратора"""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 [ORCHESTRATOR] Multi-Timeframe Orchestrator stopped")

    async def _main_loop(self):
        """Головна петля з адаптивним батчингом"""
        batch_size = 5  # Початковий розмір батчу
        batch_interval = 2.0  # Інтервал між батчами
        
        while self._running:
            try:
                await self._adaptive_batch_processing(batch_size)
                await asyncio.sleep(batch_interval)
                
                # Адаптація розміру батчу на основі продуктивності
                batch_size = self._adapt_batch_size(batch_size)
                self._adaptation_cycle += 1
                
            except Exception as e:
                logger.error(f"❌ [ORCH] Main loop error: {e}")
                await asyncio.sleep(5)

    async def _adaptive_batch_processing(self, batch_size: int):
        """Адаптивна обробка символів батчами"""
        symbols = settings.pairs.trade_pairs
        
        # Розділення на батчі
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Паралельна обробка батчу
            tasks = [self._process_single_symbol(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Коротка пауза між батчами для зниження навантаження
            await asyncio.sleep(0.1)

    def _adapt_batch_size(self, current_batch_size: int) -> int:
        """Адаптація розміру батчу на основі продуктивності"""
        # Збільшення батчу кожні 10 циклів якщо немає помилок
        if self._adaptation_cycle % 10 == 0:
            if current_batch_size < len(settings.pairs.trade_pairs):
                new_size = min(current_batch_size + 1, 10)
                logger.debug(f"[ORCH] Adapting batch size: {current_batch_size} -> {new_size}")
                return new_size
        
        return current_batch_size

    async def _fast_check_exchange_position_status(self, symbol: str) -> bool:
        """Швидка перевірка статусу позиції з кешуванням"""
        now = time.time()
        
        # Перевірка кешу
        if symbol in self._position_status_cache:
            cached = self._position_status_cache[symbol]
            if now - cached['timestamp'] < self._cache_ttl:
                return cached['can_process']
        
        try:
            # Синхронізація з біржею
            await self.storage.force_sync_positions(self.executor.api)
            
            position = self.storage.get_position(symbol)
            can_process = not (position and position.status == "OPEN")
            
            # Кешування результату
            self._position_status_cache[symbol] = {
                'can_process': can_process,
                'timestamp': now
            }
            
            return can_process
            
        except Exception as e:
            logger.warning(f"⚠️ [ORCH] Position check failed for {symbol}: {e}")
            return False

    async def _process_single_symbol(self, symbol: str):
        """Обробка одного символу з мульти-таймфрейм аналізом"""
        try:
            can_process = await self._fast_check_exchange_position_status(symbol)
            if not can_process:
                return

            ob = self.storage.get_order_book(symbol)
            if not ob:
                return

            # Обчислення сигналів з мульти-таймфрейм даними
            vol_data = self.vol.compute(symbol)
            imb_data = self.imb.compute(symbol)
            
            # Оновлюємо кеш волатильності для імбалансу
            self.imb.update_volatility_cache(symbol, vol_data)
            
            # 🆕 O'HARA METHOD 7: Spread calculation
            spread_bps = self.storage.get_current_spread_bps(symbol)
            if spread_bps is None and ob and ob.best_bid and ob.best_ask and ob.best_bid > 0 and ob.best_ask > 0:
                spread_bps = (ob.best_ask - ob.best_bid) / ob.best_bid * 10000
                if spread_bps < 0 or spread_bps > 1000:
                    spread_bps = None
            
            # Update spread monitor
            if spread_bps is not None and ob and ob.best_bid and ob.best_ask:
                self.sig_gen.spread_monitor.update(symbol, ob.best_bid, ob.best_ask)

            # Генерація сигналу з мульти-таймфрейм даними
            sig = self.sig_gen.generate(symbol, imb_data, vol_data, spread_bps)
            self._last_signal[symbol] = sig
            
            # Логування мульти-таймфрейм умов ринку
            if self._adaptation_cycle % 20 == 0:  # Кожні 20 циклів
                self._log_market_conditions(symbol, vol_data, imb_data)
            
            # Паралельна обробка
            await asyncio.gather(
                self._optimized_maybe_close(symbol, sig, ob, vol_data),
                self._optimized_maybe_open(symbol, sig, ob, vol_data),
                return_exceptions=True
            )
                
        except Exception as e:
            logger.error(f"❌ [ORCH] Error processing {symbol}: {e}")

    def _log_market_conditions(self, symbol: str, vol_data: Dict, imb_data: Dict):
        """Логування поточних ринкових умов для адаптації"""
        multi_tf = vol_data.get("multi_timeframe_data", {})
        market_mode = imb_data.get("adaptive_weights", {}).get("market_mode", "unknown")
        
        logger.info(f"📊 [MARKET_CONDITIONS] {symbol}: mode={market_mode}, "
                   f"vol_1m={multi_tf.get('1m').volatility if multi_tf.get('1m') else 0:.2f}%, "
                   f"trend_5m={multi_tf.get('5m').trend if multi_tf.get('5m') else 'N/A'}, "
                   f"imb_30m={multi_tf.get('30m').imbalance if multi_tf.get('30m') else 0:.1f}")

    async def _optimized_maybe_open(self, symbol: str, sig: Dict, ob, vol_data: Dict):
        """Оновлена логіка відкриття з мульти-таймфрейм фільтрами"""
        can_open = await self._fast_check_exchange_position_status(symbol)
        if not can_open:
            return

        if not self._quick_open_checks(symbol, sig):
            return

        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)
        
        if action == "HOLD" or strength < self.executor.tcfg.entry_signal_min_strength:
            return
        
        # 🆕 МУЛЬТИ-ТАЙМФРЕЙМ ФІЛЬТРИ
        multi_tf_data = vol_data.get("multi_timeframe_data", {})
        
        # Перевірка консистентності тренду на різних таймфреймах
        tf_1m = multi_tf_data.get('1m')
        tf_5m = multi_tf_data.get('5m')
        trend_1m = tf_1m.trend if tf_1m else 'SIDEWAYS'
        trend_5m = tf_5m.trend if tf_5m else 'SIDEWAYS'
        
        if action == "BUY" and (trend_1m == "DOWN" or trend_5m == "DOWN"):
            logger.debug(f"[MTF_FILTER] {symbol}: BUY rejected - conflicting trends 1m:{trend_1m}, 5m:{trend_5m}")
            return
        elif action == "SELL" and (trend_1m == "UP" or trend_5m == "UP"):
            logger.debug(f"[MTF_FILTER] {symbol}: SELL rejected - conflicting trends 1m:{trend_1m}, 5m:{trend_5m}")
            return
        
        # Перевірка імбалансу на вищих таймфреймах
        tf_30m = multi_tf_data.get('30m')
        imb_30m = tf_30m.imbalance if tf_30m else 0
        if abs(imb_30m) < 10:  # Занадто слабкий імбаланс на 30m
            logger.debug(f"[MTF_FILTER] {symbol}: Weak 30m imbalance ({imb_30m:.1f}) - reducing position size")
            # Зменшити розмір позиції замість відмови
        
        # 🆕 O'HARA FILTER: Check spread risk
        factors = sig.get("factors", {})
        if factors:
            spread_factor = factors.get("spread", 0)
            if spread_factor < -0.4:  # Дуже негативний spread factor
                logger.warning(f"[OHARA_FILTER] {symbol}: Spread too wide, avoiding trade")
                return

        is_reverse, double_size = await self._fast_determine_reverse(symbol, action)
        
        if symbol in self.executor.active_orders and not is_reverse:
            return

        best_bid = getattr(ob, "best_bid", None)
        best_ask = getattr(ob, "best_ask", None)
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
        else:
            mid = best_bid or best_ask
        
        if mid is None:
            return

        effective_action = action
        if self.executor.tcfg.reverse_signals:
            effective_action = "SELL" if action == "BUY" else ("BUY" if action == "SELL" else action)

        signal_info = await self._fast_create_signal_info(symbol, action, strength, sig, is_reverse)

        logger.info(f"[ORCH] 🎯 Opening {symbol}: {effective_action} with signal {signal_info}")
        
        await self.executor.open_position_limit(
            symbol=symbol,
            direction=effective_action,
            ref_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            is_reversed=is_reverse,
            double_size=double_size,
            signal_info=signal_info,
            volatility_data=vol_data
        )
        
        self._last_open_ts[symbol] = time.time()
        if action != "HOLD":
            self._last_trade_time[symbol] = time.time()

    def _quick_open_checks(self, symbol: str, sig: Dict) -> bool:
        """ШВИДКІ перевірки з мульти-таймфрейм фільтрами"""
        current_time = time.time()
        
        last_trade_time = self._last_trade_time.get(symbol, 0)
        if current_time - last_trade_time < self.executor.tcfg.min_time_between_trades_sec:
            return False

        last_close = self._last_close_ts.get(symbol, 0)
        if current_time - last_close < self.executor.tcfg.reopen_cooldown_sec:
            return False

        # 🆕 МУЛЬТИ-ТАЙМФРЕЙМ ФІЛЬТР: Перевірка волатильності на різних таймфреймах
        factors = sig.get("factors", {})
        if factors:
            vol_1m = factors.get("multi_tf_volatility_1m", 0)
            vol_5m = factors.get("multi_tf_volatility_5m", 0)
            vol_30m = factors.get("multi_tf_volatility_30m", 0)
            
            # Якщо волатильність занадто висока на всіх таймфреймах - уникати
            if vol_1m > 5 and vol_5m > 4 and vol_30m > 3:
                logger.debug(f"[MTF_VOL_FILTER] {symbol}: Extreme volatility across timeframes")
                return False

        # 🆕 O'HARA FILTER: Check O'Hara score
        ohara_score = sig.get("ohara_score", 0)
        if settings.ohara.enable_combined_ohara_score and ohara_score < settings.ohara.min_ohara_score_for_trade:
            logger.debug(f"[OHARA_FILTER] {symbol}: O'Hara score too low ({ohara_score}/{settings.ohara.min_ohara_score_for_trade})")
            return False

        if self.executor.tcfg.enable_aggressive_filtering:
            raw_values = sig.get('factors', {}).get('raw_values', {})
            momentum_score = raw_values.get('momentum_score', 0)
            if abs(momentum_score) > 90 and sig.get('strength', 0) >= 4:
                return False

        return True

    async def _fast_determine_reverse(self, symbol: str, action: str) -> tuple:
        """Швидке визначення реверсу"""
        current_pos = self.storage.get_position(symbol)
        is_reverse = False
        double_size = False
        
        if current_pos and current_pos.status == "OPEN":
            if (current_pos.side == "LONG" and action == "SELL") or \
               (current_pos.side == "SHORT" and action == "BUY"):
                is_reverse = True
                double_size = self.executor.tcfg.reverse_double_size
                logger.info(f"[REVERSE] 🔄 {symbol}: closing {current_pos.side} and opening {action}")
                self._reverse_pending[symbol] = True

        return is_reverse, double_size

    async def _fast_create_signal_info(self, symbol: str, action: str, 
                                     strength: int, sig: Dict, is_reverse: bool) -> str:
        """Швидке створення інформації про сигнал з мульти-таймфрейм деталями"""
        try:
            signal_parts = []
            if is_reverse:
                signal_parts.append("REVERSE")
            display_action = "SELL" if action == "BUY" else "BUY" if action == "SELL" else action
            if self.executor.tcfg.reverse_signals:
                signal_parts.append(f"{display_action.upper()}{strength}")
            else:
                signal_parts.append(f"{action.upper()}{strength}")

            factors = sig.get('factors', {})
            raw_values = factors.get('raw_values', {})
            if raw_values:
                imb_score = raw_values.get('imbalance_score', 0)
                mom_score = raw_values.get('momentum_score', 0)
                ohara_score = sig.get('ohara_score', 0)
                
                # Додавання мульти-таймфрейм інформації
                multi_tf_data = factors.get('multi_timeframe_data', {})
                trend_5m = multi_tf_data.get('5m', {}).get('trend', 'N/A')
                vol_30m = multi_tf_data.get('30m', {}).get('volatility', 0)
                
                signal_parts.append(f"(imb:{imb_score:.0f},mom:{mom_score:.0f},oh:{ohara_score},trend:{trend_5m},vol:{vol_30m:.1f})")

            return " ".join(signal_parts)

        except Exception as e:
            logger.error(f"❌ [FAST_SIGNAL] {symbol}: {e}")
            return f"{action.upper()}{strength}" + (" (reverse)" if is_reverse else "")

    async def _optimized_maybe_close(self, symbol: str, sig: Dict, ob, vol_data: Dict):
        """Оптимізована логіка закриття з мульти-таймфрейм умовами"""
        position = self.storage.get_position(symbol)
        if not position or position.status != "OPEN":
            return

        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)

        # Базові перевірки
        if action == "HOLD":
            return

        # Перевірка на реверс
        is_reverse_signal = ((position.side == "LONG" and action == "SELL") or 
                           (position.side == "SHORT" and action == "BUY"))

        if not is_reverse_signal:
            # Закриття за умовчанням
            close_reason = self._determine_close_reason(position, sig, vol_data)
            if close_reason:
                await self._execute_close(symbol, close_reason, sig)
        else:
            # Реверс буде оброблено в _maybe_open
            pass

    def _determine_close_reason(self, position: Position, sig: Dict, vol_data: Dict) -> Optional[str]:
        """Визначення причини закриття з мульти-таймфрейм аналізом"""
        factors = sig.get('factors', {})
        raw_values = factors.get('raw_values', {})
        
        # Перевірка часу життя позиції
        current_time = time.time()
        position_age = current_time - position.timestamp
        
        if position_age > position.max_lifetime_sec:
            return "MAX_LIFETIME"
        
        # Мульти-таймфрейм перевірка тренду
        multi_tf_data = vol_data.get("multi_timeframe_data", {})
        tf_5m = multi_tf_data.get('5m')
        tf_30m = multi_tf_data.get('30m')
        trend_5m = tf_5m.trend if tf_5m else 'SIDEWAYS'
        trend_30m = tf_30m.trend if tf_30m else 'SIDEWAYS'
        
        # Закриття LONG якщо тренд змінився на DOWN
        if position.side == "LONG" and (trend_5m == "DOWN" or trend_30m == "DOWN"):
            if sig.get('strength', 0) >= 2:  # Досить сильний сигнал проти
                return "MTF_TREND_CHANGE_DOWN"
        
        # Закриття SHORT якщо тренд змінився на UP
        if position.side == "SHORT" and (trend_5m == "UP" or trend_30m == "UP"):
            if sig.get('strength', 0) >= 2:
                return "MTF_TREND_CHANGE_UP"
        
        # Перевірка імбалансу на вищих таймфреймах
        imb_30m = tf_30m.imbalance if tf_30m else 0
        if position.side == "LONG" and imb_30m < -30:  # Сильний імбаланс проти
            return "MTF_IMBALANCE_AGAINST"
        if position.side == "SHORT" and imb_30m > 30:
            return "MTF_IMBALANCE_AGAINST"
        
        # Перевірка волатильності
        vol_1m = tf_1m.volatility if (tf_1m := multi_tf_data.get('1m')) else 0
        vol_5m = tf_5m.volatility if tf_5m else 0
        
        if vol_1m > 8 or vol_5m > 6:  # Екстремальна волатильність
            return "MTF_EXTREME_VOLATILITY"
        
        # Перевірка стоп-лосс/тейк-профіт
        if hasattr(position, 'stop_loss') and position.stop_loss:
            current_price = (sig.get('best_bid', 0) + sig.get('best_ask', 0)) / 2
            if position.side == "LONG" and current_price <= position.stop_loss:
                return "STOP_LOSS"
            if position.side == "SHORT" and current_price >= position.stop_loss:
                return "STOP_LOSS"
        
        if hasattr(position, 'take_profit') and position.take_profit:
            if position.side == "LONG" and current_price >= position.take_profit:
                return "TAKE_PROFIT"
            if position.side == "SHORT" and current_price <= position.take_profit:
                return "TAKE_PROFIT"
        
        return None

    async def _execute_close(self, symbol: str, reason: str, sig: Dict):
        """Виконання закриття позиції"""
        try:
            logger.info(f"[CLOSE] 🔒 {symbol}: {reason}")
            
            # Отримання поточних цін
            ob = self.storage.get_order_book(symbol)
            if ob:
                best_bid = ob.best_bid
                best_ask = ob.best_ask
                mid_price = (best_bid + best_ask) / 2
            else:
                mid_price = 0
            
            # Закриття через executor
            await self.executor.close_position_market(
                symbol=symbol,
                close_reason=reason,
                current_price=mid_price
            )
            
            self._last_close_ts[symbol] = time.time()
            
        except Exception as e:
            logger.error(f"❌ [CLOSE_ERROR] {symbol}: {e}")

    def get_market_condition_report(self, symbol: str) -> Dict[str, Any]:
        """Отримання звіту про ринкові умови для символу"""
        vol_data = self.vol.compute(symbol)
        imb_data = self.imb.compute(symbol)
        
        multi_tf = vol_data.get("multi_timeframe_data", {})
        adaptive_weights = imb_data.get("adaptive_weights", {})
        
        return {
            "symbol": symbol,
            "market_mode": adaptive_weights.get("market_mode", "unknown"),
            "volatility_1m": multi_tf.get('1m').volatility if multi_tf.get('1m') else 0,
            "trend_5m": multi_tf.get('5m').trend if multi_tf.get('5m') else 'SIDEWAYS',
            "imbalance_30m": multi_tf.get('30m').imbalance if multi_tf.get('30m') else 0,
            "adaptation_weights": adaptive_weights.get("weight_multipliers", {}),
            "timestamp": time.time()
        }