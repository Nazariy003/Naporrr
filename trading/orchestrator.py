# trading/orchestrator.py
import asyncio
import time
import json
from typing import Optional, Dict, Any
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage, Position
from analysis.imbalance import ImbalanceAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.signals import SignalGenerator
from trading.executor import TradeExecutor

class TradingOrchestrator:
    """Оновлений Orchestrator з мультитаймфреймовою підтримкою"""
    
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
        
        # 🆕 MTF трекінг
        self._mtf_signal_history = {}
        self._signal_convergence_cache = {}

    async def start(self):
        if self._task:
            return
        self._running = True
        self._task = asyncio.create_task(self._optimized_loop())
        logger.info("✅ [ORCH] Trading orchestrator started with MTF support")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("✅ [ORCH] Trading orchestrator cancelled")
            except Exception as e:
                logger.error(f"❌ [ORCH] error during stop: {e}")
            self._task = None
        logger.info("✅ [ORCH] Trading orchestrator stopped")

    async def _optimized_loop(self):
        """Оптимізований головний цикл"""
        interval = settings.trading.decision_interval_sec
        symbol_batches = self._create_symbol_batches(settings.pairs.trade_pairs, batch_size=3)
        batch_index = 0
        
        while self._running:
            start_iter = time.time()
            try:
                current_batch = symbol_batches[batch_index]
                await self._process_symbol_batch(current_batch)
                batch_index = (batch_index + 1) % len(symbol_batches)
            except Exception as e:
                logger.error(f"❌ [ORCH] iteration error: {e}", exc_info=True)
            
            elapsed = time.time() - start_iter
            await asyncio.sleep(max(0.0, interval - elapsed))

    def _create_symbol_batches(self, symbols: list, batch_size: int = 3) -> list:
        """Розділення символів на батчі"""
        return [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    async def _process_symbol_batch(self, symbols: list):
        """Обробка батчу символів"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_single_symbol(symbol))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_single_symbol(self, symbol: str):
        """Обробка одного символу з MTF аналізом"""
        try:
            can_process = await self._fast_check_exchange_position_status(symbol)
            if not can_process:
                return

            ob = self.storage.get_order_book(symbol)
            if not ob:
                return

            # Обчислення сигналів з MTF
            vol_data = self.vol.compute(symbol)
            imb_data = self.imb.compute(symbol)
            
            # 🆕 Аналіз MTF конвергенції
            mtf_convergence = await self._analyze_mtf_convergence(symbol, imb_data, vol_data)
            
            # Оновлюємо кеш волатильності
            self.imb.update_volatility_cache(symbol, vol_data)
            
            # Spread calculation
            spread_bps = self.storage.get_current_spread_bps(symbol)
            if spread_bps is None and ob and ob.best_bid and ob.best_ask and ob.best_bid > 0 and ob.best_ask > 0:
                spread_bps = (ob.best_ask - ob.best_bid) / ob.best_bid * 10000
                if spread_bps < 0 or spread_bps > 1000:
                    spread_bps = None
            
            # Update spread monitor
            if spread_bps is not None and ob and ob.best_bid and ob.best_ask:
                self.sig_gen.spread_monitor.update(symbol, ob.best_bid, ob.best_ask)

            # Генерація сигналу з MTF даними
            sig = self.sig_gen.generate(symbol, imb_data, vol_data, spread_bps)
            self._last_signal[symbol] = sig
            
            # 🆕 Оцінка MTF конвергенції
            mtf_quality = self._evaluate_mtf_signal_quality(symbol, sig, mtf_convergence)
            
            # Логування MTF інформації
            if mtf_quality['score'] > 0.7:
                logger.info(f"[MTF_HIGH_QUALITY] {symbol}: {mtf_quality['score']:.2f}, "
                          f"confirmed={mtf_quality['confirmed']}, "
                          f"timeframes={mtf_quality['confirmed_timeframes']}")
            
            # Паралельна обробка з MTG фільтрами
            await asyncio.gather(
                self._optimized_maybe_close(symbol, sig, ob, vol_data, mtf_quality),
                self._optimized_maybe_open(symbol, sig, ob, vol_data, mtf_quality),
                return_exceptions=True
            )
                
        except Exception as e:
            logger.error(f"❌ [ORCH] Error processing {symbol}: {e}")

    async def _analyze_mtf_convergence(self, symbol: str, imb_data: Dict, vol_data: Dict) -> Dict[str, Any]:
        """Аналіз MTF конвергенції для символу"""
        convergence_data = {
            'imbalance': {'score': 0, 'confirmed': False},
            'momentum': {'score': 0, 'confirmed': False},
            'overall': {'score': 0, 'confirmed': False}
        }
        
        try:
            # Отримуємо MTF дані
            if 'mtf_analysis' in imb_data:
                imb_mtf = imb_data['mtf_analysis'].get('convergence', {})
                convergence_data['imbalance'] = {
                    'score': imb_mtf.get('score', 0),
                    'confirmed': imb_mtf.get('confirmed', False),
                    'alignment': imb_mtf.get('alignment', 'NONE')
                }
            
            if 'mtf_analysis' in vol_data:
                vol_mtf = vol_data['mtf_analysis'].get('convergence', {})
                convergence_data['momentum'] = {
                    'score': vol_mtf.get('score', 0),
                    'confirmed': vol_mtf.get('confirmed', False),
                    'alignment': vol_mtf.get('alignment', 'NONE')
                }
            
            # Загальна оцінка
            overall_score = (convergence_data['imbalance']['score'] + 
                           convergence_data['momentum']['score']) / 2
            overall_confirmed = (convergence_data['imbalance']['confirmed'] and 
                               convergence_data['momentum']['confirmed'])
            
            convergence_data['overall'] = {
                'score': overall_score,
                'confirmed': overall_confirmed
            }
            
        except Exception as e:
            logger.error(f"[MTF_CONV_ANALYSIS] Error for {symbol}: {e}")
        
        return convergence_data

    def _evaluate_mtf_signal_quality(self, symbol: str, signal: Dict, 
                                   convergence: Dict) -> Dict[str, Any]:
        """Оцінка якості сигналу на основі MTF конвергенції"""
        quality = {
            'score': 0,
            'confirmed': False,
            'confirmed_timeframes': 0,
            'recommendation': 'HOLD'
        }
        
        try:
            # Базові параметри сигналу
            action = signal.get('action', 'HOLD')
            strength = signal.get('strength', 0)
            
            if action == 'HOLD':
                return quality
            
            # MTF дані з сигналу
            mtf_data = signal.get('mtf_data', {})
            imb_conv = mtf_data.get('imbalance_convergence', {})
            mom_conv = mtf_data.get('momentum_convergence', {})
            
            # Кількість підтверджених таймфреймів
            confirmed_timeframes = 0
            if imb_conv.get('confirmed', False):
                confirmed_timeframes += 1
            if mom_conv.get('confirmed', False):
                confirmed_timeframes += 1
            
            # Розрахунок оцінки
            base_score = strength / 5.0
            
            # Бонус за MTF конвергенцію
            conv_score = convergence['overall']['score']
            mtf_bonus = conv_score * 0.5
            
            # Бонус за кількість підтверджених таймфреймів
            tf_bonus = confirmed_timeframes * 0.2
            
            # Фінальна оцінка
            quality_score = min(1.0, base_score + mtf_bonus + tf_bonus)
            
            # Рекомендація
            if quality_score > 0.8 and confirmed_timeframes >= 2:
                recommendation = 'STRONG_ENTER'
            elif quality_score > 0.6 and confirmed_timeframes >= 1:
                recommendation = 'ENTER'
            elif quality_score > 0.4:
                recommendation = 'CAUTIOUS_ENTER'
            else:
                recommendation = 'HOLD'
            
            quality.update({
                'score': quality_score,
                'confirmed': convergence['overall']['confirmed'],
                'confirmed_timeframes': confirmed_timeframes,
                'recommendation': recommendation,
                'conv_score': conv_score,
                'base_score': base_score
            })
            
        except Exception as e:
            logger.error(f"[MTF_QUALITY] Error for {symbol}: {e}")
        
        return quality

    async def _optimized_maybe_open(self, symbol: str, sig: Dict, ob, vol_data: Dict, 
                                  mtf_quality: Dict):
        """Оновлена логіка відкриття з MTG фільтрами"""
        can_open = await self._fast_check_exchange_position_status(symbol)
        if not can_open:
            return

        if not self._quick_open_checks(symbol, sig):
            return

        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)
        
        # 🆕 MTG фільтр
        mtf_filter_enabled = getattr(self.executor.tcfg, 'enable_mtf_filter', True)
        
        if mtf_filter_enabled:
            if mtf_quality['recommendation'] in ['HOLD', 'CAUTIOUS_ENTER']:
                logger.debug(f"[MTF_FILTER] {symbol}: MTG recommendation is {mtf_quality['recommendation']}, skipping")
                return
        
        if action == "HOLD" or strength < self.executor.tcfg.entry_signal_min_strength:
            return
        
        # 🆕 MTG-покращена перевірка спреду
        mtf_data = sig.get('mtf_data', {})
        if mtf_data.get('confirmed', False):
            spread_factor = sig.get('factors', {}).get('spread', 0)
            if spread_factor < -0.5:
                logger.warning(f"[MTF_SPREAD] {symbol}: Spread too wide but MTF confirmed, proceeding")
        
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
        
        # 🆕 Додаємо MTG якість до логу
        mtf_score = mtf_quality.get('score', 0)
        logger.info(f"[MTF_ENTER] {symbol}: {effective_action} with MTF score {mtf_score:.2f}, "
                   f"signal {signal_info}")
        
        await self.executor.open_position_limit(
            symbol=symbol,
            direction=effective_action,
            ref_price=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            is_reversed=is_reverse,
            double_size=double_size,
            signal_info=signal_info,
            volatility_data=vol_data,
            mtf_quality=mtf_quality  # 🆕 Передаємо MTG якість
        )
        
        self._last_open_ts[symbol] = time.time()
        if action != "HOLD":
            self._last_trade_time[symbol] = time.time()

    async def _fast_check_exchange_position_status(self, symbol: str) -> bool:
        """ШВИДКА перевірка статусу"""
        current_time = time.time()
        
        if symbol in self._position_status_cache:
            cached = self._position_status_cache[symbol]
            if current_time - cached['timestamp'] < self._cache_ttl:
                return cached['is_open']
        
        try:
            async with asyncio.timeout(5):
                pos = self.storage.get_position(symbol)
                if not pos:
                    result = True
                else:
                    result = pos.status != "OPEN"
                
                self._position_status_cache[symbol] = {
                    'is_open': result,
                    'timestamp': current_time
                }
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"❌ [FAST_STATUS_CHECK] Timeout for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ [FAST_STATUS_CHECK] Error for {symbol}: {e}")
            return True

    def _quick_open_checks(self, symbol: str, sig: Dict) -> bool:
        """ШВИДКІ перевірки з MTG фільтрами"""
        current_time = time.time()
        
        last_trade_time = self._last_trade_time.get(symbol, 0)
        if current_time - last_trade_time < self.executor.tcfg.min_time_between_trades_sec:
            return False

        last_close = self._last_close_ts.get(symbol, 0)
        if current_time - last_close < self.executor.tcfg.reopen_cooldown_sec:
            return False

        # 🆕 MTG фільтр: O'Hara score з MTG даними
        mtf_data = sig.get('mtf_data', {})
        mtf_confirmed = mtf_data.get('confirmed', False)
        
        ohara_score = sig.get("ohara_score", 0)
        
        # Перевіряємо чи доступна ця налаштування
        if hasattr(settings.ohara, 'enable_combined_ohara_score') and settings.ohara.enable_combined_ohara_score:
            if not mtf_confirmed and ohara_score < getattr(settings.ohara, 'min_ohara_score_for_trade', 5):
                logger.debug(f"[MTG_O'HARA_FILTER] {symbol}: O'Hara score too low ({ohara_score}) without MTF confirmation")
                return False
            elif mtf_confirmed and ohara_score < getattr(settings.ohara, 'min_ohara_score_for_trade', 5) - 1:
                logger.debug(f"[MTG_O'HARA_FILTER] {symbol}: MTF confirmed but O'Hara still too low")
                return False

        if getattr(self.executor.tcfg, 'enable_aggressive_filtering', True):
            raw_values = sig.get('factors', {}).get('raw_values', {})
            momentum_score = raw_values.get('momentum_score', 0)
            if abs(momentum_score) > 90 and sig.get('strength', 0) >= 4:
                mtf_momentum = mtf_data.get('momentum_convergence', {})
                if not mtf_momentum.get('confirmed', False):
                    logger.debug(f"[MTG_MOMENTUM_FILTER] {symbol}: Extreme momentum without MTF confirmation")
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
                double_size = getattr(self.executor.tcfg, 'reverse_double_size', False)
                logger.info(f"[REVERSE] 🔄 {symbol}: closing {current_pos.side} and opening {action}")
                self._reverse_pending[symbol] = True

        return is_reverse, double_size

    async def _fast_create_signal_info(self, symbol: str, action: str, 
                                      strength: int, sig: Dict, is_reverse: bool) -> str:
        """Швидке створення інформації про сигнал з MTF даними"""
        try:
            signal_parts = []
            if is_reverse:
                signal_parts.append("REVERSE")
            display_action = "SELL" if action == "BUY" else "BUY" if action == "SELL" else action
            if getattr(self.executor.tcfg, 'reverse_signals', False):
                signal_parts.append(f"{display_action.upper()}{strength}")
            else:
                signal_parts.append(f"{action.upper()}{strength}")

            factors = sig.get('factors', {})
            raw_values = factors.get('raw_values', {})
            if raw_values:
                imb_score = raw_values.get('imbalance_score', 0)
                mom_score = raw_values.get('momentum_score', 0)
                ohara_score = sig.get('ohara_score', 0)
                mtf_data = sig.get('mtf_data', {})
                mtf_confirmed = mtf_data.get('confirmed', False)
                
                signal_parts.append(f"(imb:{imb_score:.0f},mom:{mom_score:.0f},oh:{ohara_score}")
                if mtf_confirmed:
                    signal_parts.append(f"MTF✓)")

            return " ".join(signal_parts)

        except Exception as e:
            logger.error(f"❌ [FAST_SIGNAL] {symbol}: {e}")
            return f"{action.upper()}{strength}" + (" (reverse)" if is_reverse else "")

    async def _optimized_maybe_close(self, symbol: str, sig: Dict, ob, vol_data: Dict, 
                                   mtf_quality: Dict):
        """Оновлена логіка закриття з MTG аналізом"""
        can_process = await self._fast_check_exchange_position_status(symbol)
        if not can_process:
            return
            
        pos = self.storage.get_position(symbol)
        if not pos or pos.status != "OPEN":
            return

        if not self._quick_close_checks(symbol, pos):
            return

        current_time = time.time()
        
        # Адаптивний lifetime з MTG корекцією
        if hasattr(pos, 'max_lifetime_sec') and pos.max_lifetime_sec > 0:
            max_life = pos.max_lifetime_sec
        else:
            current_volatility = vol_data.get('recent_volatility', 0.1)
            max_life = self.executor.risk.get_adaptive_lifetime_seconds(symbol, current_volatility)
        
        # 🆕 MTG-корекція lifetime при сильній конвергенції
        if mtf_quality['score'] > 0.8:
            max_life *= 1.2
            logger.debug(f"[MTG_LIFETIME] {symbol}: Extending lifetime due to high MTG convergence")
        
        # Пріоритетна перевірка TIME_EXIT
        if current_time - pos.timestamp > max_life:
            lifetime_min = (current_time - pos.timestamp) / 60.0
            logger.info(f"[MTG_TIME_EXIT] {symbol} {pos.side}: TIME_EXIT "
                       f"({lifetime_min:.1f}min > {max_life/60:.1f}min), "
                       f"MTG score was {mtf_quality['score']:.2f}")
            await self.executor.close_position(symbol, reason="TIME_EXIT")
            self._last_close_ts[symbol] = current_time
            return

        # Перевірка реверсу
        if symbol in self._reverse_pending and self._reverse_pending[symbol]:
            logger.info(f"[MTG_REVERSE] {symbol} {pos.side}: REVERSE, "
                       f"MTG score {mtf_quality['score']:.2f}")
            await self.executor.close_position(symbol, reason="REVERSE")
            self._reverse_pending.pop(symbol, None)
            self._last_close_ts[symbol] = current_time
            return

        # Перевірка протилежного сигналу з MTG фільтром
        action = sig.get("action", "HOLD")
        strength = sig.get("strength", 0)
        opposite_strength_req = self.executor.tcfg.close_on_opposite_strength
        
        # 🆕 MTG-фільтр: закриваємо тільки при підтвердженому протилежному сигналі
        if (pos.side == "LONG" and action == "SELL" and strength >= opposite_strength_req) or \
           (pos.side == "SHORT" and action == "BUY" and strength >= opposite_strength_req):
            
            # Перевіряємо MTG підтвердження
            if mtf_quality['confirmed']:
                logger.info(f"[MTG_CLOSE] {symbol} {pos.side}: Strong opposite signal with MTF confirmation")
                await self.executor.close_position(symbol, reason="opp_signal_mtf")
                self._last_close_ts[symbol] = current_time
            elif mtf_quality['score'] > 0.6:
                logger.info(f"[MTG_CLOSE] {symbol} {pos.side}: Opposite signal with good MTG score")
                await self.executor.close_position(symbol, reason="opp_signal")
                self._last_close_ts[symbol] = current_time
            else:
                logger.debug(f"[MTG_HOLD] {symbol}: Opposite signal but weak MTG confirmation, holding")
        
        return

    def _quick_close_checks(self, symbol: str, pos: Position) -> bool:
        """ШВИДКІ перевірки для закриття"""
        current_time = time.time()
        
        if current_time - pos.timestamp < self.executor.tcfg.min_position_hold_time_sec:
            return False
            
        return True

    def get_last_signal(self, symbol: str) -> Optional[Dict]:
        return self._last_signal.get(symbol)

    async def close_all(self, reason: str = "force_close"):
        """Закрити всі позиції"""
        symbols = list(self.storage.positions.keys())
        close_tasks = []
        for sym in symbols:
            pos = self.storage.positions[sym]
            if pos.status == "OPEN":
                task = asyncio.create_task(self.executor.close_position(sym, reason=reason))
                close_tasks.append(task)
        
        await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info(f"[ORCH] 🔒 Force closed {len(close_tasks)} positions")