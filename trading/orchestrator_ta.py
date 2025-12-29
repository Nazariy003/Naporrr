"""
Trading Orchestrator with Technical Analysis Support
Supports both modes:
1. Original: Orderbook imbalance + volume analysis
2. New: Technical Analysis (chart patterns + candlestick patterns + indicators)
"""

import asyncio
import time
from typing import Optional, Dict, Any
from utils.logger import logger
from config.settings import settings
from data.storage import DataStorage
# Видалено імпорти старих файлів: analysis.imbalance, analysis.volume, analysis.signals
from analysis.ta_adapter import TechnicalAnalysisAdapter
from trading.executor import TradeExecutor


class TradingOrchestratorTA:
    """
    Trading Orchestrator with TA Support
    
    Modes:
    - TA Mode (enable_ta_mode=True): Uses chart patterns, candlestick patterns, and technical indicators
    - Legacy Mode (enable_ta_mode=False): Uses orderbook imbalance and volume analysis
    """
    
    def __init__(
        self,
        storage: DataStorage,
        imb_analyzer,  # Тепер None, бо файл видалено
        vol_analyzer,  # Тепер None, бо файл видалено
        signal_generator,  # Тепер None, бо файл видалено
        executor: TradeExecutor
    ):
        self.storage = storage
        # Видалено призначення imb_analyzer, vol_analyzer, signal_generator - файли видалені
        self.executor = executor
        
        # TA Mode components
        self.ta_enabled = settings.technical_analysis.enable_ta_mode
        if self.ta_enabled:
            self.ta_adapter = TechnicalAnalysisAdapter(storage)
            logger.info("✅ [ORCH_TA] Technical Analysis mode ENABLED")
        else:
            self.ta_adapter = None
            logger.warning("⚠️ [ORCH_TA] Legacy mode requested but modules removed - using TA only")
        
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_trade_time: Dict[str, float] = {}
        self._last_signal: Dict[str, Dict] = {}
        
        # Cache for TA data loading status
        self._ta_data_ready: Dict[str, bool] = {}
    
    async def start(self):
        """Start the orchestrator"""
        if self._task:
            return
        
        self._running = True
        
        # Завантажуємо історичні свічки для TA після collector.start()
        if self.ta_enabled:
            symbols = settings.pairs.trade_pairs
            for symbol in symbols:
                await self.ta_adapter.load_historical_candles_from_api(symbol)
                await asyncio.sleep(0.1)  # Не перевантажувати API
        
        self._task = asyncio.create_task(self._main_loop())
        
        mode = "Technical Analysis (TA)" if self.ta_enabled else "Legacy (Disabled)"
        logger.info(f"✅ [ORCH_TA] Trading orchestrator started in {mode} mode")
    
    async def stop(self):
        """Stop the orchestrator"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("✅ [ORCH_TA] Trading orchestrator cancelled")
            except Exception as e:
                logger.error(f"❌ [ORCH_TA] error during stop: {e}")
            self._task = None
        logger.info("✅ [ORCH_TA] Trading orchestrator stopped")
    
    async def _main_loop(self):
        """Main decision loop"""
        interval = settings.trading.decision_interval_sec
        
        # Create symbol batches for parallel processing
        symbols = settings.pairs.trade_pairs
        batch_size = settings.technical_analysis.orchestrator_batch_size  # From configuration
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        batch_index = 0
        
        while self._running:
            start_time = time.time()
            
            try:
                # Process current batch
                current_batch = symbol_batches[batch_index]
                await self._process_symbol_batch(current_batch)
                
                # Rotate to next batch
                batch_index = (batch_index + 1) % len(symbol_batches)
                
            except Exception as e:
                logger.error(f"❌ [ORCH_TA] Error in main loop: {e}", exc_info=True)
            
            # Sleep for remainder of interval
            elapsed = time.time() - start_time
            await asyncio.sleep(max(0.0, interval - elapsed))
    
    async def _process_symbol_batch(self, symbols: list):
        """Process a batch of symbols"""
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_symbol(symbol))
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol"""
        try:
            # Check if we can trade this symbol
            if not await self._can_trade_symbol(symbol):
                return
            
            # Generate signal based on mode
            if self.ta_enabled:
                signal = await self._generate_ta_signal(symbol)
            else:
                # Legacy disabled - fallback to hold
                signal = self._create_hold_signal(symbol, "legacy_disabled")
            
            # Store signal
            self._last_signal[symbol] = signal
            
            # Execute trade if signal is strong enough
            if signal['action'] in ['BUY', 'SELL'] and signal['strength'] >= settings.trading.entry_signal_min_strength:
                await self._execute_signal(symbol, signal)
        
        except Exception as e:
            logger.error(f"❌ [ORCH_TA] Error processing {symbol}: {e}")
    
    async def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade this symbol"""
        # Check if already have position
        position = self.storage.get_position(symbol)
        if position and position.status == "OPEN":
            return False  # Already have position
        
        # Check cooldown
        min_time_between_trades = settings.trading.min_time_between_trades_sec
        last_trade_time = self._last_trade_time.get(symbol, 0)
        
        if time.time() - last_trade_time < min_time_between_trades:
            return False
        
        return True
    
    async def _generate_ta_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate signal using Technical Analysis"""
        try:
            # Спочатку спробуємо оновити з трейдів для історії
            self.ta_adapter.update_from_trades(symbol)
            
            # Потім оновимо з orderbook для поточних даних
            self.ta_adapter.update_from_orderbook(symbol)
            
            # Check if we have enough TA data
            candle_count = self.ta_adapter.get_candle_count(symbol)
            if candle_count < self.ta_adapter.min_candles_for_ta:
                # Only log once when transitioning
                if not self._ta_data_ready.get(symbol, False):
                    logger.info(
                        f"⏳ [TA_DATA] {symbol}: Building candle history from trades... "
                        f"({candle_count}/{self.ta_adapter.min_candles_for_ta})"
                    )
                
                return self._create_hold_signal(symbol, "insufficient_ta_data")
            else:
                # Data just became ready
                if not self._ta_data_ready.get(symbol, False):
                    logger.info(f"✅ [TA_DATA] {symbol}: Sufficient candle data ready for TA analysis ({candle_count} candles)")
                    self._ta_data_ready[symbol] = True
            
            # Generate TA signal
            signal = self.ta_adapter.generate_signal(symbol)
            return signal
        
        except Exception as e:
            logger.error(f"Error generating TA signal for {symbol}: {e}")
            return self._create_hold_signal(symbol, "error")
    
    async def _generate_legacy_signal(self, symbol: str) -> Dict[str, Any]:
        """Legacy mode disabled - return hold"""
        return self._create_hold_signal(symbol, "legacy_removed")
    
    async def _execute_signal(self, symbol: str, signal: Dict[str, Any]):
        """Execute a trading signal"""
        try:
            action = signal['action']
            strength = signal['strength']
            
            logger.info(
                f"💼 [ORCH_TA] Executing {action}{strength} for {symbol} "
                f"(confidence={signal.get('confidence', 0):.0f}%, "
                f"reason={signal.get('reason', 'unknown')})"
            )
            
            # In TA mode, we have stop-loss and take-profit from signal
            if self.ta_enabled:
                stop_loss = signal.get('stop_loss', 0)
                take_profit = signal.get('take_profit', 0)
                position_size_pct = signal.get('position_size_pct', 0.02)
                leverage = signal.get('leverage_recommended', 3.0)
                
                logger.info(
                    f"   Entry: {signal.get('entry_price', 0):.2f}, "
                    f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                    f"R:R={signal.get('risk_reward_ratio', 0):.1f}:1, "
                    f"Size: {position_size_pct*100:.1f}%, Leverage: {leverage}x"
                )
                
                # TODO: Integrate with executor to actually place orders
                # For now, just log
                logger.info(
                    f"   Pattern: {signal.get('chart_pattern') or signal.get('candlestick_pattern') or 'None'}, "
                    f"Trend: {signal.get('trend', 'NEUTRAL')}, "
                    f"RSI: {signal.get('rsi_signal', 'N/A')}, "
                    f"MACD: {signal.get('macd_signal', 'N/A')}"
                )
            
            # Update last trade time
            self._last_trade_time[symbol] = time.time()
        
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
    
    def _create_hold_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Create a HOLD signal"""
        return {
            "symbol": symbol,
            "action": "HOLD",
            "strength": 0,
            "score_smoothed": 0,
            "score_raw": 0,
            "reason": reason,
            "confidence": 0
        }