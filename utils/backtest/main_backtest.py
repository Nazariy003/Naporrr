# utils/backtest/main_backtest.py
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import settings
from utils.logger import logger
from utils.notifications import notifier
from utils.backtest.data_collector import DataCollector
from utils.backtest.replay_engine import ReplayEngine
from utils.backtest.optimizer import ParameterOptimizer
from utils.backtest.validator import WalkForwardValidator
from utils.backtest.settings_updater import SettingsUpdater
from utils.backtest.metrics import MetricsCalculator

class BacktestOrchestrator:
    """
    🎯 Головний оркестратор бектестингу
    
    Workflow:
    1. Data Collection (24/7)
    2. Periodic Backtest (кожні N годин)
    3. Parameter Optimization
    4. Walk-Forward Validation
    5. Auto-apply (якщо увімкнено)
    """
    
    def __init__(self, storage, signal_generator):
        self.storage = storage
        self.signal_generator = signal_generator
        
        # Компоненти
        self.data_collector = DataCollector(settings.backtest.data_storage_path)
        self.replay_engine = ReplayEngine(settings.backtest.data_storage_path)
        self.optimizer = ParameterOptimizer()
        self.validator = WalkForwardValidator(
            train_ratio=settings.backtest.walk_forward_train_ratio
        )
        self.settings_updater = SettingsUpdater()
        
        # Стан
        self._running = False
        self._last_backtest = 0
        self._backtest_task = None
        
    async def start(self):
        """Запуск оркестратора"""
        if not settings.backtest.enable_backtest:
            logger.info("⏹️ [BACKTEST] Disabled in settings")
            return
        
        logger.info("🚀 [BACKTEST] Starting orchestrator...")
        
        # Запуск збору даних
        await self.data_collector.start(self.storage, self.signal_generator)
        
        # Запуск циклічного бектесту
        self._running = True
        self._backtest_task = asyncio.create_task(self._backtest_loop())
        
        logger.info("✅ [BACKTEST] Orchestrator started")
    
    async def stop(self):
        """Зупинка оркестратора"""
        logger.info("🛑 [BACKTEST] Stopping orchestrator...")
        
        self._running = False
        
        if self._backtest_task:
            self._backtest_task.cancel()
            try:
                await self._backtest_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ [BACKTEST] Orchestrator stopped")
    
    async def _backtest_loop(self):
        """Циклічний запуск бектесту"""
        # Чекаємо до запланованого часу
        await self._wait_until_scheduled_time()
        
        while self._running:
            try:
                current_time = time.time()
                
                # Перевіряємо чи час запускати
                hours_since_last = (current_time - self._last_backtest) / 3600
                
                if hours_since_last >= settings.backtest.cycle_hours:
                    logger.info("="*70)
                    logger.info("🎬 [BACKTEST] Starting scheduled backtest run...")
                    logger.info("="*70)
                    
                    # Запуск повного циклу
                    await self._run_full_backtest_cycle()
                    
                    self._last_backtest = current_time
                    
                    logger.info("="*70)
                    logger.info("✅ [BACKTEST] Cycle completed")
                    logger.info("="*70)
                
                # Чекаємо 1 годину перед наступною перевіркою
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ [BACKTEST_LOOP] Error: {e}")
                await asyncio.sleep(3600)
    
    async def _wait_until_scheduled_time(self):
        """Очікування до запланованого часу"""
        target_hour, target_minute = map(int, settings.backtest.backtest_start_time.split(':'))
        
        now = datetime.utcnow()
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        if target_time < now:
            target_time += timedelta(days=1)
        
        wait_seconds = (target_time - now).total_seconds()
        
        logger.info(f"⏰ [BACKTEST] Scheduled at {settings.backtest.backtest_start_time} UTC")
        logger.info(f"⏰ [BACKTEST] First run in {wait_seconds/3600:.1f} hours")
        
        await asyncio.sleep(wait_seconds)
    
    async def _run_full_backtest_cycle(self):
        """Повний цикл бектесту та оптимізації"""
        start_time = time.time()
        
        try:
            # 1. Визначення періоду
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=settings.backtest.lookback_days)
            
            symbols = settings.backtest.optimization_symbols or settings.pairs.trade_pairs
            
            logger.info(f"📅 [BACKTEST] Period: {start_date.date()} to {end_date.date()}")
            logger.info(f"💎 [BACKTEST] Symbols: {', '.join(symbols)}")
            
            # 2. Оптимізація параметрів
            if settings.backtest.enable_optimization:
                logger.info("\n🔍 [BACKTEST] Phase 1: Parameter Optimization")
                
                best_result, all_results = self.optimizer.optimize(
                    replay_engine=self.replay_engine,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    max_combinations=settings.backtest.max_optimization_combinations
                )
                
                if not best_result:
                    logger.error("❌ [BACKTEST] Optimization failed")
                    return
                
                best_params = best_result['params']
                best_metrics = best_result['metrics']
                
                logger.info(f"🏆 [BACKTEST] Best optimization score: {best_result['score']:.4f}")
                self._log_metrics(best_metrics)
            else:
                logger.info("⏩ [BACKTEST] Optimization skipped")
                return
            
            # 3. Walk-Forward Validation
            if settings.backtest.enable_walk_forward:
                logger.info("\n🔬 [BACKTEST] Phase 2: Walk-Forward Validation")
                
                validation_result = self.validator.validate(
                    replay_engine=self.replay_engine,
                    optimizer=self.optimizer,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols
                )
                
                if validation_result:
                    consistency = validation_result['summary']['avg_consistency_score']
                    logger.info(f"📊 [BACKTEST] Validation consistency: {consistency:.1f}%")
                    
                    recommendation = validation_result['recommendation']
                    logger.info(f"💡 [BACKTEST] Verdict: {recommendation['verdict']}")
                    logger.info(f"💡 [BACKTEST] {recommendation['message']}")
                    
                    if not recommendation['should_apply']:
                        logger.warning("⚠️ [BACKTEST] Validation failed, parameters will not be applied")
                        
                        # Нотифікація
                        if settings.backtest.notify_on_completion:
                            await self._send_notification(
                                "⚠️ Backtest Completed",
                                f"Validation failed: {recommendation['message']}"
                            )
                        return
            else:
                logger.info("⏩ [BACKTEST] Validation skipped")
            
            # 4. Порівняння з поточними результатами
            logger.info("\n📊 [BACKTEST] Phase 3: Comparison with Current Performance")
            
            comparison = self.optimizer.compare_with_current(best_params, best_metrics)
            
            self._log_comparison(comparison)
            
            # 5. Рішення про застосування
            should_apply = comparison['should_apply']
            
            if should_apply and settings.backtest.auto_apply_params:
                logger.info("\n✅ [BACKTEST] Phase 4: Applying New Parameters")
                
                # Перевірка на manual approval
                if settings.backtest.require_manual_approval:
                    logger.info("⏸️ [BACKTEST] Manual approval required")
                    
                    await self._request_manual_approval(best_params, comparison)
                else:
                    # Автоматичне застосування
                    success = self.settings_updater.update_parameters(
                        new_params=best_params,
                        gradual=settings.backtest.gradual_adjustment,
                        adjustment_factor=settings.backtest.adjustment_factor
                    )
                    
                    if success:
                        logger.info("✅ [BACKTEST] Parameters updated successfully")
                        logger.info("🔄 [BACKTEST] Restart bot to apply new settings")
                        
                        # Нотифікація
                        if settings.backtest.notify_on_better_params:
                            await self._send_notification(
                                "✅ Parameters Updated",
                                self._format_update_message(best_params, comparison)
                            )
                    else:
                        logger.error("❌ [BACKTEST] Failed to update parameters")
            else:
                if not should_apply:
                    logger.info("⏹️ [BACKTEST] Current parameters are already optimal")
                else:
                    logger.info("⏹️ [BACKTEST] Auto-apply disabled, manual update required")
            
            # 6. Фінальна нотифікація
            elapsed = time.time() - start_time
            
            if settings.backtest.notify_on_completion:
                await self._send_notification(
                    "📊 Backtest Completed",
                    f"Duration: {elapsed/60:.1f}min\n"
                    f"Best Score: {best_result['score']:.4f}\n"
                    f"Win Rate: {best_metrics.get('win_rate', 0):.1f}%\n"
                    f"Profit Factor: {best_metrics.get('profit_factor', 0):.2f}\n"
                    f"Applied: {'Yes' if should_apply and settings.backtest.auto_apply_params else 'No'}"
                )
            
            logger.info(f"\n⏱️ [BACKTEST] Total duration: {elapsed/60:.1f} minutes")
            
        except Exception as e:
            logger.error(f"❌ [BACKTEST] Cycle error: {e}", exc_info=True)
            
            if settings.backtest.notify_on_completion:
                await self._send_notification(
                    "❌ Backtest Error",
                    f"Error: {str(e)}"
                )
    
    def _log_metrics(self, metrics: dict):
        """Логування метрик"""
        logger.info("📊 [METRICS]")
        logger.info(f"  • Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  • Win Rate: {metrics.get('win_rate', 0):.1f}%")
        logger.info(f"  • Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"  • Total PnL: ${metrics.get('total_pnl', 0):.2f}")
        logger.info(f"  • Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
        logger.info(f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    
    def _log_comparison(self, comparison: dict):
        """Логування порівняння"""
        logger.info("📈 [COMPARISON]")
        
        for metric, data in comparison['improvement'].items():
            current = data['current']
            new = data['new']
            change = data['change_pct']
            improved = data['improved']
            
            symbol = "📈" if improved else "📉"
            logger.info(f"  {symbol} {metric}: {current:.2f} -> {new:.2f} ({change:+.1f}%)")
        
        logger.info(f"\n💡 Decision: {comparison['reason']}")
    
    async def _send_notification(self, title: str, message: str):
        """Відправка нотифікації"""
        try:
            await notifier.send(f"{title}\n\n{message}")
        except Exception as e:
            logger.error(f"❌ [NOTIFICATION] Error: {e}")
    
    def _format_update_message(self, params: dict, comparison: dict) -> str:
        """Форматування повідомлення про оновлення"""
        msg = "Parameters updated:\n\n"
        
        for param_name, param_value in list(params.items())[:5]:
            msg += f"• {param_name}: {param_value}\n"
        
        msg += f"\nImprovements:\n"
        
        for metric, data in list(comparison['improvement'].items())[:3]:
            if data['improved']:
                msg += f"• {metric}: {data['change_pct']:+.1f}%\n"
        
        return msg
    
    async def _request_manual_approval(self, params: dict, comparison: dict):
        """Запит ручного підтвердження"""
        message = (
            "🤔 Manual Approval Required\n\n"
            "Better parameters found!\n\n"
            f"{self._format_update_message(params, comparison)}\n"
            "React to this message or update settings manually."
        )
        
        await self._send_notification("🔔 Approval Needed", message)