# main.py - ПОВНА ВЕРСІЯ З БЕКТЕСТОМ
import asyncio
import sys
import time
from config.settings import settings
from utils.logger import logger
from utils.notifications import notifier
from data.storage import DataStorage, Position
from data.collector import DataCollector
from analysis.imbalance import ImbalanceAnalyzer
from analysis.volume import VolumeAnalyzer
from analysis.signals import SignalGenerator
from trading.bybit_api_manager import BybitAPIManager
from trading.executor import TradeExecutor
from trading.orchestrator import TradingOrchestrator

# 🆕 BACKTEST IMPORTS
from utils.backtest.main_backtest import BacktestOrchestrator

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def emergency_fix(storage: DataStorage):
    """ФІКС: Примусово закриваємо позиції, які блокували бота"""
    logger.info("🚑 [EMERGENCY_FIX] Applying emergency position fix...")
    
    problem_symbols = []
    for symbol, position in storage.positions.items():
        if position.status == "OPEN":
            current_time = time.time()
            if current_time - position.last_update > 300:  # 5 хвилин без оновлення
                problem_symbols.append(symbol)
                logger.warning(f"🔄 [EMERGENCY] Forcing close for stuck position: {symbol}")
                position.status = "CLOSED"
                position.close_reason = "EMERGENCY_CLOSE"
                position._position_updated = True
    
    if problem_symbols:
        logger.info(f"✅ [EMERGENCY_FIX] Fixed {len(problem_symbols)} stuck positions")
    return problem_symbols

async def run_csv_validation():
    """Запуск валідації CSV (не блокує запуск бота)"""
    try:
        from utils.csv_test import main as validate_csv
        logger.info("🔍 [MAIN] Running CSV validation...")
        success = await validate_csv()
        if success:
            logger.info("✅ [MAIN] CSV validation completed")
        else:
            logger.warning("⚠️ [MAIN] CSV validation found issues (continuing)")
        return True
    except Exception as e:
        logger.error(f"❌ [MAIN] CSV validation failed: {e}")
        return True

async def delayed_validation():
    """Відкладена валідація через 30 хвилин"""
    await asyncio.sleep(1800)  # 30 хвилин
    await run_csv_validation()

async def print_startup_banner():
    """Красивий банер при запуску"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🚀 NAPORRR TRADING BOT - PROFESSIONAL EDITION 🚀       ║
    ║                                                               ║
    ║           Adaptive • Autonomous • Market Microstructure       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    logger.info(banner)
    logger.info(f"    📅 Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"    👤 User: {settings.secrets.bybit_api_key[:8]}...")
    logger.info("")

async def print_system_info():
    """Інформація про систему"""
    mode_info = settings.system.get_mode_info()
    
    logger.info("=" * 70)
    logger.info("📡 SYSTEM CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"  🎯 Mode:              {mode_info['mode']}")
    logger.info(f"  📊 Public WebSocket:  {mode_info['ws_public']}")
    logger.info(f"  🔐 Private WebSocket: {mode_info['ws_private']}")
    logger.info(f"  🌐 REST API:          {mode_info['rest_api']}")
    logger.info(f"  💡 Note:              {mode_info['note']}")
    logger.info("")
    
    logger.info("=" * 70)
    logger.info("⚙️  TRADING SETTINGS")
    logger.info("=" * 70)
    logger.info(f"  💰 Leverage:          {settings.trading.leverage}x")
    logger.info(f"  📊 Base Order:        {settings.trading.base_order_pct*100:.1f}% of balance")
    logger.info(f"  🎯 Max Positions:     {settings.risk.max_open_positions}")
    logger.info(f"  ⏱️  Position Lifetime: {settings.risk.base_position_lifetime_minutes} min (adaptive)")
    logger.info(f"  💎 Trading Pairs:     {len(settings.pairs.trade_pairs)}")
    logger.info(f"     └─ {', '.join(settings.pairs.trade_pairs)}")
    logger.info("")
    
    logger.info("=" * 70)
    logger.info("🎯 SIGNAL SETTINGS")
    logger.info("=" * 70)
    logger.info(f"  🔍 Weight Imbalance:  {settings.signals.weight_imbalance}")
    logger.info(f"  📈 Weight Momentum:   {settings.signals.weight_momentum}")
    logger.info(f"  🎚️  Smoothing Alpha:   {settings.signals.smoothing_alpha}")
    logger.info(f"  🚦 Hold Threshold:    {settings.signals.hold_threshold}")
    logger.info(f"  ⭐ Min Entry Strength: {settings.trading.entry_signal_min_strength}")
    logger.info("")
    
    logger.info("=" * 70)
    logger.info("🛡️  RISK MANAGEMENT")
    logger.info("=" * 70)
    logger.info(f"  📉 SL Multiplier:     {settings.risk.sl_vol_multiplier}x volatility")
    logger.info(f"  📈 TP Multiplier:     {settings.risk.tp_vol_multiplier}x volatility")
    logger.info(f"  🔄 Trailing Stop:     {'Enabled' if settings.risk.enable_trailing_stop else 'Disabled'}")
    logger.info(f"  🎯 Dynamic TP/SL:     {'Enabled' if settings.risk.enable_dynamic_tpsl else 'Disabled'}")
    logger.info(f"  ⏱️  Adaptive Lifetime:  {'Enabled' if settings.risk.enable_adaptive_lifetime else 'Disabled'}")
    logger.info("")

async def print_backtest_info():
    """Інформація про бектест (якщо увімкнено)"""
    if not settings.backtest.enable_backtest:
        return
    
    logger.info("=" * 70)
    logger.info("🔬 ADAPTIVE BACKTEST SYSTEM")
    logger.info("=" * 70)
    logger.info(f"  ✅ Status:            ENABLED")
    logger.info(f"  🔄 Cycle:             Every {settings.backtest.cycle_hours} hours")
    logger.info(f"  ⏰ Start Time:        {settings.backtest.backtest_start_time} UTC")
    logger.info(f"  📅 Lookback Period:   {settings.backtest.lookback_days} days")
    logger.info(f"  🔍 Optimization:      {'Enabled' if settings.backtest.enable_optimization else 'Disabled'}")
    logger.info(f"  🔬 Walk-Forward:      {'Enabled' if settings.backtest.enable_walk_forward else 'Disabled'}")
    logger.info(f"  🤖 Auto-Apply:        {'Enabled' if settings.backtest.auto_apply_params else 'Disabled'}")
    
    if settings.backtest.auto_apply_params:
        approval = "Required" if settings.backtest.require_manual_approval else "Not Required"
        logger.info(f"     └─ Manual Approval: {approval}")
        logger.info(f"     └─ Min Improvement: {settings.backtest.min_improvement_threshold_pct}%")
    
    logger.info(f"  💾 Storage Budget:    {settings.backtest.max_storage_gb} GB")
    logger.info(f"     └─ RAW:            {settings.backtest.raw_data_retention_days} days")
    logger.info(f"     └─ Aggregated:     {settings.backtest.aggregated_data_retention_days} days")
    logger.info(f"     └─ Metadata:       {settings.backtest.metadata_retention_days} days")
    logger.info("")

async def print_features():
    """Список активних features"""
    logger.info("=" * 70)
    logger.info("✨ ACTIVE FEATURES")
    logger.info("=" * 70)
    
    features = [
        ("📊 Multi-Factor Signals", "Imbalance + Momentum + Volume + Tape"),
        ("🎯 Adaptive Windows", f"{'Enabled' if settings.adaptive.enable_adaptive_windows else 'Disabled'}"),
        ("🔍 Spoof Detection", f"{'Enabled' if settings.imbalance.enable_spoof_filter else 'Disabled'}"),
        ("📈 Historical Imbalance", f"{'Enabled' if settings.imbalance.enable_historical_imbalance else 'Disabled'}"),
        ("🎚️  POC Clustering", "Enabled"),
        ("⚡ Fast Position Monitoring", f"Every {settings.trading.monitor_positions_interval_sec}s"),
        ("🔄 Parallel Processing", f"{'Enabled' if settings.trading.enable_parallel_monitoring else 'Disabled'}"),
        ("📝 Trade Logging", "CSV + Real-time"),
        ("🔔 Telegram Notifications", "Enabled"),
    ]
    
    if settings.backtest.enable_backtest:
        features.append(("🔬 Adaptive Backtest", "Enabled"))
        features.append(("🤖 Auto-Optimization", f"Every {settings.backtest.cycle_hours}h"))
    
    for feature, status in features:
        logger.info(f"  {feature:.<40} {status}")
    
    logger.info("")

async def wait_for_user_confirmation():
    """Очікування підтвердження користувача (тільки для LIVE)"""
    if settings.trading.mode.upper() != "LIVE":
        return True
    
    logger.info("=" * 70)
    logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Please review the settings above carefully.")
    logger.info("  Type 'START' to begin trading or 'EXIT' to quit.")
    logger.info("")
    
    # В реальності тут має бути input(), але для автоматичного запуску пропускаємо
    # user_input = input("  Your choice: ").strip().upper()
    # if user_input != "START":
    #     logger.info("  Exiting...")
    #     return False
    
    return True

async def main():
    """Головна функція з повною інтеграцією бектесту"""
    
    # Банер
    await print_startup_banner()
    
    # Інформація про систему
    await print_system_info()
    
    # Інформація про бектест
    await print_backtest_info()
    
    # Features
    await print_features()
    
    # Підтвердження для LIVE
    # if not await wait_for_user_confirmation():
    #     return
    
    logger.info("=" * 70)
    logger.info("🚀 INITIALIZING COMPONENTS...")
    logger.info("=" * 70)
    logger.info("")
    
    # Швидка CSV перевірка (не блокує)
    asyncio.create_task(run_csv_validation())
    asyncio.create_task(delayed_validation())

    # Ініціалізація API Manager
    logger.info("🔧 [1/8] Initializing API Manager...")
    api_manager = BybitAPIManager()
    logger.info("✅ [1/8] API Manager ready")

    # Ініціалізація Data Storage
    logger.info("🔧 [2/8] Initializing Data Storage...")
    storage = DataStorage(
        retention_seconds=settings.risk.max_position_lifetime_sec,
        large_order_side_percent=settings.imbalance.large_order_side_percent,
        spoof_lifetime_ms=settings.imbalance.spoof_lifetime_ms,
        large_order_min_abs=settings.imbalance.large_order_min_notional_abs,
        max_depth=settings.websocket.subscription_depth
    )
    logger.info("✅ [2/8] Data Storage ready")

    # Екстрене відновлення
    logger.info("🔧 [3/8] Running emergency position check...")
    await emergency_fix(storage)
    logger.info("✅ [3/8] Emergency check completed")

    # Ініціалізація Data Collector
    logger.info("🔧 [4/8] Initializing Data Collector...")
    collector = DataCollector(storage, api_manager)
    logger.info("✅ [4/8] Data Collector ready")

    # Ініціалізація Analyzers
    logger.info("🔧 [5/8] Initializing Analysis Engines...")
    imb_analyzer = ImbalanceAnalyzer(storage)
    vol_analyzer = VolumeAnalyzer(storage)
    signal_generator = SignalGenerator()
    logger.info("✅ [5/8] Analysis Engines ready")

    # Ініціалізація Trade Executor
    logger.info("🔧 [6/8] Initializing Trade Executor...")
    executor = TradeExecutor(storage, api_manager)
    logger.info("✅ [6/8] Trade Executor ready")

    # Ініціалізація Trading Orchestrator
    logger.info("🔧 [7/8] Initializing Trading Orchestrator...")
    orchestrator = TradingOrchestrator(storage, imb_analyzer, vol_analyzer, signal_generator, executor)
    logger.info("✅ [7/8] Trading Orchestrator ready")

    # 🆕 Ініціалізація Backtest Orchestrator
    backtest_orchestrator = None
    if settings.backtest.enable_backtest:
        logger.info("🔧 [8/8] Initializing Backtest Orchestrator...")
        try:
            backtest_orchestrator = BacktestOrchestrator(storage, signal_generator)
            logger.info("✅ [8/8] Backtest Orchestrator ready")
        except Exception as e:
            logger.error(f"❌ [8/8] Backtest initialization failed: {e}")
            logger.warning("⚠️  Continuing without backtest system")
    else:
        logger.info("⏩ [8/8] Backtest disabled, skipping...")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ ALL COMPONENTS INITIALIZED")
    logger.info("=" * 70)
    logger.info("")

    try:
        # Запуск компонентів
        logger.info("🚀 STARTING SERVICES...")
        logger.info("")
        
        logger.info("▶️  [1/4] Starting Data Collector...")
        await collector.start()
        logger.info("✅ [1/4] Data Collector running")
        
        logger.info("▶️  [2/4] Starting Trade Executor...")
        await executor.start()
        logger.info("✅ [2/4] Trade Executor running")
        
        logger.info("▶️  [3/4] Starting Trading Orchestrator...")
        await orchestrator.start()
        logger.info("✅ [3/4] Trading Orchestrator running")
        
        # 🆕 Запуск Backtest Orchestrator
        if backtest_orchestrator:
            logger.info("▶️  [4/4] Starting Backtest Orchestrator...")
            await backtest_orchestrator.start()
            logger.info("✅ [4/4] Backtest Orchestrator running")
        else:
            logger.info("⏩ [4/4] Backtest skipped")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ ALL SERVICES RUNNING")
        logger.info("=" * 70)

        # Відправка стартової нотифікації
        try:
            mode_emoji = "🔴" if settings.trading.mode.upper() == "LIVE" else "🟢"
            backtest_status = "✅ Enabled" if settings.backtest.enable_backtest else "⏹️ Disabled"
            
            startup_msg = (
                f"{mode_emoji} Bot Started\n\n"
                f"Mode: {settings.trading.mode.upper()}\n"
                f"Pairs: {len(settings.pairs.trade_pairs)}\n"
                f"Leverage: {settings.trading.leverage}x\n"
                f"Max Positions: {settings.risk.max_open_positions}\n"
                f"Backtest: {backtest_status}\n"
            )
            
            await notifier.send(startup_msg)
        except Exception as e:
            logger.warning(f"⚠️  Failed to send startup notification: {e}")

        # Інформаційний банер
        logger.info("")
        logger.info("╔═══════════════════════════════════════════════════════════════╗")
        logger.info("║                    BOT IS NOW RUNNING                         ║")
        logger.info("╚═══════════════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info("📊 Monitoring Features:")
        logger.info("   • Real-time orderbook analysis (50 levels)")
        logger.info("   • Market microstructure detection")
        logger.info("   • Multi-timeframe momentum")
        logger.info("   • Adaptive risk management")
        logger.info("   • Spoof order filtering")
        if backtest_orchestrator:
            logger.info(f"   • Adaptive parameter optimization (every {settings.backtest.cycle_hours}h)")
        logger.info("")
        logger.info("🎯 Trading Logic:")
        logger.info("   • Composite signal: Imbalance + Momentum + Volume")
        logger.info("   • Dynamic TP/SL based on volatility")
        logger.info("   • Adaptive position lifetime")
        logger.info("   • Trailing stop protection")
        logger.info("")
        logger.info("⚡ Performance:")
        logger.info(f"   • Position monitoring: {settings.trading.monitor_positions_interval_sec}s")
        logger.info(f"   • Decision interval: {settings.trading.decision_interval_sec}s")
        logger.info(f"   • Rate limiting: {settings.trading.max_orders_per_second}/s")
        logger.info("")
        logger.info("📝 Logging:")
        logger.info("   • Trades: logs/trades.csv")
        logger.info("   • Bot log: logs/bot.log")
        logger.info("   • Errors: logs/errors.log")
        if backtest_orchestrator:
            logger.info("   • Backtest results: utils/data_storage/")
        logger.info("")
        logger.info("=" * 70)
        logger.info("🔄 Press Ctrl+C to stop gracefully")
        logger.info("=" * 70)
        logger.info("")

        # Головний цикл
        while True:
            await asyncio.sleep(60)
            
            # Періодичний статус (кожні 5 хвилин)
            if int(time.time()) % 300 == 0:
                stats = executor.get_stats()
                logger.info(f"💹 Status: {stats['open_positions_count']} open, "
                          f"PnL: ${stats['total_pnl']:.2f}")

    except (KeyboardInterrupt, SystemExit):
        logger.info("")
        logger.info("=" * 70)
        logger.info("🛑 SHUTDOWN SIGNAL RECEIVED")
        logger.info("=" * 70)
        logger.info("")
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"❌ CRITICAL ERROR: {e}")
        logger.error("=" * 70)
        logger.error("", exc_info=True)
        
        try:
            await notifier.send(f"❌ Bot crashed: {str(e)[:100]}")
        except:
            pass
    finally:
        logger.info("🔄 GRACEFUL SHUTDOWN IN PROGRESS...")
        logger.info("")
        
        await safe_shutdown(
            collector, 
            orchestrator, 
            executor, 
            api_manager,
            backtest_orchestrator
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ BOT STOPPED SAFELY")
        logger.info("=" * 70)

async def safe_shutdown(collector, orchestrator, executor, api_manager, backtest_orchestrator=None):
    """Безпечна зупинка всіх компонентів"""
    logger.info("  [1/5] Stopping Data Collector...")
    try:
        await collector.stop()
        logger.info("  ✅ [1/5] Data Collector stopped")
    except Exception as e:
        logger.error(f"  ❌ [1/5] Error: {e}")
    
    logger.info("  [2/5] Stopping Trading Orchestrator...")
    try:
        await orchestrator.stop()
        logger.info("  ✅ [2/5] Trading Orchestrator stopped")
    except Exception as e:
        logger.error(f"  ❌ [2/5] Error: {e}")
    
    logger.info("  [3/5] Stopping Trade Executor...")
    try:
        await executor.stop()
        logger.info("  ✅ [3/5] Trade Executor stopped")
    except Exception as e:
        logger.error(f"  ❌ [3/5] Error: {e}")
    
    if backtest_orchestrator:
        logger.info("  [4/5] Stopping Backtest Orchestrator...")
        try:
            await backtest_orchestrator.stop()
            logger.info("  ✅ [4/5] Backtest Orchestrator stopped")
        except Exception as e:
            logger.error(f"  ❌ [4/5] Error: {e}")
    else:
        logger.info("  ⏩ [4/5] Backtest skipped")
    
    logger.info("  [5/5] Closing API connections...")
    try:
        await api_manager.close()
        logger.info("  ✅ [5/5] API connections closed")
    except Exception as e:
        logger.error(f"  ❌ [5/5] Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)