# config/settings.py
import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class SystemSettings(BaseSettings):
    """Системні налаштування для різних режимів"""
    rest_market_base: str = "https://api.bybit.com"
    rest_market_base_demo: str = "https://api-demo.bybit.com"
    ws_public_linear: str = "wss://stream.bybit.com/v5/public/linear"
    ws_public_linear_demo: str = "wss://stream-demo.bybit.com/v5/public/linear"
    ws_private: str = "wss://stream.bybit.com/v5/private"
    ws_private_demo: str = "wss://stream-demo.bybit.com/v5/private"

    def get_mode_info(self) -> Dict[str, Any]:
        """Повертає інформацію про поточний режим роботи"""
        from config.settings import settings
        mode = settings.trading.mode.upper()
        
        if mode == "DEMO":
            return {
                "mode": "DEMO (Paper Trading)",
                "ws_public": self.ws_public_linear_demo,
                "ws_private": self.ws_private_demo,
                "rest_api": self.rest_market_base_demo,
                "note": "Using demo environment with virtual funds"
            }
        else:
            return {
                "mode": "LIVE (Real Trading)",
                "ws_public": self.ws_public_linear,
                "ws_private": self.ws_private,
                "rest_api": self.rest_market_base,
                "note": "⚠️ REAL MONEY - Trading with actual funds"
            }

class SecretsSettings(BaseSettings):
    """API ключі та секрети"""
    bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET")
    
    demo_bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY_DEMO")
    demo_bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET_DEMO")
    
    live_bybit_api_key: str = Field(default="", alias="BYBIT_API_KEY_LIVE")
    live_bybit_api_secret: str = Field(default="", alias="BYBIT_API_SECRET_LIVE")
    
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True

class PairsSettings(BaseSettings):
    """Налаштування торгових пар"""
    trade_pairs: list = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT",
        "AAVEUSDT", "STRKUSDT"
    ]
    
    low_liquidity_pairs: list = ["HFTUSDT", "TRXUSDT"]
    excluded_pairs: list = ["HFTUSDT"]

class TradingSettings(BaseSettings):
    """Основні налаштування торгівлі"""
    mode: str = "DEMO"
    leverage: int = 10
    base_order_usdt: float = 0.0
    base_order_pct: float = 0.1
    start_balance_usdt: float = 0.0
    
    max_orders_per_second: int = 5
    max_orders_per_minute: int = 100
    max_reprice_attempts: int = 8
    
    entry_signal_min_strength: int = 3
    close_on_opposite_strength: int = 5
    
    decision_interval_sec: float = 2.0
    min_time_between_trades_sec: float = 15.0
    reopen_cooldown_sec: float = 10.0
    min_position_hold_time_sec: float = 30.0
    
    monitor_positions_interval_sec: float = 5.0
    enable_parallel_monitoring: bool = True
    monitoring_batch_size: int = 5
    
    reverse_signals: bool = False
    reverse_double_size: bool = False
    
    enable_aggressive_filtering: bool = True
    
    # 🆕 MTF налаштування для оркестратора
    enable_mtf_filter: bool = True
    mtf_convergence_threshold: float = 0.3
    min_mtf_timeframes_confirmed: int = 2

class RiskSettings(BaseSettings):
    """Налаштування ризик-менеджменту"""
    
    max_open_positions: int = 5
    max_position_notional_pct: float = 1.0
    
    # Адаптивний lifetime
    base_position_lifetime_minutes: int = 120
    enable_adaptive_lifetime: bool = True
    
    low_volatility_lifetime_multiplier: float = 1.5
    high_volatility_lifetime_multiplier: float = 0.7
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 2.0
    
    # =====================================================
    # 🐋 WHALE STRATEGY: Динамічне TP/SL (ОНОВЛЕНО)
    # =====================================================
    enable_dynamic_tpsl: bool = True
    
    min_sl_pct: float = 0.003       # 0.3% - тайтіший SL (було 0.5%)
    min_tp_pct: float = 0.004       # 0.4% - досяжний TP (було 1%)
    max_sl_pct: float = 0.008       # 0.8% - макс SL (було 3%)
    max_tp_pct: float = 0.012       # 1.2% - макс TP (було 6%)
    
    sl_vol_multiplier: float = 1.2  # зменшено з 1.5
    tp_vol_multiplier: float = 2.0  # зменшено з 3.0
    max_vol_used_pct: float = 5.0
    
    # Динамічне співвідношення TP/SL
    enable_dynamic_tpsl_ratio: bool = True
    tpsl_ratio_high_winrate: float = 1.5    # було 2.0 - менший, досяжніший TP
    tpsl_ratio_medium_winrate: float = 1.8  # було 2.5
    tpsl_ratio_low_winrate: float = 2.0     # було 3.0
    
    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.005  # 0.5% - раніше активуємо (було 1%)
    trailing_stop_distance_pct: float = 0.003    # 0.3% - тайтіший трейл (було 0.5%)
    
    position_history_size: int = 100
    min_history_for_adaptation: int = 20
    
    @property
    def position_lifetime_minutes(self) -> int:
        return self.base_position_lifetime_minutes
    
    @property
    def max_position_lifetime_sec(self) -> int:
        return self.base_position_lifetime_minutes * 60

class ExecutionSettings(BaseSettings):
    """Налаштування виконання ордерів"""
    poll_interval_sec: float = 0.5
    max_wait_sec: float = 60.0
    reprice_every_sec: float = 3.0
    reprice_step_bps: float = 5.0
    passive_improve_bps: float = 2.0
    
    require_full_fill: bool = False
    min_partial_pct: float = 0.8
    
    fallback_mode: str = "market"
    fallback_after_sec: float = 30.0
    cancel_before_fallback: bool = True

class WebSocketSettings(BaseSettings):
    """Налаштування WebSocket"""
    subscription_depth: int = 50
    ping_interval: float = 20.0
    reconnect_delay_seconds: float = 5.0
    data_retention_seconds: int = 300
    
    enable_private_ws: bool = True
    private_ws_heartbeat_interval: float = 20.0
    private_ws_reconnect_attempts: int = 5

class APISettings(BaseSettings):
    """Налаштування API"""
    retry_attempts: int = 3
    retry_delay: float = 1.0
    validate_time_diff_sec: int = 5
    instrument_cache_ttl: int = 3600
    ticker_cache_ttl: int = 5

class LoggingSettings(BaseSettings):
    """Налаштування логування"""
    mode: str = "work"
    
    console_level_debug: str = "DEBUG"
    file_level_debug: str = "DEBUG"
    console_level_work: str = "INFO"
    file_level_work: str = "DEBUG"
    
    log_dir: Path = Path("logs")
    common_log: Path = Path("logs/bot.log")
    errors_log: Path = Path("logs/errors.log")
    trades_log: Path = Path("logs/trades.csv")

class ImbalanceSettings(BaseSettings):
    """Налаштування аналізу дисбалансу"""
    depth_limit_for_calc: int = 50
    min_volume_epsilon: float = 1e-9
    large_order_side_percent: float = 0.05
    
    enable_adaptive_large_orders: bool = True
    large_order_zscore_threshold: float = 2.0
    large_order_lookback_periods: int = 100
    large_order_min_samples: int = 20
    
    large_order_min_notional_abs: float = 500.0
    
    spoof_lifetime_ms: int = 3000
    
    enable_spoof_filter: bool = True
    smoothing_factor: float = 0.3
    universal_imbalance_cap: float = 100.0
    
    enable_historical_imbalance: bool = True
    historical_window_minutes: int = 15
    historical_samples: int = 10
    long_term_smoothing: float = 0.1
    
    # 🆕 MTF налаштування
    enable_multi_timeframe_analysis: bool = True

class VolumeSettings(BaseSettings):
    """Налаштування аналізу обсягів"""
    short_window_sec: int = 30
    long_window_sec: int = 300
    default_min_trades: int = 5
    vwap_min_volume: float = 100.0
    
    enable_multi_timeframe_momentum: bool = True
    momentum_windows: list = [15, 30, 60, 120]
    momentum_weights: list = [0.4, 0.3, 0.2, 0.1]
    
    enable_adaptive_volume_analysis: bool = True
    volume_zscore_threshold_high: float = 2.0
    volume_zscore_threshold_very_high: float = 3.0
    volume_zscore_threshold_low: float = -1.0
    volume_lookback_periods: int = 96
    volume_min_samples: int = 20
    
    enable_percentile_method: bool = True
    volume_percentile_very_high: float = 95.0
    volume_percentile_high: float = 75.0
    volume_percentile_low: float = 25.0
    
    enable_ema_volume_analysis: bool = True
    ema_fast_period: int = 20
    ema_slow_period: int = 100
    ema_ratio_high: float = 2.0
    ema_ratio_very_high: float = 3.0
    
    enable_trade_frequency_analysis: bool = True
    frequency_baseline_window_sec: int = 300
    frequency_very_high_multiplier: float = 5.0
    frequency_high_multiplier: float = 2.5
    frequency_very_low_multiplier: float = 0.3
    
    enable_volume_confirmation: bool = True
    volume_baseline_window_sec: int = 86400
    volume_confirmation_zscore: float = 1.2
    volume_weak_zscore: float = -0.5
    
    enable_large_order_tracker: bool = True
    large_order_lookback_sec: int = 600
    large_order_strong_threshold: int = 3
    
    # 🆕 MTF налаштування
    enable_multi_timeframe_analysis: bool = True

class AdaptiveSettings(BaseSettings):
    """Налаштування адаптивних механізмів"""
    enable_adaptive_windows: bool = True
    base_volatility_threshold: float = 1.0
    
    low_volatility_multiplier: float = 1.5
    high_volatility_multiplier: float = 0.7
    
    max_window_expansion: float = 2.0
    min_window_reduction: float = 0.5

class SignalSettings(BaseSettings):
    """Налаштування генерації сигналів"""
    
    # =====================================================
    # 🐋 WHALE STRATEGY: ОНОВЛЕНІ ВАГИ (більше на momentum і large orders)
    # =====================================================
    weight_momentum: float = 0.25           # підвищено з 0.20 - momentum важливіший
    weight_ohara_bayesian: float = 0.10     # знижено з 0.12
    weight_ohara_large_orders: float = 0.12 # підвищено з 0.08 - слідуємо за китами! 
    weight_imbalance: float = 0.38          # знижено з 0.45
    weight_ohara_frequency: float = 0.07    # було 0.075
    weight_ohara_volume_confirm: float = 0.08  # підвищено - об'єм підтверджує
    spike_bonus: float = 0.1
    
    smoothing_alpha: float = 0.75
    hold_threshold: float = 0.15            # підвищено з 0.12 - менше шуму
    
    # Composite score thresholds - підвищені пороги
    composite_thresholds: dict = {
        "strength_1": 0.20,   # було 0.15
        "strength_2": 0.35,   # було 0.30
        "strength_3": 0.45,   # було 0.40 - головний поріг входу
        "strength_4": 0.60,   # було 0.65
        "strength_5": 0.75    # було 0.80
    }
    
    min_strength_for_action: int = 3
    
    # =====================================================
    # 🐋 WHALE STRATEGY: АДАПТИВНІ ПОРОГИ (підвищені)
    # =====================================================
    enable_adaptive_threshold: bool = True
    base_threshold: float = 0.45            # підвищено з 0.40 - чекаємо сильніший сигнал
    min_threshold: float = 0.38             # підвищено з 0.32
    max_threshold: float = 0.55             # підвищено з 0.50
    
    high_volatility_threshold_reduction: float = 0.03  # зменшено з 0.05
    low_volatility_threshold_increase: float = 0.05    # підвищено з 0.03
    volatility_high_level: float = 2.0
    volatility_low_level: float = 0.5
    
    high_liquidity_threshold_reduction: float = 0.02   # зменшено з 0.03
    low_liquidity_threshold_increase: float = 0.07     # підвищено з 0.05
    
    # =====================================================
    # 🐋 WHALE STRATEGY: ВИМКНЕНО EARLY ENTRY (причина збитків!)
    # =====================================================
    early_entry_enabled: bool = False       # 🚫 ВИМКНЕНО!  було True
    early_entry_momentum_threshold: float = 40.0
    early_entry_volatility_threshold: float = 0.3
    early_entry_ohara_threshold: int = 6
    early_entry_imbalance_threshold: float = 35.0
    early_entry_threshold_multiplier: float = 0.72
    
    # =====================================================
    # 🐋 WHALE STRATEGY: ПІДТВЕРДЖЕННЯ РУХУ (нові параметри)
    # =====================================================
    # Мінімальний momentum для входу - чекаємо підтвердження
    min_momentum_for_entry: float = 45.0    # 🆕 не входимо якщо momentum < 45%
    max_momentum_for_entry: float = 88.0    # 🆕 не входимо якщо > 88% (занадто пізно)
    
    # Мінімальний imbalance для входу
    min_imbalance_for_entry: float = 8.0   # 🆕 не входимо якщо imbalance < 8%
    
    # =====================================================
    # 🐋 WHALE STRATEGY: LATE ENTRY (тепер основний режим)
    # =====================================================
    late_entry_momentum_threshold: float = 92.0  # підвищено з 85.0
    late_entry_allow_strong_trend: bool = True
    late_entry_min_ohara_for_override: int = 7
    late_entry_position_size_reduction: float = 0.5
    late_entry_high_momentum_threshold: float = 80.0  # підвищено з 70.0
    
    # =====================================================
    # 🐋 WHALE STRATEGY: LARGE ORDER REQUIREMENTS (підвищені)
    # =====================================================
    large_order_count_bonus_threshold: int = 3
    large_order_count_bonus_per_order: float = 0.04   # підвищено з 0.03
    large_order_count_bonus_max: float = 0.20         # підвищено з 0.15
    
    # Мінімум великих ордерів для входу
    min_large_orders_for_entry: int = 1     # 🆕 потрібно мінімум 1 великий ордер
    
    # =====================================================
    # 🐋 WHALE STRATEGY: O'HARA SCORE (підвищені вимоги)
    # =====================================================
    ohara_strong_score_threshold: int = 7   # знижено з 8 для більшої гнучкості
    ohara_threshold_reduction: float = 0.04 # підвищено з 0.03
    min_ohara_for_entry: int = 4            # 🆕 мінімальний O'Hara score для входу
    
    # =====================================================
    # 🆕 MULTI-TIMEFRAME SETTINGS (MTF)
    # =====================================================
    enable_mtf_filter: bool = True
    mtf_convergence_threshold: float = 0.3
    min_mtf_timeframes_confirmed: int = 2
    mtf_confirmation_boost: float = 1.2
    mtf_weight_1min: float = 0.4
    mtf_weight_5min: float = 0.35
    mtf_weight_30min: float = 0.25
    mtf_require_confirmation_for_entry: bool = True
    mtf_allow_override_on_strong_signal: bool = True
    mtf_override_strength_threshold: int = 4
    enable_multi_timeframe_analysis: bool = True
    
    # Contradictory orders
    allow_override_contradictory_orders: bool = False  # 🚫 ВИМКНЕНО - не йдемо проти китів
    override_imbalance_threshold: float = 45.0         # підвищено з 40.0
    override_momentum_threshold: float = 55.0          # підвищено з 50.0
    strong_cooldown_level: int = 3
    cooldown_seconds: float = 180.0
    
    allow_reversal_during_cooldown: bool = True
    require_signal_consistency: bool = True
    max_imbalance_contradiction: float = 15.0  # знижено з 20.0 - суворіший фільтр
    
    enable_volume_validation: bool = True
    min_short_volume_for_signal: float = 1500.0  # підвищено з 1000.0
    min_trades_for_signal: int = 15              # підвищено з 10
    
    volatility_filter_threshold: float = 0.20    # знижено з 0.25 - суворіший фільтр
    
    # =====================================================
    # 🐋 WHALE STRATEGY: EXHAUSTION FILTER (підсилений)
    # =====================================================
    enable_exhaustion_filter: bool = True
    max_momentum_for_entry: float = 88.0         # знижено з 80.0 (тепер = max_momentum)
    min_imbalance_for_high_momentum: float = 20.0  # підвищено з 15.0

class SpreadSettings(BaseSettings):
    """O'HARA METHOD 7: Spread as Risk Measure"""
    enable_spread_monitor: bool = True
    
    max_spread_threshold_bps: float = 15.0       # знижено з 20.0 - суворіший контроль
    high_risk_spread_multiplier: float = 2.5     # знижено з 3.0
    very_high_risk_spread_multiplier: float = 4.0  # знижено з 5.0
    
    spread_history_size: int = 100
    spread_baseline_window_sec: int = 3600
    
    avoid_trading_on_very_high_spread: bool = True
    reduce_size_on_high_spread: bool = True
    high_spread_size_reduction_pct: float = 0.6  # підвищено з 0.5

class OHaraSettings(BaseSettings):
    """O'HARA METHODS: Comprehensive Settings"""
    
    enable_bayesian_updating: bool = True
    bayesian_update_step: float = 0.05
    bayesian_bullish_threshold: float = 0.65
    bayesian_bearish_threshold: float = 0.35
    bayesian_decay_factor: float = 0.98
    
    # =====================================================
    # 🐋 WHALE STRATEGY: LARGE ORDERS (підвищені вимоги)
    # =====================================================
    large_order_min_count_strong: int = 4   # підвищено з 3 - потрібно більше підтверджень
    large_order_min_count_medium: int = 2
    large_order_net_threshold: int = 3      # підвищено з 2
    
    enable_combined_ohara_score: bool = True
    min_ohara_score_for_trade: int = 5      # підвищено з 4 - суворіший поріг
    strong_ohara_score_threshold: int = 7   # знижено з 8 для гнучкості

class MultiTimeframeSettings(BaseSettings):
    """🆕 Налаштування багатотаймфреймового аналізу"""
    enable_multi_timeframe_analysis: bool = True
    timeframes_seconds: list = [60, 300, 1800]  # 1, 5, 30 хвилин
    timeframe_weights: list = [0.4, 0.35, 0.25]  # Ваги для комбінації
    
    # Конвергенція
    convergence_threshold: float = 0.3
    min_confirmed_timeframes: int = 2
    
    # Фільтри
    require_mtf_confirmation_for_entry: bool = True
    mtf_confirmation_boost: float = 1.2  # Бонус до сили сигналу
    
    # Lifetime корекція
    enable_mtf_lifetime_adjustment: bool = True
    high_convergence_lifetime_multiplier: float = 1.2
    low_convergence_lifetime_multiplier: float = 0.8
    
    # Таймфреймові ваги для різних аналізів
    mtf_weight_imbalance: list = [0.4, 0.35, 0.25]
    mtf_weight_momentum: list = [0.45, 0.35, 0.20]
    mtf_weight_volatility: list = [0.3, 0.35, 0.35]

class TechnicalAnalysisSettings(BaseSettings):
    """
    Technical Analysis Settings
    Based on Murphy, Bulkowski, Nison, Bigalow
    """
    # Enable TA mode (replaces orderbook imbalance analysis)
    enable_ta_mode: bool = True
    
    # Candle aggregation
    candle_timeframe_seconds: int = 3600  # 1 hour candles
    min_candles_required: int = 210  # Need 200+ for MA200
    
    # Minimum signal requirements (Bulkowski/Bigalow)
    min_signal_strength: int = 3  # 1-5 scale
    min_confidence: float = 65.0  # Minimum confidence %
    
    # Pattern detection
    enable_chart_patterns: bool = True
    enable_candlestick_patterns: bool = True
    
    # Technical indicators (Murphy)
    enable_trend_analysis: bool = True  # 200-day MA
    enable_rsi: bool = True  # RSI for overbought/oversold
    enable_macd: bool = True  # MACD for trend confirmation
    enable_stochastic: bool = True  # Stochastic oscillator
    enable_bollinger_bands: bool = True  # Volatility
    
    # Confirmation requirements
    require_trend_confirmation: bool = True  # Must align with 200 MA
    require_volume_confirmation: bool = True  # Patterns must have volume
    require_indicator_confirmation: bool = False  # RSI + MACD (optional)
    
    # Pattern-specific settings
    double_bottom_min_confidence: float = 60.0
    head_shoulders_min_confidence: float = 60.0
    triangle_min_confidence: float = 60.0
    hammer_min_confidence: float = 60.0
    engulfing_min_confidence: float = 65.0
    morning_star_min_confidence: float = 70.0
    
    # Risk management (Bigalow)
    risk_per_trade_pct: float = 0.015  # 1.5% per trade
    max_portfolio_risk_pct: float = 0.08  # 8% max
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1
    target_risk_reward_ratio: float = 2.0  # Target 2:1
    
    # Position sizing
    base_position_size_pct: float = 0.02  # 2% per trade
    strong_signal_multiplier: float = 1.5  # Increase for strength 5
    high_volatility_reduction: float = 0.7  # Reduce 30% in high vol
    
    # Leverage (2-5x per problem statement)
    min_leverage: float = 2.0
    max_leverage: float = 5.0
    default_leverage: float = 3.0
    
    # Stop-loss (Bigalow: 1-2% below pattern)
    stop_loss_pct_min: float = 0.01  # 1%
    stop_loss_pct_max: float = 0.02  # 2%
    stop_loss_atr_multiplier: float = 1.5  # 1.5x ATR
    
    # Take-profit (Bigalow: 2:1 reward-risk)
    use_pattern_projection: bool = True  # Use pattern height
    take_profit_atr_multiplier: float = 3.0  # 2:1 from 1.5 ATR SL
    
    # Holding periods
    max_holding_period_hours: int = 72  # 3 days max
    min_holding_period_hours: int = 1  # 1 hour min
    
    # Backtesting
    backtest_enabled: bool = False
    backtest_initial_balance: float = 10000.0
    backtest_commission_rate: float = 0.0006  # 0.06% Bybit taker

class Settings(BaseSettings):
    """Головний клас налаштувань"""
    system: SystemSettings = SystemSettings()
    secrets: SecretsSettings = SecretsSettings()
    pairs: PairsSettings = PairsSettings()
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    execution: ExecutionSettings = ExecutionSettings()
    websocket: WebSocketSettings = WebSocketSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    imbalance: ImbalanceSettings = ImbalanceSettings()
    volume: VolumeSettings = VolumeSettings()
    adaptive: AdaptiveSettings = AdaptiveSettings()
    signals: SignalSettings = SignalSettings()
    spread: SpreadSettings = SpreadSettings()
    ohara: OHaraSettings = OHaraSettings()
    multi_timeframe: MultiTimeframeSettings = MultiTimeframeSettings()
    technical_analysis: TechnicalAnalysisSettings = TechnicalAnalysisSettings()  # 🆕 TA settings

settings = Settings()