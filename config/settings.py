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
        "HFTUSDT", "AAVEUSDT", "STRKUSDT"
    ]

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
    
    entry_signal_min_strength: int = 4
    close_on_opposite_strength: int = 5
    
    decision_interval_sec: float = 2.0
    min_time_between_trades_sec: float = 15.0
    reopen_cooldown_sec: float = 10.0
    min_position_hold_time_sec: float = 30.0
    
    monitor_positions_interval_sec: float = 5.0
    enable_parallel_monitoring: bool = True
    monitoring_batch_size: int = 5
    
    reverse_signals: bool = False  # ✅ ВИПРАВЛЕНО: використовуємо правильну логіку імбалансу
    reverse_double_size: bool = False
    
    enable_aggressive_filtering: bool = True

class RiskSettings(BaseSettings):
    """Налаштування ризик-менеджменту"""
    
    max_open_positions: int = 5
    max_position_notional_pct: float = 1.0
    
    # Адаптивний lifetime
    base_position_lifetime_minutes: int = 30
    enable_adaptive_lifetime: bool = True
    
    low_volatility_lifetime_multiplier: float = 1.5
    high_volatility_lifetime_multiplier: float = 0.7
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 2.0
    
    # Динамічне TP/SL
    enable_dynamic_tpsl: bool = True
    
    min_sl_pct: float = 0.005
    min_tp_pct: float = 0.01
    max_sl_pct: float = 0.03
    max_tp_pct: float = 0.06
    
    sl_vol_multiplier: float = 1.5
    tp_vol_multiplier: float = 3.0
    max_vol_used_pct: float = 5.0
    
    # Динамічне співвідношення TP/SL
    enable_dynamic_tpsl_ratio: bool = True
    tpsl_ratio_high_winrate: float = 2.0
    tpsl_ratio_medium_winrate: float = 2.5
    tpsl_ratio_low_winrate: float = 3.0
    
    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.01
    trailing_stop_distance_pct: float = 0.005
    
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
    large_order_min_notional_abs: float = 500.0
    spoof_lifetime_ms: int = 3000
    
    enable_spoof_filter: bool = True
    smoothing_factor: float = 0.3
    universal_imbalance_cap: float = 100.0
    
    enable_historical_imbalance: bool = True
    historical_window_minutes: int = 15
    historical_samples: int = 10
    long_term_smoothing: float = 0.1

class VolumeSettings(BaseSettings):
    """Налаштування аналізу обсягів"""
    short_window_sec: int = 30
    long_window_sec: int = 300
    default_min_trades: int = 5
    vwap_min_volume: float = 100.0
    
    enable_multi_timeframe_momentum: bool = True
    momentum_windows: list = [15, 30, 60, 120]
    momentum_weights: list = [0.4, 0.3, 0.2, 0.1]
    
    # 🆕 O'HARA METHOD 3: Trade Frequency Analysis
    enable_trade_frequency_analysis: bool = True
    frequency_baseline_window_sec: int = 300  # 5 хвилин для baseline
    frequency_very_high_multiplier: float = 5.0  # >5x від baseline = VERY_HIGH
    frequency_high_multiplier: float = 2.5  # >2.5x від baseline = HIGH
    frequency_very_low_multiplier: float = 0.3  # <0.3x від baseline = VERY_LOW
    
    # 🆕 O'HARA METHOD 5: Volume Confirmation
    enable_volume_confirmation: bool = True
    volume_baseline_window_sec: int = 86400  # 24 години для baseline
    volume_confirmation_multiplier: float = 2.0  # Обсяг повинен бути >2x від середнього
    volume_weak_threshold: float = 0.8  # <0.8x від середнього = слабкий рух
    
    # 🆕 O'HARA METHOD 2: Large Order Tracking (Enhanced)
    enable_large_order_tracker: bool = True
    large_order_lookback_sec: int = 600  # 10 хвилин історії
    large_order_significance_multiplier: float = 5.0  # >5x від середнього = великий
    large_order_strong_threshold: int = 3  # 3+ великих ордера = сильний сигнал

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
    # ✅ ВИПРАВЛЕНО: Більша вага на імбаланс (leading indicator)
    weight_imbalance: float = 0.40           # Було 0.30
    weight_momentum: float = 0.20            # Було 0.25
    weight_ohara_bayesian: float = 0.12      # Було 0.15
    weight_ohara_large_orders: float = 0.15
    weight_ohara_frequency: float = 0.065    # Було 0.075
    weight_ohara_volume_confirm: float = 0.065  # Було 0.075
    spike_bonus: float = 0.1
    
    smoothing_alpha: float = 0.4
    hold_threshold: float = 0.12
    
    # ✅ ВИПРАВЛЕНО: Вищі пороги для раніших входів
    composite_thresholds: dict = {
        "strength_1": 0.15,
        "strength_2": 0.30,  # Було 0.25
        "strength_3": 0.45,  # Було 0.40
        "strength_4": 0.65,  # Було 0.60 ← Важливо! 
        "strength_5": 0.80   # Було 0.75
    }
    
    min_strength_for_action: int = 3
    strong_cooldown_level: int = 3
    cooldown_seconds: float = 180.0
    
    allow_reversal_during_cooldown: bool = True
    require_signal_consistency: bool = True
    max_imbalance_contradiction: float = 20.0  # ✅ ВИПРАВЛЕНО: Було 30.0
    
    enable_volume_validation: bool = True
    min_short_volume_for_signal: float = 1000.0
    min_trades_for_signal: int = 10
    
    volatility_filter_threshold: float = 0.25
    
    # 🆕 ДОДАНО: Фільтри для запобігання пізньому входу
    enable_exhaustion_filter: bool = True
    max_momentum_for_entry: float = 70.0  # Не входимо якщо momentum > 70%
    min_imbalance_for_high_momentum: float = 15.0  # При mom>60 треба imb>15

class SpreadSettings(BaseSettings):
    """🆕 O'HARA METHOD 7: Spread as Risk Measure"""
    enable_spread_monitor: bool = True
    
    # Базові пороги spread (в basis points)
    max_spread_threshold_bps: float = 20.0
    high_risk_spread_multiplier: float = 3.0  # >3x від середнього = HIGH_RISK
    very_high_risk_spread_multiplier: float = 5.0  # >5x = VERY_HIGH_RISK
    
    # Історія для розрахунку baseline
    spread_history_size: int = 100
    spread_baseline_window_sec: int = 3600  # 1 година
    
    # Фільтрація торгівлі
    avoid_trading_on_very_high_spread: bool = True
    reduce_size_on_high_spread: bool = True
    high_spread_size_reduction_pct: float = 0.5  # Зменшити на 50%

class OHaraSettings(BaseSettings):
    """🆕 O'HARA METHODS: Comprehensive Settings"""
    
    # METHOD 1: Bayesian Price Updating
    enable_bayesian_updating: bool = True
    bayesian_update_step: float = 0.05  # Крок оновлення ймовірності
    bayesian_bullish_threshold: float = 0.65  # >65% = BULLISH
    bayesian_bearish_threshold: float = 0.35  # <35% = BEARISH
    bayesian_decay_factor: float = 0.98  # Згасання для повернення до 0.5
    
    # METHOD 2: Large Order Detection (Enhanced)
    large_order_min_count_strong: int = 3  # 3+ великих = сильний сигнал
    large_order_min_count_medium: int = 2  # 2 великих = середній сигнал
    large_order_net_threshold: int = 2  # Різниця buy/sell >= 2
    
    # METHOD 3: Trade Frequency (див.VolumeSettings)
    # METHOD 4: Buy/Sell Imbalance (вже в ImbalanceSettings)
    # METHOD 5: Volume Confirmation (див.VolumeSettings)
    
    # METHOD 7: Spread Risk (див.SpreadSettings)
    
    # Combined Signal Scoring
    enable_combined_ohara_score: bool = True
    min_ohara_score_for_trade: int = 5  # Мінімум 5 балів з усіх методів
    strong_ohara_score_threshold: int = 8  # 8+ балів = дуже сильний сигнал

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
    ohara: OHaraSettings = OHaraSettings()  # 🆕 O'Hara settings

settings = Settings()