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
        env_file = "config/.env"  # 🔑 Шлях до вашого .env файлу
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True  # Дозволяє використовувати alias

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
    
    # 🆕 АДАПТИВНИЙ МОНІТОРИНГ
    monitor_positions_interval_sec: float = 5.0
    enable_parallel_monitoring: bool = True
    monitoring_batch_size: int = 5
    
    reverse_signals: bool = True
    reverse_double_size: bool = False
    
    enable_aggressive_filtering: bool = True

class RiskSettings(BaseSettings):
    """🆕 ОНОВЛЕНІ налаштування ризик-менеджменту"""
    
    max_open_positions: int = 5
    max_position_notional_pct: float = 1.0
    
    # 🆕 АДАПТИВНИЙ LIFETIME
    base_position_lifetime_minutes: int = 30
    enable_adaptive_lifetime: bool = True
    
    low_volatility_lifetime_multiplier: float = 1.5
    high_volatility_lifetime_multiplier: float = 0.7
    volatility_threshold_low: float = 0.5
    volatility_threshold_high: float = 2.0
    
    # 🆕 ДИНАМІЧНЕ TP/SL
    enable_dynamic_tpsl: bool = True
    
    min_sl_pct: float = 0.005
    min_tp_pct: float = 0.01
    max_sl_pct: float = 0.03
    max_tp_pct: float = 0.06
    
    sl_vol_multiplier: float = 1.5
    tp_vol_multiplier: float = 3.0
    max_vol_used_pct: float = 5.0
    
    # 🆕 Динамічне співвідношення TP/SL
    enable_dynamic_tpsl_ratio: bool = True
    tpsl_ratio_high_winrate: float = 2.0
    tpsl_ratio_medium_winrate: float = 2.5
    tpsl_ratio_low_winrate: float = 3.0
    
    # 🆕 TRAILING STOP
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.01
    trailing_stop_distance_pct: float = 0.005
    
    position_history_size: int = 100
    min_history_for_adaptation: int = 20
    
    @property
    def position_lifetime_minutes(self) -> int:
        """Для сумісності"""
        return self.base_position_lifetime_minutes
    
    @property
    def max_position_lifetime_sec(self) -> int:
        """Для сумісності"""
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
    weight_imbalance: float = 0.4
    weight_momentum: float = 0.4
    spike_bonus: float = 0.1
    
    smoothing_alpha: float = 0.4
    hold_threshold: float = 0.12
    
    composite_thresholds: dict = {
        "strength_1": 0.15,
        "strength_2": 0.25,
        "strength_3": 0.40,
        "strength_4": 0.60,
        "strength_5": 0.75
    }
    
    min_strength_for_action: int = 3
    strong_cooldown_level: int = 3
    cooldown_seconds: float = 180.0
    
    allow_reversal_during_cooldown: bool = True
    require_signal_consistency: bool = True
    max_imbalance_contradiction: float = 30.0
    
    enable_volume_validation: bool = True
    min_short_volume_for_signal: float = 1000.0
    min_trades_for_signal: int = 10
    
    volatility_filter_threshold: float = 0.25

class SpreadSettings(BaseSettings):
    """Налаштування spread"""
    max_spread_threshold_bps: float = 20.0

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

settings = Settings()