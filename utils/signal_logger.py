# utils/signal_logger.py
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

class SignalLogger:
    """Логує всі сигнали (прийняті та відхилені) в CSV файл для аналізу"""
    
    def __init__(self, log_path: str = "logs/signals.csv"):
        self.log_path = Path(log_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Створює файл та заголовки якщо файл не існує"""
        if not self.log_path.exists():
            # Створюємо директорію якщо потрібно
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Створюємо файл з заголовками
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "symbol",
                    "action",
                    "strength",
                    "composite",
                    "ema",
                    "imbalance",
                    "momentum",
                    "bayesian",
                    "large_orders",
                    "frequency",
                    "vol_confirm",
                    "ohara_score",
                    "mtf_convergence",  # 🆕 Додано нове поле
                    "mtf_score",        # 🆕 Додано нове поле
                    "reason",
                    "accepted"
                ])
    
    def log_signal(
        self,
        symbol: str,
        action: str,
        strength: int,
        composite: float,
        ema: float,
        imbalance: float,
        momentum: float,
        bayesian: str,
        large_orders: str,
        frequency: str,
        vol_confirm: str,
        ohara_score: int,
        mtf_convergence: float = 0.0,  # 🆕 Додано параметр
        mtf_score: float = 0.0,        # 🆕 Додано параметр
        reason: str = "",
        accepted: bool = False
    ):
        """Логує один сигнал в CSV файл"""
        try:
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol,
                    action,
                    strength,
                    f"{composite:.3f}",
                    f"{ema:.3f}",
                    f"{imbalance:.1f}",
                    f"{momentum:.1f}",
                    bayesian,
                    large_orders,
                    frequency,
                    vol_confirm,
                    ohara_score,
                    f"{mtf_convergence:.2f}",  # 🆕 Додано значення
                    f"{mtf_score:.3f}",        # 🆕 Додано значення
                    reason,
                    "YES" if accepted else "NO"
                ])
        except Exception as e:
            print(f"❌ [SIGNAL_LOGGER] Error logging signal: {e}")

# Глобальний інстанс для використання
signal_logger = SignalLogger()