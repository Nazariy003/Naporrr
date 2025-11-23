# utils/backtest/data_collector.py
import time
import asyncio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from config.settings import settings
from utils.logger import logger

class DataCollector:
    """
    🎯 Збір і ротація даних з бюджетом 10 ГБ
    
    Рівні зберігання:
    - RAW (7 днів): orderbook snapshots (5s) + trades + signals
    - AGGREGATED (30 днів): 1-min bars + aggregated signals
    - METADATA (90 днів): тільки результати сигналів
    """
    
    def __init__(self, storage_path: str = "utils/data_storage"):
        self.storage_path = Path(storage_path)
        self.raw_path = self.storage_path / "raw"
        self.agg_path = self.storage_path / "aggregated"
        self.meta_path = self.storage_path / "metadata"
        
        # Створення директорій
        for path in [self.raw_path, self.agg_path, self.meta_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Буфери для батчового запису
        self.buffers = {
            'orderbook': {},
            'trades': {},
            'signals': {}
        }
        
        self.buffer_size = 100  # Записуємо кожні 100 записів
        self.last_flush = time.time()
        
    async def start(self, storage, signal_generator):
        """Запуск збору даних"""
        logger.info("🎬 [DATA_COLLECTOR] Starting...")
        
        # Підписка на події
        storage.add_position_callback(self._on_position_update)
        
        # Циклічні задачі
        asyncio.create_task(self._snapshot_loop(storage))
        asyncio.create_task(self._trades_loop(storage))
        asyncio.create_task(self._signals_loop(signal_generator))
        asyncio.create_task(self._flush_loop())
        asyncio.create_task(self._rotation_loop())
        
        logger.info("✅ [DATA_COLLECTOR] Started")
    
    async def _snapshot_loop(self, storage):
        """Знімки orderbook кожні 5 секунд"""
        while True:
            try:
                await asyncio.sleep(5)
                
                for symbol in settings.pairs.trade_pairs:
                    ob = storage.get_order_book(symbol)
                    if not ob:
                        continue
                    
                    # Зберігаємо тільки топ-10 рівнів для економії місця
                    snapshot = {
                        'timestamp': ob.ts,
                        'symbol': symbol,
                        'best_bid': ob.best_bid,
                        'best_ask': ob.best_ask,
                        'bid_levels': [(lvl.price, lvl.size) for lvl in ob.bids[:10]],
                        'ask_levels': [(lvl.price, lvl.size) for lvl in ob.asks[:10]],
                    }
                    
                    self._add_to_buffer('orderbook', symbol, snapshot)
                    
            except Exception as e:
                logger.error(f"❌ [SNAPSHOT_LOOP] Error: {e}")
    
    async def _trades_loop(self, storage):
        """Збір trades кожні 10 секунд"""
        while True:
            try:
                await asyncio.sleep(10)
                
                for symbol in settings.pairs.trade_pairs:
                    trades = storage.get_trades(symbol)
                    if not trades:
                        continue
                    
                    for trade in trades:
                        trade_data = {
                            'timestamp': trade.ts,
                            'symbol': symbol,
                            'price': trade.price,
                            'size': trade.size,
                            'side': trade.side,
                            'is_aggressive': trade.is_aggressive
                        }
                        self._add_to_buffer('trades', symbol, trade_data)
                        
            except Exception as e:
                logger.error(f"❌ [TRADES_LOOP] Error: {e}")
    
    async def _signals_loop(self, signal_generator):
        """Збір сигналів кожні 2 секунди"""
        while True:
            try:
                await asyncio.sleep(2)
                
                # Зберігаємо metadata сигналів для replay
                current_time = time.time()
                
                for symbol in settings.pairs.trade_pairs:
                    # Тут має бути логіка отримання поточного сигналу
                    # Наразі placeholder
                    signal_data = {
                        'timestamp': current_time,
                        'symbol': symbol,
                        'signal': 'HOLD',  # BUY/SELL/HOLD
                        'strength': 0,
                        'composite': 0.0,
                        'imbalance': 0.0,
                        'momentum': 0.0,
                        'volatility': 0.0,
                        # Додаткові параметри для replay
                        'settings_snapshot': {
                            'weight_imbalance': settings.signals.weight_imbalance,
                            'weight_momentum': settings.signals.weight_momentum,
                            'hold_threshold': settings.signals.hold_threshold,
                        }
                    }
                    self._add_to_buffer('signals', symbol, signal_data)
                    
            except Exception as e:
                logger.error(f"❌ [SIGNALS_LOOP] Error: {e}")
    
    def _add_to_buffer(self, buffer_type: str, symbol: str, data: Dict):
        """Додавання до буфера"""
        key = f"{symbol}_{buffer_type}"
        if key not in self.buffers[buffer_type]:
            self.buffers[buffer_type][key] = []
        
        self.buffers[buffer_type][key].append(data)
        
        # Flush якщо досягли розміру буфера
        if len(self.buffers[buffer_type][key]) >= self.buffer_size:
            self._flush_buffer(buffer_type, symbol)
    
    async def _flush_loop(self):
        """Періодичний flush буферів (кожні 60 сек)"""
        while True:
            try:
                await asyncio.sleep(60)
                
                current_time = time.time()
                if current_time - self.last_flush >= 60:
                    self._flush_all_buffers()
                    self.last_flush = current_time
                    
            except Exception as e:
                logger.error(f"❌ [FLUSH_LOOP] Error: {e}")
    
    def _flush_buffer(self, buffer_type: str, symbol: str):
        """Запис буфера в Parquet"""
        key = f"{symbol}_{buffer_type}"
        
        if key not in self.buffers[buffer_type] or not self.buffers[buffer_type][key]:
            return
        
        try:
            # Конвертуємо в DataFrame
            df = pd.DataFrame(self.buffers[buffer_type][key])
            
            # Шлях до файлу
            today = datetime.utcnow().strftime("%Y-%m-%d")
            file_path = self.raw_path / today / f"{symbol}_{buffer_type}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append до існуючого файлу або створення нового
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # Запис з компресією
            df.to_parquet(
                file_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            # Очищення буфера
            self.buffers[buffer_type][key] = []
            
            logger.debug(f"💾 [FLUSH] {buffer_type}/{symbol}: {len(df)} records")
            
        except Exception as e:
            logger.error(f"❌ [FLUSH] {buffer_type}/{symbol}: {e}")
    
    def _flush_all_buffers(self):
        """Flush всіх буферів"""
        for buffer_type in ['orderbook', 'trades', 'signals']:
            for symbol in settings.pairs.trade_pairs:
                self._flush_buffer(buffer_type, symbol)
    
    async def _rotation_loop(self):
        """Ротація даних (кожні 24 год)"""
        while True:
            try:
                await asyncio.sleep(86400)  # 24 години
                
                logger.info("🔄 [ROTATION] Starting data rotation...")
                
                # 1. Видалення старих RAW даних (> 7 днів)
                self._cleanup_raw_data(days=7)
                
                # 2. Агрегація старих RAW в AGGREGATED (7-30 днів)
                self._aggregate_old_data()
                
                # 3. Компресія AGGREGATED в METADATA (30-90 днів)
                self._compress_to_metadata()
                
                # 4. Видалення METADATA старіше 90 днів
                self._cleanup_metadata(days=90)
                
                # 5. Перевірка розміру
                total_size = self._check_storage_size()
                logger.info(f"💾 [ROTATION] Total storage: {total_size:.2f} GB")
                
            except Exception as e:
                logger.error(f"❌ [ROTATION] Error: {e}")
    
    def _cleanup_raw_data(self, days: int):
        """Видалення RAW даних старіше N днів"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for date_folder in self.raw_path.iterdir():
            if not date_folder.is_dir():
                continue
            
            try:
                folder_date = datetime.strptime(date_folder.name, "%Y-%m-%d")
                if folder_date < cutoff:
                    # Видаляємо папку
                    import shutil
                    shutil.rmtree(date_folder)
                    logger.info(f"🗑️ [CLEANUP] Removed RAW: {date_folder.name}")
            except Exception as e:
                logger.error(f"❌ [CLEANUP] {date_folder}: {e}")
    
    def _aggregate_old_data(self):
        """Агрегація 7+ днів даних в 1-min bars"""
        # Placeholder - реалізація агрегації
        pass
    
    def _compress_to_metadata(self):
        """Компресія 30+ днів в metadata"""
        # Placeholder - реалізація компресії
        pass
    
    def _cleanup_metadata(self, days: int):
        """Видалення metadata старіше 90 днів"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for meta_file in self.meta_path.glob("*.parquet"):
            try:
                # Парсимо дату з назви файлу (напр. 2024-10_signals.parquet)
                date_str = meta_file.stem.split('_')[0]
                file_date = datetime.strptime(date_str, "%Y-%m")
                
                if file_date < cutoff:
                    meta_file.unlink()
                    logger.info(f"🗑️ [CLEANUP] Removed METADATA: {meta_file.name}")
            except Exception as e:
                logger.error(f"❌ [CLEANUP] {meta_file}: {e}")
    
    def _check_storage_size(self) -> float:
        """Перевірка розміру сховища в ГБ"""
        total_size = 0
        
        for path in [self.raw_path, self.agg_path, self.meta_path]:
            for file in path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
        
        return total_size / (1024 ** 3)  # Конвертація в ГБ
    
    async def _on_position_update(self, position):
        """Callback при оновленні позиції"""
        # Зберігаємо результати трейдів для аналізу
        if position.status == "CLOSED" and position.pnl_confirmed:
            trade_result = {
                'timestamp': position.closed_timestamp,
                'symbol': position.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': position.avg_exit_price,
                'pnl': position.realised_pnl,
                'close_reason': position.close_reason,
                'lifetime_sec': position.closed_timestamp - position.timestamp,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
            }
            
            self._add_to_buffer('signals', position.symbol, trade_result)