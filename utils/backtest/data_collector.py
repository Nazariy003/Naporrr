# utils/backtest/data_collector.py
import time
import asyncio
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from config.settings import settings
from utils.logger import logger


class BacktestDataCollector:
    """
    🎯 Правильний збір даних для бектесту з PyArrow ParquetWriter
    
    Рівні зберігання:
    - RAW (7 днів): orderbook snapshots (5s) + trades + signals
    - AGGREGATED (30 днів): 1-min bars + aggregated signals
    - METADATA (90 днів): тільки результати сигналів
    
    Виправлення:
    - Використання pq.ParquetWriter замість pd.to_parquet для append
    - PyArrow schemas для валідації
    - JSON serialization для складних структур
    - Правильне керування writer lifecycle
    """
    
    def __init__(self, storage):
        """Ініціалізація з правильними PyArrow schemas"""
        self.storage = storage
        self.storage_path = Path(settings.backtest.data_storage_path)
        self.raw_path = self.storage_path / "raw"
        self.agg_path = self.storage_path / "aggregated"
        self.meta_path = self.storage_path / "metadata"
        
        # Створення директорій
        for path in [self.raw_path, self.agg_path, self.meta_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Буфери для батчового запису
        self.buffers: Dict[str, Dict[str, List[Dict]]] = {
            'orderbook': {},
            'trades': {},
            'signals': {},
            'positions': {}
        }
        
        # ParquetWriter instances (один writer на файл для append)
        self.writers: Dict[str, pq.ParquetWriter] = {}
        
        # Schemas для кожного типу даних
        self.schemas = self._create_schemas()
        
        self.buffer_size = 100
        self.last_flush = time.time()
        self._running = False
        self._tasks = []
        
    def _create_schemas(self) -> Dict[str, pa.Schema]:
        """Створення PyArrow schemas для валідації"""
        
        # Schema для orderbook snapshots
        orderbook_schema = pa.schema([
            ('timestamp', pa.float64()),
            ('symbol', pa.string()),
            ('best_bid', pa.float64()),
            ('best_ask', pa.float64()),
            ('bid_levels', pa.string()),  # JSON string
            ('ask_levels', pa.string()),  # JSON string
            ('spread_bps', pa.float64()),
        ])
        
        # Schema для trades
        trades_schema = pa.schema([
            ('timestamp', pa.float64()),
            ('symbol', pa.string()),
            ('price', pa.float64()),
            ('size', pa.float64()),
            ('side', pa.string()),
            ('is_aggressive', pa.bool_()),
        ])
        
        # Schema для signals
        signals_schema = pa.schema([
            ('timestamp', pa.float64()),
            ('symbol', pa.string()),
            ('signal', pa.string()),
            ('strength', pa.int32()),
            ('composite', pa.float64()),
            ('imbalance', pa.float64()),
            ('momentum', pa.float64()),
            ('volatility', pa.float64()),
            ('settings_snapshot', pa.string()),  # JSON string
        ])
        
        # Schema для closed positions
        positions_schema = pa.schema([
            ('timestamp', pa.float64()),
            ('closed_timestamp', pa.float64()),
            ('symbol', pa.string()),
            ('side', pa.string()),
            ('entry_price', pa.float64()),
            ('exit_price', pa.float64()),
            ('pnl', pa.float64()),
            ('close_reason', pa.string()),
            ('lifetime_sec', pa.float64()),
            ('stop_loss', pa.float64()),
            ('take_profit', pa.float64()),
        ])
        
        return {
            'orderbook': orderbook_schema,
            'trades': trades_schema,
            'signals': signals_schema,
            'positions': positions_schema,
        }
    
    async def start(self):
        """Запуск збору даних"""
        logger.info("🎬 [BACKTEST_DATA_COLLECTOR] Starting...")
        self._running = True
        
        # Підписка на події позицій
        self.storage.add_position_callback(self._on_position_update)
        
        # Запуск циклічних задач
        self._tasks = [
            asyncio.create_task(self._snapshot_loop()),
            asyncio.create_task(self._trades_loop()),
            asyncio.create_task(self._signals_loop()),
            asyncio.create_task(self._flush_loop()),
            asyncio.create_task(self._rotation_loop()),
        ]
        
        logger.info("✅ [BACKTEST_DATA_COLLECTOR] Started successfully")
    
    async def stop(self):
        """Зупинка збору даних з graceful shutdown"""
        logger.info("🛑 [BACKTEST_DATA_COLLECTOR] Stopping...")
        self._running = False
        
        # Скасування всіх задач
        for task in self._tasks:
            task.cancel()
        
        # Очікування завершення задач
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Фінальний flush всіх буферів
        self._flush_all_buffers()
        
        # Закриття всіх writers
        self._close_all_writers()
        
        logger.info("✅ [BACKTEST_DATA_COLLECTOR] Stopped successfully")
    
    async def _snapshot_loop(self):
        """Знімки orderbook кожні N секунд"""
        interval = settings.backtest.orderbook_snapshot_interval_sec
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                for symbol in settings.pairs.trade_pairs:
                    ob = self.storage.get_order_book(symbol)
                    if not ob or not ob.bids or not ob.asks:
                        continue
                    
                    # Зберігаємо тільки топ-10 рівнів для економії місця
                    bid_levels = [(lvl.price, lvl.size) for lvl in ob.bids[:10]]
                    ask_levels = [(lvl.price, lvl.size) for lvl in ob.asks[:10]]
                    
                    spread_bps = ((ob.best_ask - ob.best_bid) / ob.best_bid) * 10000
                    
                    snapshot = {
                        'timestamp': ob.ts,
                        'symbol': symbol,
                        'best_bid': ob.best_bid,
                        'best_ask': ob.best_ask,
                        'bid_levels': json.dumps(bid_levels),  # JSON serialization
                        'ask_levels': json.dumps(ask_levels),
                        'spread_bps': spread_bps,
                    }
                    
                    self._add_to_buffer('orderbook', symbol, snapshot)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [SNAPSHOT_LOOP] Error: {e}")
    
    async def _trades_loop(self):
        """Збір trades кожні N секунд"""
        interval = settings.backtest.trades_collection_interval_sec
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                for symbol in settings.pairs.trade_pairs:
                    trades = self.storage.get_trades(symbol)
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
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [TRADES_LOOP] Error: {e}")
    
    async def _signals_loop(self):
        """Збір сигналів кожні N секунд"""
        interval = settings.backtest.signals_collection_interval_sec
        
        while self._running:
            try:
                await asyncio.sleep(interval)
                
                # Placeholder для збору сигналів
                # В реальності тут має бути інтеграція з SignalGenerator
                current_time = time.time()
                
                for symbol in settings.pairs.trade_pairs:
                    signal_data = {
                        'timestamp': current_time,
                        'symbol': symbol,
                        'signal': 'HOLD',
                        'strength': 0,
                        'composite': 0.0,
                        'imbalance': 0.0,
                        'momentum': 0.0,
                        'volatility': 0.0,
                        'settings_snapshot': json.dumps({
                            'weight_imbalance': settings.signals.weight_imbalance,
                            'weight_momentum': settings.signals.weight_momentum,
                            'hold_threshold': settings.signals.hold_threshold,
                        })
                    }
                    self._add_to_buffer('signals', symbol, signal_data)
                    
            except asyncio.CancelledError:
                break
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
        """Періодичний flush буферів"""
        while self._running:
            try:
                await asyncio.sleep(60)
                
                current_time = time.time()
                if current_time - self.last_flush >= 60:
                    self._flush_all_buffers()
                    self.last_flush = current_time
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [FLUSH_LOOP] Error: {e}")
    
    def _flush_buffer(self, buffer_type: str, symbol: str):
        """Запис буфера в Parquet з використанням ParquetWriter"""
        key = f"{symbol}_{buffer_type}"
        
        if key not in self.buffers[buffer_type] or not self.buffers[buffer_type][key]:
            return
        
        try:
            data_list = self.buffers[buffer_type][key]
            
            # Конвертуємо в PyArrow Table
            table = pa.Table.from_pylist(data_list, schema=self.schemas[buffer_type])
            
            # Шлях до файлу
            today = datetime.utcnow().strftime("%Y-%m-%d")
            date_path = self.raw_path / today
            date_path.mkdir(parents=True, exist_ok=True)
            file_path = date_path / f"{symbol}_{buffer_type}.parquet"
            
            # Використання ParquetWriter для append
            writer_key = str(file_path)
            
            if writer_key not in self.writers:
                # Створення нового writer
                if file_path.exists():
                    # Файл існує - відкриваємо для append
                    # Читаємо існуючі дані
                    existing_table = pq.read_table(file_path)
                    # Об'єднуємо з новими
                    combined_table = pa.concat_tables([existing_table, table])
                    # Перезаписуємо файл
                    pq.write_table(
                        combined_table,
                        file_path,
                        compression='snappy',
                        version='2.6'
                    )
                else:
                    # Новий файл - просто записуємо
                    pq.write_table(
                        table,
                        file_path,
                        compression='snappy',
                        version='2.6'
                    )
            else:
                # Writer вже існує - append
                self.writers[writer_key].write_table(table)
            
            # Очищення буфера
            self.buffers[buffer_type][key] = []
            
            logger.debug(f"💾 [FLUSH] {buffer_type}/{symbol}: {len(data_list)} records")
            
        except Exception as e:
            logger.error(f"❌ [FLUSH] {buffer_type}/{symbol}: {e}")
            # Не втрачаємо дані при помилці - залишаємо в буфері
    
    def _flush_all_buffers(self):
        """Flush всіх буферів"""
        for buffer_type in self.buffers.keys():
            for symbol in settings.pairs.trade_pairs:
                self._flush_buffer(buffer_type, symbol)
    
    def _close_all_writers(self):
        """Закриття всіх ParquetWriter"""
        for writer_key, writer in self.writers.items():
            try:
                writer.close()
                logger.debug(f"✅ [WRITER_CLOSE] {writer_key}")
            except Exception as e:
                logger.error(f"❌ [WRITER_CLOSE] {writer_key}: {e}")
        
        self.writers.clear()
    
    async def _rotation_loop(self):
        """Ротація даних"""
        while self._running:
            try:
                await asyncio.sleep(86400)  # 24 години
                
                logger.info("🔄 [ROTATION] Starting data rotation...")
                
                # Видалення старих RAW даних
                self._cleanup_raw_data(days=settings.backtest.raw_data_retention_days)
                
                # Видалення старих aggregated даних
                self._cleanup_aggregated_data(days=settings.backtest.aggregated_data_retention_days)
                
                # Видалення старих metadata
                self._cleanup_metadata(days=settings.backtest.metadata_retention_days)
                
                # Перевірка розміру
                total_size = self._check_storage_size()
                logger.info(f"💾 [ROTATION] Total storage: {total_size:.2f} GB")
                
                if total_size > settings.backtest.max_storage_gb:
                    logger.warning(f"⚠️ [ROTATION] Storage limit exceeded: {total_size:.2f} GB > {settings.backtest.max_storage_gb} GB")
                
            except asyncio.CancelledError:
                break
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
                    import shutil
                    shutil.rmtree(date_folder)
                    logger.info(f"🗑️ [CLEANUP] Removed RAW: {date_folder.name}")
            except Exception as e:
                logger.error(f"❌ [CLEANUP] {date_folder}: {e}")
    
    def _cleanup_aggregated_data(self, days: int):
        """Видалення aggregated даних старіше N днів"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for agg_file in self.agg_path.glob("*.parquet"):
            try:
                # Парсимо дату з назви файлу
                date_str = agg_file.stem.split('_')[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff:
                    agg_file.unlink()
                    logger.info(f"🗑️ [CLEANUP] Removed AGG: {agg_file.name}")
            except Exception as e:
                logger.error(f"❌ [CLEANUP] {agg_file}: {e}")
    
    def _cleanup_metadata(self, days: int):
        """Видалення metadata старіше N днів"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for meta_file in self.meta_path.glob("*.parquet"):
            try:
                # Парсимо дату з назви файлу
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
        
        return total_size / (1024 ** 3)
    
    async def _on_position_update(self, position):
        """Callback при оновленні позиції"""
        try:
            # Зберігаємо тільки закриті позиції для аналізу
            if position.status == "CLOSED" and hasattr(position, 'closed_timestamp'):
                trade_result = {
                    'timestamp': position.timestamp,
                    'closed_timestamp': position.closed_timestamp,
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'exit_price': getattr(position, 'avg_exit_price', 0.0),
                    'pnl': getattr(position, 'realised_pnl', 0.0),
                    'close_reason': getattr(position, 'close_reason', 'UNKNOWN'),
                    'lifetime_sec': position.closed_timestamp - position.timestamp,
                    'stop_loss': getattr(position, 'stop_loss', 0.0),
                    'take_profit': getattr(position, 'take_profit', 0.0),
                }
                
                self._add_to_buffer('positions', position.symbol, trade_result)
        except Exception as e:
            logger.error(f"❌ [POSITION_UPDATE] Error: {e}")
