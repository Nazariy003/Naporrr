# utils/backtest/replay_engine.py - ОСТАТОЧНА ВИПРАВЛЕНА ВЕРСІЯ
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from config.settings import settings
from utils.logger import logger
from utils.backtest.metrics import MetricsCalculator

@dataclass
class SimulatedTrade:
    """Симульований трейд"""
    timestamp: float
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    close_reason: str
    lifetime_sec: float
    signal_strength: int
    stop_loss: float
    take_profit: float

class ReplayEngine:
    """
    🎯 Replay історичних даних через існуючі аналізатори
    
    ✅ ВИПРАВЛЕНО: Реалістичні пороги для крипто-трейдингу
    """
    
    def __init__(self, data_path: str = "utils/data_storage"):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.metrics_calc = MetricsCalculator()
        
    def replay_period(self, 
                     start_date: datetime,
                     end_date: datetime,
                     symbols: List[str],
                     test_params: Dict) -> Dict:
        """Replay періоду з альтернативними параметрами"""
        logger.info(f"🎬 [REPLAY] Period: {start_date.date()} to {end_date.date()}")
        logger.debug(f"🎯 [REPLAY] Test params: {test_params}")
        
        # Завантаження даних
        data = self._load_historical_data(start_date, end_date, symbols)
        
        if not data:
            logger.error("❌ [REPLAY] No data found")
            return {'trades': [], 'metrics': {}, 'signals_log': []}
        
        # Replay для кожного символу
        all_trades = []
        all_signals = []
        
        for symbol in symbols:
            if symbol not in data:
                continue
            
            symbol_trades, symbol_signals = self._replay_symbol(
                symbol=symbol,
                data=data[symbol],
                test_params=test_params
            )
            
            all_trades.extend(symbol_trades)
            all_signals.extend(symbol_signals)
        
        # Розрахунок метрик
        if all_trades:
            metrics = self.metrics_calc.calculate_all_metrics(all_trades)
            logger.info(f"✅ [REPLAY] Completed: {len(all_trades)} trades, "
                       f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        else:
            logger.warning("⚠️  [REPLAY] No trades generated")
            metrics = {}
        
        return {
            'trades': all_trades,
            'metrics': metrics,
            'signals_log': all_signals
        }
    
    def _load_historical_data(self, 
                             start_date: datetime,
                             end_date: datetime,
                             symbols: List[str]) -> Dict:
        """Завантаження історичних даних"""
        data = {}
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_folder = self.raw_path / date_str
            
            if not date_folder.exists():
                logger.debug(f"⏩ [LOAD] No data folder for {date_str}")
                current_date += timedelta(days=1)
                continue
            
            for symbol in symbols:
                if symbol not in data:
                    data[symbol] = {
                        'orderbook': [],
                        'trades': [],
                        'signals': []
                    }
                
                for data_type, filename in [
                    ('orderbook', f"{symbol}_orderbook.parquet"),
                    ('trades', f"{symbol}_trades.parquet"),
                    ('signals', f"{symbol}_signals.parquet")
                ]:
                    file_path = date_folder / filename
                    if file_path.exists():
                        try:
                            df = pd.read_parquet(file_path)
                            data[symbol][data_type].append(df)
                            logger.debug(f"✅ [LOAD] {data_type.title()} {symbol} {date_str}: {len(df)} rows")
                        except Exception as e:
                            logger.error(f"❌ [LOAD] {data_type} {symbol} {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        # Конкатенація
        for symbol in data:
            for key in ['orderbook', 'trades', 'signals']:
                if data[symbol][key]:
                    data[symbol][key] = pd.concat(data[symbol][key], ignore_index=True)
                    data[symbol][key] = data[symbol][key].sort_values('timestamp').reset_index(drop=True)
                else:
                    data[symbol][key] = pd.DataFrame()
        
        logger.info(f"✅ [LOAD] Loaded {len(data)} symbols")
        
        return data
    
    def _replay_symbol(self, symbol: str, data: Dict, test_params: Dict) -> Tuple[List, List]:
        """Replay одного символу з ВИПРАВЛЕНИМИ порогами"""
        trades = []
        signals_log = []
        
        if data['trades'].empty or data['orderbook'].empty:
            logger.warning(f"⚠️  [REPLAY] Insufficient data for {symbol}")
            return trades, signals_log
        
        trades_df = data['trades']
        orderbook_df = data['orderbook']
        
        logger.info(f"🔄 [REPLAY] Processing {symbol}:")
        logger.info(f"   • Trades: {len(trades_df)} rows")
        logger.info(f"   • Orderbook: {len(orderbook_df)} rows")
        
        # Генерація 1-min bars
        trades_df['minute'] = pd.to_datetime(trades_df['timestamp'], unit='s').dt.floor('1min')
        minute_bars = trades_df.groupby('minute').agg({
            'price': ['first', 'last', 'min', 'max'],
            'size': 'sum',
            'timestamp': 'first'
        }).reset_index()
        
        minute_bars.columns = ['minute', 'open', 'close', 'low', 'high', 'volume', 'timestamp']
        
        logger.info(f"   • Generated {len(minute_bars)} 1-min bars")
        
        # ✅ DEBUG: Показати діапазон цін
        if len(minute_bars) > 0:
            price_start = minute_bars.iloc[0]['close']
            price_end = minute_bars.iloc[-1]['close']
            price_change_total = ((price_end - price_start) / price_start) * 100
            price_max = minute_bars['high'].max()
            price_min = minute_bars['low'].min()
            price_range_pct = ((price_max - price_min) / price_min) * 100
            
            logger.info(f"   • Price start: ${price_start:.4f}")
            logger.info(f"   • Price end: ${price_end:.4f}")
            logger.info(f"   • Total change: {price_change_total:+.3f}%")
            logger.info(f"   • Range: ${price_min:.4f} - ${price_max:.4f} ({price_range_pct:.2f}%)")
        
        if len(minute_bars) < 5:
            logger.warning(f"⚠️  [REPLAY] Too few bars for {symbol}, skipping")
            return trades, signals_log
        
        # Симуляція трейдів
        position = None
        trades_opened = 0
        signals_generated = 0
        
        for idx in range(5, len(minute_bars)):  # ✅ 5 хвилин історії (було 10)
            current_bar = minute_bars.iloc[idx]
            timestamp = current_bar['timestamp']
            close_price = current_bar['close']
            
            # Перевірка відкритої позиції
            if position:
                exit_reason = None
                exit_price = None
                
                # TP/SL check
                if position['side'] == 'LONG':
                    if close_price >= position['tp']:
                        exit_reason = 'TP_HIT'
                        exit_price = position['tp']
                    elif close_price <= position['sl']:
                        exit_reason = 'SL_HIT'
                        exit_price = position['sl']
                    elif timestamp - position['entry_time'] >= position['max_lifetime']:
                        exit_reason = 'TIME_EXIT'
                        exit_price = close_price
                else:  # SHORT
                    if close_price <= position['tp']:
                        exit_reason = 'TP_HIT'
                        exit_price = position['tp']
                    elif close_price >= position['sl']:
                        exit_reason = 'SL_HIT'
                        exit_price = position['sl']
                    elif timestamp - position['entry_time'] >= position['max_lifetime']:
                        exit_reason = 'TIME_EXIT'
                        exit_price = close_price
                
                # Закриття
                if exit_reason:
                    if position['side'] == 'LONG':
                        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                    else:
                        pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100
                    
                    lifetime_sec = timestamp - position['entry_time']
                    
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl_pct,
                        'close_reason': exit_reason,
                        'lifetime_sec': lifetime_sec,
                        'signal_strength': position['strength'],
                        'stop_loss': position['sl'],
                        'take_profit': position['tp']
                    }
                    
                    trades.append(trade)
                    logger.debug(f"📉 [CLOSE] {exit_reason} @ {exit_price:.4f}, PnL={pnl_pct:+.2f}%")
                    position = None
                    continue
            
            # Генерація сигналу
            if not position:
                lookback_bars = minute_bars.iloc[idx-5:idx]  # ✅ 5 хвилин (було 10)
                price_change = ((current_bar['close'] - lookback_bars['close'].iloc[0]) / 
                               lookback_bars['close'].iloc[0]) * 100
                
                signal, strength = self._calculate_signal(
                    price_change=price_change,
                    volume=current_bar['volume'],
                    test_params=test_params
                )
                
                signals_generated += 1
                
                signals_log.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'signal': signal,
                    'strength': strength,
                    'price_change': price_change
                })
                
                # Відкриття позиції
                min_strength = test_params.get('entry_signal_min_strength', 
                                              settings.trading.entry_signal_min_strength)
                
                if strength >= min_strength and signal in ['BUY', 'SELL']:
                    side = 'LONG' if signal == 'BUY' else 'SHORT'
                    
                    # TP/SL
                    volatility = lookback_bars['close'].pct_change().std() * 100
                    volatility = max(volatility, 0.3)  # Мінімум 0.3%
                    
                    sl, tp = self._recalculate_tp_sl(
                        side=side,
                        entry_price=close_price,
                        volatility=volatility,
                        test_params=test_params
                    )
                    
                    # Lifetime
                    max_lifetime = self._recalculate_lifetime(
                        volatility=volatility,
                        test_params=test_params
                    )
                    
                    position = {
                        'side': side,
                        'entry_price': close_price,
                        'entry_time': timestamp,
                        'sl': sl,
                        'tp': tp,
                        'max_lifetime': max_lifetime,
                        'strength': strength
                    }
                    
                    trades_opened += 1
                    logger.debug(f"📈 [OPEN] {side} @ {close_price:.4f}, SL={sl:.4f}, TP={tp:.4f}")
        
        # Закриваємо останню позицію
        if position:
            exit_price = minute_bars.iloc[-1]['close']
            timestamp = minute_bars.iloc[-1]['timestamp']
            
            if position['side'] == 'LONG':
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            else:
                pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100
            
            lifetime_sec = timestamp - position['entry_time']
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl_pct,
                'close_reason': 'TIME_EXIT',
                'lifetime_sec': lifetime_sec,
                'signal_strength': position['strength'],
                'stop_loss': position['sl'],
                'take_profit': position['tp']
            }
            
            trades.append(trade)
        
        # Підсумки
        if trades:
            logger.info(f"   ✅ Generated {len(trades)} trades (opened {trades_opened})")
        else:
            logger.warning(f"   ⚠️  No trades (ринок надто стабільний або пороги завищені)")
            logger.info(f"   • Сигналів згенеровано: {signals_generated}")
            logger.info(f"   • Підказка: Зменшіть hold_threshold або збільшіть період збору даних")
        
        return trades, signals_log
    
    def _calculate_signal(self, price_change: float, volume: float, test_params: Dict) -> Tuple[str, int]:
        """✅ ВИПРАВЛЕНИЙ розрахунок сигналу з реалістичними порогами"""
        
        # ✅ ФІКСОВАНИЙ ПОРІГ: 0.1% (ігноруємо settings.hold_threshold=12%)
        hold_threshold = 0.1  # 0.1% мінімальна зміна для сигналу
        
        # Можна перевизначити з test_params
        if 'signal_threshold_pct' in test_params:
            hold_threshold = test_params['signal_threshold_pct']
        
        # Сигнал
        if abs(price_change) < hold_threshold:
            return 'HOLD', 0
        
        signal = 'BUY' if price_change > 0 else 'SELL'
        
        # Сила сигналу (реалістичні пороги для крипти)
        abs_change = abs(price_change)
        
        if abs_change >= 2.0:      # >=2%
            strength = 5
        elif abs_change >= 1.0:    # >=1%
            strength = 4
        elif abs_change >= 0.5:    # >=0.5%
            strength = 3
        elif abs_change >= 0.2:    # >=0.2%
            strength = 2
        else:                      # <0.2%
            strength = 1
        
        return signal, strength
    
    def _recalculate_tp_sl(self, side: str, entry_price: float, 
                          volatility: float, test_params: Dict) -> Tuple[float, float]:
        """Перерахунок TP/SL"""
        if entry_price <= 0:
            return entry_price, entry_price
        
        sl_mult = test_params.get('sl_vol_multiplier', settings.risk.sl_vol_multiplier)
        tp_mult = test_params.get('tp_vol_multiplier', settings.risk.tp_vol_multiplier)
        
        sl_pct = (volatility * sl_mult) / 100
        tp_pct = (volatility * tp_mult) / 100
        
        sl_pct = max(settings.risk.min_sl_pct, min(sl_pct, settings.risk.max_sl_pct))
        tp_pct = max(settings.risk.min_tp_pct, min(tp_pct, settings.risk.max_tp_pct))
        
        if side == 'LONG':
            sl = entry_price * (1 - sl_pct)
            tp = entry_price * (1 + tp_pct)
        else:
            sl = entry_price * (1 + sl_pct)
            tp = entry_price * (1 - tp_pct)
        
        return sl, tp
    
    def _recalculate_lifetime(self, volatility: float, test_params: Dict) -> float:
        """Перерахунок lifetime"""
        base_lifetime = settings.risk.base_position_lifetime_minutes * 60
        
        low_mult = test_params.get('low_volatility_lifetime_multiplier', 
                                   settings.risk.low_volatility_lifetime_multiplier)
        high_mult = test_params.get('high_volatility_lifetime_multiplier',
                                    settings.risk.high_volatility_lifetime_multiplier)
        
        if volatility < settings.risk.volatility_threshold_low:
            multiplier = low_mult
        elif volatility > settings.risk.volatility_threshold_high:
            multiplier = high_mult
        else:
            multiplier = 1.0
        
        return base_lifetime * multiplier