# utils/backtest/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List
from utils.logger import logger

class MetricsCalculator:
    """
    🎯 Розрахунок торгових метрик
    
    Metrics:
    - Basic: win_rate, total_trades, avg_win/loss
    - Risk: max_drawdown, sharpe, sortino, calmar
    - Operational: avg_duration, tp/sl/time_exit counts
    """
    
    @staticmethod
    def calculate_all_metrics(trades: List[Dict]) -> Dict:
        """Розрахунок всіх метрик"""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        
        metrics = {}
        
        # Basic metrics
        metrics.update(MetricsCalculator._calculate_basic(df))
        
        # Risk metrics
        metrics.update(MetricsCalculator._calculate_risk(df))
        
        # Operational metrics
        metrics.update(MetricsCalculator._calculate_operational(df))
        
        return metrics
    
    @staticmethod
    def _calculate_basic(df: pd.DataFrame) -> Dict:
        """Базові метрики"""
        total_trades = len(df)
        
        if 'pnl' not in df.columns:
            return {'total_trades': total_trades}
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
        
        total_pnl = df['pnl'].sum()
        
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1e-9
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'total_pnl': round(total_pnl, 4),
            'gross_profit': round(gross_profit, 4),
            'gross_loss': round(gross_loss, 4),
            'profit_factor': round(profit_factor, 2),
            'largest_win': round(df['pnl'].max(), 4),
            'largest_loss': round(df['pnl'].min(), 4),
        }
    
    @staticmethod
    def _calculate_risk(df: pd.DataFrame) -> Dict:
        """Метрики ризику"""
        if 'pnl' not in df.columns:
            return {}
        
        # Equity curve
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Max Drawdown
        running_max = df['cumulative_pnl'].cummax()
        drawdown = df['cumulative_pnl'] - running_max
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / running_max.max() * 100) if running_max.max() > 0 else 0
        
        # Sharpe Ratio (annualized)
        returns = df['pnl']
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)  # Assume daily
        else:
            sharpe = 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0
        
        # Calmar Ratio
        total_return = df['cumulative_pnl'].iloc[-1] if len(df) > 0 else 0
        calmar = (total_return / abs(max_dd)) if max_dd < 0 else 0
        
        return {
            'max_drawdown': round(max_dd, 4),
            'max_drawdown_pct': round(abs(max_dd_pct), 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
        }
    
    @staticmethod
    def _calculate_operational(df: pd.DataFrame) -> Dict:
        """Операційні метрики"""
        if 'lifetime_sec' in df.columns:
            avg_duration_min = df['lifetime_sec'].mean() / 60
        else:
            avg_duration_min = 0
        
        # Close reasons
        if 'close_reason' in df.columns:
            close_reasons = df['close_reason'].value_counts().to_dict()
            
            tp_hit = close_reasons.get('TP_HIT', 0)
            sl_hit = close_reasons.get('SL_HIT', 0)
            time_exit = close_reasons.get('TIME_EXIT', 0)
            
            tp_pct = (tp_hit / len(df) * 100) if len(df) > 0 else 0
            sl_pct = (sl_hit / len(df) * 100) if len(df) > 0 else 0
            time_pct = (time_exit / len(df) * 100) if len(df) > 0 else 0
        else:
            close_reasons = {}
            tp_pct = sl_pct = time_pct = 0
        
        return {
            'avg_duration_min': round(avg_duration_min, 2),
            'close_reasons': close_reasons,
            'tp_hit_pct': round(tp_pct, 2),
            'sl_hit_pct': round(sl_pct, 2),
            'time_exit_pct': round(time_pct, 2),
        }