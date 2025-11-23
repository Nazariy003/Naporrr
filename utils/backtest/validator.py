# utils/backtest/validator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from utils.logger import logger

class WalkForwardValidator:
    """
    🎯 Walk-Forward Validation
    
    Розділяє історичні дані на:
    - Training periods (60%)
    - Validation periods (40%)
    
    Перевіряє overfitting
    """
    
    def __init__(self, train_ratio: float = 0.6):
        self.train_ratio = train_ratio
        
    def split_data(self, 
                  start_date: datetime,
                  end_date: datetime,
                  n_splits: int = 3) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Розділення на train/validation periods
        
        Returns:
            List[(train_start, train_end, val_start, val_end)]
        """
        total_days = (end_date - start_date).days
        split_size = total_days // n_splits
        
        splits = []
        
        for i in range(n_splits):
            split_start = start_date + timedelta(days=i * split_size)
            split_end = split_start + timedelta(days=split_size)
            
            # Train/Val split
            train_days = int(split_size * self.train_ratio)
            
            train_start = split_start
            train_end = split_start + timedelta(days=train_days)
            val_start = train_end
            val_end = split_end
            
            splits.append((train_start, train_end, val_start, val_end))
            
            logger.info(f"📅 [WF_SPLIT_{i+1}] "
                       f"Train: {train_start.date()} to {train_end.date()} | "
                       f"Val: {val_start.date()} to {val_end.date()}")
        
        return splits
    
    def validate(self, 
                replay_engine,
                optimizer,
                start_date: datetime,
                end_date: datetime,
                symbols: List[str]) -> Dict:
        """
        Walk-forward validation
        
        Returns:
            Validation results з consistency check
        """
        logger.info("🔬 [WALK_FORWARD] Starting validation...")
        
        # Розділення на periods
        splits = self.split_data(start_date, end_date, n_splits=3)
        
        all_results = []
        
        for idx, (train_start, train_end, val_start, val_end) in enumerate(splits, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 [WF_FOLD_{idx}] Training period...")
            
            # Оптимізація на training period
            best_result, _ = optimizer.optimize(
                replay_engine=replay_engine,
                start_date=train_start,
                end_date=train_end,
                symbols=symbols,
                max_combinations=50  # Обмежуємо для швидкості
            )
            
            if not best_result:
                logger.error(f"❌ [WF_FOLD_{idx}] No valid optimization results")
                continue
            
            best_params = best_result['params']
            train_metrics = best_result['metrics']
            
            logger.info(f"✅ [WF_FOLD_{idx}] Best train score: {best_result['score']:.4f}")
            
            # Валідація на validation period
            logger.info(f"🧪 [WF_FOLD_{idx}] Validating on out-of-sample period...")
            
            val_result = replay_engine.replay_period(
                start_date=val_start,
                end_date=val_end,
                symbols=symbols,
                test_params=best_params
            )
            
            if not val_result:
                logger.error(f"❌ [WF_FOLD_{idx}] Validation failed")
                continue
            
            val_metrics = val_result['metrics']
            
            # Порівняння train vs validation
            consistency = self._check_consistency(train_metrics, val_metrics)
            
            all_results.append({
                'fold': idx,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_params': best_params,
                'consistency': consistency
            })
            
            logger.info(f"📊 [WF_FOLD_{idx}] Consistency score: {consistency['score']:.2f}")
        
        # Агрегація результатів
        summary = self._summarize_validation(all_results)
        
        return {
            'splits': all_results,
            'summary': summary,
            'recommendation': self._get_recommendation(summary)
        }
    
    def _check_consistency(self, train_metrics: Dict, val_metrics: Dict) -> Dict:
        """
        Перевірка consistency між train і validation
        
        Overfitting якщо валідація значно гірша
        """
        consistency = {}
        
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio']:
            train_val = train_metrics.get(metric, 0)
            val_val = val_metrics.get(metric, 0)
            
            if train_val > 0:
                degradation_pct = ((train_val - val_val) / train_val) * 100
            else:
                degradation_pct = 0
            
            consistency[metric] = {
                'train': train_val,
                'validation': val_val,
                'degradation_pct': degradation_pct,
                'acceptable': degradation_pct < 20  # Менше 20% деградації - OK
            }
        
        # Загальний consistency score
        acceptable_count = sum(1 for v in consistency.values() if v.get('acceptable', False))
        consistency['score'] = (acceptable_count / len(consistency)) * 100
        
        return consistency
    
    def _summarize_validation(self, all_results: List[Dict]) -> Dict:
        """Агрегація validation results"""
        if not all_results:
            return {}
        
        # Середній consistency score
        avg_consistency = np.mean([r['consistency']['score'] for r in all_results])
        
        # Середні метрики на validation
        val_metrics = {}
        for metric in ['win_rate', 'profit_factor', 'sharpe_ratio', 'total_pnl']:
            values = [r['val_metrics'].get(metric, 0) for r in all_results]
            val_metrics[f'avg_{metric}'] = np.mean(values)
            val_metrics[f'std_{metric}'] = np.std(values)
        
        return {
            'avg_consistency_score': avg_consistency,
            'validation_metrics': val_metrics,
            'n_folds': len(all_results)
        }
    
    def _get_recommendation(self, summary: Dict) -> Dict:
        """Рекомендація на основі валідації"""
        consistency = summary.get('avg_consistency_score', 0)
        
        if consistency >= 80:
            verdict = "EXCELLENT"
            message = "Parameters are robust and generalize well"
        elif consistency >= 60:
            verdict = "GOOD"
            message = "Parameters show acceptable consistency"
        elif consistency >= 40:
            verdict = "MODERATE"
            message = "Some overfitting detected, use with caution"
        else:
            verdict = "POOR"
            message = "Significant overfitting, DO NOT apply parameters"
        
        return {
            'verdict': verdict,
            'message': message,
            'should_apply': consistency >= 60
        }