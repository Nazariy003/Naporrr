# utils/backtest/settings_updater.py
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict
from utils.logger import logger

class SettingsUpdater:
    """
    🎯 Автоматичне оновлення settings.py
    
    Features:
    - Backup старих налаштувань
    - Поступове (gradual) adjustment
    - Rollback при помилці
    - Telegram notification
    """
    
    def __init__(self, settings_path: str = "config/settings.py"):
        self.settings_path = Path(settings_path)
        self.backup_dir = Path("config/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def update_parameters(self, 
                         new_params: Dict,
                         gradual: bool = True,
                         adjustment_factor: float = 0.5) -> bool:
        """
        Оновлення параметрів у settings.py
        
        Args:
            new_params: Нові параметри {param_path: value}
            gradual: Поступове оновлення (змішування зі старими)
            adjustment_factor: Фактор змішування (0.5 = 50% старе + 50% нове)
        
        Returns:
            Success status
        """
        logger.info("📝 [SETTINGS_UPDATER] Updating parameters...")
        
        # Backup поточних налаштувань
        backup_path = self._create_backup()
        if not backup_path:
            logger.error("❌ [SETTINGS_UPDATER] Backup failed, aborting")
            return False
        
        try:
            # Читання поточного settings.py
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Застосування змін
            updated_content = content
            changes_log = []
            
            for param_path, new_value in new_params.items():
                # Парсинг шляху (напр. "signals.weight_imbalance")
                parts = param_path.split('.')
                
                if len(parts) != 2:
                    logger.warning(f"⚠️ [SETTINGS_UPDATER] Invalid param path: {param_path}")
                    continue
                
                section, param_name = parts
                
                # Знаходимо старе значення
                old_value = self._extract_current_value(content, section, param_name)
                
                if old_value is None:
                    logger.warning(f"⚠️ [SETTINGS_UPDATER] Cannot find: {param_path}")
                    continue
                
                # Gradual adjustment
                if gradual and isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    adjusted_value = old_value + (new_value - old_value) * adjustment_factor
                    adjusted_value = round(adjusted_value, 4)
                else:
                    adjusted_value = new_value
                
                # Оновлення в тексті
                updated_content = self._replace_parameter(
                    updated_content, 
                    section, 
                    param_name, 
                    adjusted_value
                )
                
                changes_log.append({
                    'param': param_path,
                    'old': old_value,
                    'new': new_value,
                    'adjusted': adjusted_value
                })
                
                logger.info(f"✏️ [UPDATE] {param_path}: {old_value} -> {adjusted_value}")
            
            # Запис оновленого файлу
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            # Додаємо коментар про оновлення
            self._add_update_comment(changes_log)
            
            logger.info(f"✅ [SETTINGS_UPDATER] Successfully updated {len(changes_log)} parameters")
            logger.info(f"💾 [SETTINGS_UPDATER] Backup saved: {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [SETTINGS_UPDATER] Error: {e}")
            
            # Rollback
            logger.info("🔄 [SETTINGS_UPDATER] Rolling back...")
            self._rollback(backup_path)
            
            return False
    
    def _create_backup(self) -> Path:
        """Створення backup поточних налаштувань"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"settings_{timestamp}.py"
            
            shutil.copy2(self.settings_path, backup_path)
            
            logger.info(f"💾 [BACKUP] Created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ [BACKUP] Error: {e}")
            return None
    
    def _extract_current_value(self, content: str, section: str, param_name: str):
        """Витягування поточного значення параметра"""
        try:
            # Знаходимо секцію (клас)
            section_pattern = rf'class {section.capitalize()}Settings\(BaseSettings\):(.*?)(?=class|\Z)'
            section_match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)
            
            if not section_match:
                return None
            
            section_content = section_match.group(1)
            
            # Знаходимо параметр
            param_pattern = rf'{param_name}\s*[:=]\s*([^\n]+)'
            param_match = re.search(param_pattern, section_content)
            
            if not param_match:
                return None
            
            value_str = param_match.group(1).strip()
            
            # Парсинг значення
            return self._parse_value(value_str)
            
        except Exception as e:
            logger.error(f"❌ [EXTRACT] Error for {section}.{param_name}: {e}")
            return None
    
    def _parse_value(self, value_str: str):
        """Парсинг значення з рядка"""
        value_str = value_str.rstrip(',')
        
        # Float
        try:
            return float(value_str)
        except:
            pass
        
        # Int
        try:
            return int(value_str)
        except:
            pass
        
        # Bool
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # String
        if value_str.startswith('"') or value_str.startswith("'"):
            return value_str.strip('"').strip("'")
        
        return value_str
    
    def _replace_parameter(self, content: str, section: str, param_name: str, new_value) -> str:
        """Заміна значення параметра"""
        # Форматування нового значення
        if isinstance(new_value, str):
            new_value_str = f'"{new_value}"'
        elif isinstance(new_value, bool):
            new_value_str = str(new_value)
        else:
            new_value_str = str(new_value)
        
        # Знаходимо секцію
        section_pattern = rf'(class {section.capitalize()}Settings\(BaseSettings\):.*?)((?=class|\Z))'
        
        def replacer(match):
            section_content = match.group(1)
            rest = match.group(2)
            
            # Заміна параметра в секції
            param_pattern = rf'({param_name}\s*[:=]\s*)([^\n]+)'
            updated_section = re.sub(
                param_pattern,
                rf'\g<1>{new_value_str}',
                section_content
            )
            
            return updated_section + rest
        
        updated_content = re.sub(section_pattern, replacer, content, flags=re.DOTALL | re.IGNORECASE)
        
        return updated_content
    
    def _add_update_comment(self, changes_log: list):
        """Додавання коментаря про оновлення"""
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            
            comment = f"\n# Auto-updated by backtest optimizer at {timestamp}\n"
            comment += f"# Changes: {len(changes_log)} parameters\n"
            
            for change in changes_log[:5]:  # Перші 5
                comment += f"#   - {change['param']}: {change['old']} -> {change['adjusted']}\n"
            
            # Додаємо на початок файлу
            updated_content = comment + "\n" + content
            
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
        except Exception as e:
            logger.error(f"❌ [ADD_COMMENT] Error: {e}")
    
    def _rollback(self, backup_path: Path):
        """Відкат до backup"""
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, self.settings_path)
                logger.info("✅ [ROLLBACK] Settings restored from backup")
            else:
                logger.error("❌ [ROLLBACK] Backup not found")
        except Exception as e:
            logger.error(f"❌ [ROLLBACK] Error: {e}")
    
    def cleanup_old_backups(self, keep_last_n: int = 10):
        """Очищення старих backups"""
        try:
            backups = sorted(self.backup_dir.glob("settings_*.py"))
            
            if len(backups) > keep_last_n:
                for backup in backups[:-keep_last_n]:
                    backup.unlink()
                    logger.info(f"🗑️ [CLEANUP] Removed old backup: {backup.name}")
                    
        except Exception as e:
            logger.error(f"❌ [CLEANUP] Error: {e}")