import logging
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """
    Centralized logging for the entire system.
    Logs to both file and console.
    """
    
    def __init__(self, 
                 name: str = "SSA-PromptTuning",
                 log_dir: str = "outputs/logs",
                 log_level: int = logging.INFO):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # Fix Windows encoding issue
        if sys.platform == 'win32':
            # Force UTF-8 for Windows console
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    errors='replace'
                )
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    errors='replace'
                )
        
        # Console handler (simple format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        # Set encoding explicitly
        if hasattr(console_handler, 'setEncoding'):
            console_handler.setEncoding('utf-8')
        self.logger.addHandler(console_handler)
        
        # File handler (detailed format) - UTF-8 encoding
        log_file = self.log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def info(self, message: str):
        """Log info message"""
        # Replace checkmarks with [OK] for compatibility
        message = message.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        message = message.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        message = message.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        message = message.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        message = message.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.logger.critical(message)
    
    def section(self, title: str):
        """Log a section header"""
        title = title.replace('✓', '[OK]').replace('✗', '[FAIL]')
        self.info(f"\n{'='*60}")
        self.info(f"  {title}")
        self.info(f"{'='*60}\n")
    
    def get_log_file(self) -> str:
        """Get path to current log file"""
        return str(self.log_file)


# Global logger instance
_logger = None

def get_logger(name: str = "SSA-PromptTuning") -> Logger:
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = Logger(name)
    return _logger