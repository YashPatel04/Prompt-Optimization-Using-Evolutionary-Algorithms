import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """
    Centralized logging for the entire system.
    Logs to both file and console with proper Linux support.
    """
    
    def __init__(self, 
                 name: str = "SSA-PromptTuning",
                 log_dir: str = "outputs/logs",
                 log_level: int = logging.INFO):
        """
        Initialize logger with file and console handlers.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            log_level: Logging level
        """
        self.lg_dir = Path(log_dir)
        self.lg_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.lgr = logging.getLogger(name)
        self.lgr.setLevel(log_level)
        self.lgr.propagate = False
        
        # Remove existing handlers
        self.lgr.handlers = []
        
        # Create log file with timestamp
        lg_file = self.lg_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.lg_file = str(lg_file)
        
        # Detailed formatter for file
        det_fmt = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simple formatter for console
        simp_fmt = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # File handler - write everything to file
        try:
            file_hdlr = logging.FileHandler(
                lg_file,
                encoding='utf-8',
                mode='w'
            )
            file_hdlr.setLevel(logging.DEBUG)  # Capture everything in file
            file_hdlr.setFormatter(det_fmt)
            self.lgr.addHandler(file_hdlr)
        except Exception as e:
            print(f"[WARNING] Could not create file handler: {e}")
        
        # Console handler - show everything (same as file)
        cons_hdlr = logging.StreamHandler(sys.stdout)
        cons_hdlr.setLevel(logging.DEBUG)  # Show all messages on console
        cons_hdlr.setFormatter(simp_fmt)
        self.lgr.addHandler(cons_hdlr)
    
    def info(self, msg: str):
        """Log info message"""
        msg = self._sanitize_msg(msg)
        self.lgr.info(msg)
        self._flush()
    
    def debug(self, msg: str):
        """Log debug message"""
        msg = self._sanitize_msg(msg)
        self.lgr.debug(msg)
        self._flush()
    
    def warning(self, msg: str):
        """Log warning message"""
        msg = self._sanitize_msg(msg)
        self.lgr.warning(msg)
        self._flush()
    
    def error(self, msg: str):
        """Log error message"""
        msg = self._sanitize_msg(msg)
        self.lgr.error(msg)
        self._flush()
    
    def critical(self, msg: str):
        """Log critical message"""
        msg = self._sanitize_msg(msg)
        self.lgr.critical(msg)
        self._flush()
    
    def section(self, title: str):
        """Log a section header"""
        title = self._sanitize_msg(title)
        separator = "="*60
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}\n")
    
    def _flush(self):
        """Flush all handlers to disk"""
        for handler in self.lgr.handlers:
            if hasattr(handler, 'flush'):
                try:
                    handler.flush()
                except Exception:
                    pass
    
    def _sanitize_msg(self, msg: str) -> str:
        """Convert special characters to ASCII for Linux compatibility"""
        if not isinstance(msg, str):
            msg = str(msg)
        
        replacements = {
            '✓': '[OK]',
            '✗': '[FAIL]',
            '█': '#',
            '░': '-',
            '→': '->',
            '★': '*',
            '╔': '+',
            '╚': '+',
            '║': '|',
            '═': '=',
            '⠋': '.',
            '⠙': '.',
            '⠹': '.',
            '⠸': '.',
            '⠼': '.',
            '⠴': '.',
            '⠦': '.',
            '⠧': '.',
            '⠇': '.',
            '⠏': '.',
        }
        for special, ascii_char in replacements.items():
            msg = msg.replace(special, ascii_char)
        return msg
    
    def get_log_file(self) -> str:
        """Get path to current log file"""
        return self.lg_file


# Global logger instance
_lgr_inst = None
_lgr_dir = None

def get_logger(name: str = "SSA-PromptTuning", log_dir: Optional[str] = None) -> Logger:
    """Get or create global logger instance"""
    global _lgr_inst, _lgr_dir
    
    # Reset logger if log_dir changes
    if log_dir is not None and log_dir != _lgr_dir:
        _lgr_inst = None
        _lgr_dir = log_dir
    
    if _lgr_inst is None:
        _lgr_inst = Logger(name, log_dir=log_dir or "outputs/logs")
    return _lgr_inst