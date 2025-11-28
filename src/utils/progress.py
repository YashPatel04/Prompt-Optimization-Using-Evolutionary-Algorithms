import sys
import time
from datetime import datetime, timedelta
from typing import Optional


class ProgressBar:
    """Modern CLI progress bar with live updates (no logging)"""
    
    def __init__(self, total: int, title: str = "Progress"):
        self.total = total
        self.current = 0
        self.title = title
        self.start_time = time.time()
        self.bar_length = 40
    
    def update(self, amount: int = 1):
        """Update progress"""
        self.current = min(self.current + amount, self.total)
        self._display()
    
    def set(self, current: int):
        """Set progress to specific value"""
        self.current = min(current, self.total)
        self._display()
    
    def _display(self):
        """Display progress bar"""
        progress = self.current / self.total
        filled = int(self.bar_length * progress)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        
        percent = 100 * progress
        
        # Calculate time
        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta = self._format_time(remaining)
            elapsed_str = self._format_time(elapsed)
        else:
            eta = "N/A"
            elapsed_str = "0s"
        
        # Display
        output = (f"\r{self.title} |{bar}| {percent:6.2f}% "
                 f"[{self.current}/{self.total}] "
                 f"Elapsed: {elapsed_str} | ETA: {eta}")
        
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def finish(self):
        """Mark as complete"""
        self.current = self.total
        self._display()
        print()  # New line
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


class Spinner:
    """Animated loading spinner"""
    
    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'line': ['-', '\\', '|', '/'],
        'dots2': ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'],
        'dots3': ['⠋', '⠙', '⠚', '⠞', '⠖', '⠦', '⠴', '⠲', '⠳', '⠓'],
        'dots4': ['⠄', '⠆', '⠇', '⠋', '⠙', '⠸', '⠰', '⠠', '⠰', '⠸', '⠙', '⠋', '⠇', '⠆'],
        'bounce': ['⠁', '⠂', '⠄', '⠂'],
    }
    
    def __init__(self, message: str = "Loading", spinner_type: str = 'dots'):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.frame_index = 0
        self.running = False
        self.start_time = time.time()
    
    def start(self):
        """Start spinner"""
        self.running = True
        self.start_time = time.time()
        self._display()
    
    def update(self, message: Optional[str] = None):
        """Update spinner message"""
        if message:
            self.message = message
        if self.running:
            self._display()
    
    def stop(self, final_message: str = "Done"):
        """Stop spinner"""
        self.running = False
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        print(f"\r✓ {final_message} ({elapsed_str})")
    
    def _display(self):
        """Display spinner"""
        frame = self.spinner_frames[self.frame_index % len(self.spinner_frames)]
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        output = f"\r{frame} {self.message} ({elapsed_str})"
        sys.stdout.write(output)
        sys.stdout.flush()
        
        self.frame_index += 1
        time.sleep(0.1)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        else:
            return f"{int(seconds/3600)}h"


class IterationTracker:
    """Track iterations with live stats"""
    
    def __init__(self, total_iterations: int):
        self.total = total_iterations
        self.current = 0
        self.start_time = time.time()
        self.iteration_times = []
        self.best_fitness = float('inf')
        self.improvements = 0
    
    def start_iteration(self, iteration_num: int):
        """Start new iteration"""
        self.current = iteration_num
        self.iteration_start = time.time()
    
    def end_iteration(self, fitness: float, improved: bool = False):
        """End iteration and update stats"""
        elapsed = time.time() - self.iteration_start
        self.iteration_times.append(elapsed)
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.improvements += 1
        
        self._display(fitness, improved)
    
    def _display(self, fitness: float, improved: bool):
        """Display iteration stats"""
        progress = self.current / self.total
        filled = int(30 * progress)
        bar = '█' * filled + '░' * (30 - filled)
        
        percent = 100 * progress
        
        # Calculate time
        avg_time = sum(self.iteration_times) / len(self.iteration_times)
        remaining_iters = self.total - self.current
        eta_seconds = avg_time * remaining_iters
        eta = self._format_time(eta_seconds)
        
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # Status indicator
        status = "✓ Improved" if improved else "  No change"
        
        output = (f"\r[{bar}] Iteration {self.current}/{self.total} ({percent:5.1f}%) | "
                 f"Fitness: {fitness:.6f} | {status} | "
                 f"Elapsed: {elapsed_str} | ETA: {eta}")
        
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def finish(self):
        """Finish tracking"""
        print()  # New line
        print(f"Optimization complete! Improvements: {self.improvements}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


class StatusMessage:
    """Single line status message that updates in place"""
    
    def __init__(self, initial_message: str = ""):
        self.message = initial_message
        if initial_message:
            self._display()
    
    def update(self, message: str):
        """Update status message"""
        self.message = message
        self._display()
    
    def _display(self):
        """Display message"""
        sys.stdout.write(f"\r{self.message:<100}")
        sys.stdout.flush()
    
    def done(self, final_message: str = "Done"):
        """Print final message and newline"""
        print(f"\r{final_message}")