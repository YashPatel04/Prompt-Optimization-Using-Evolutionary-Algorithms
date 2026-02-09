"""
Token usage tracker for LLM calls.
Tracks input and output tokens across iterations.
"""

from typing import Dict, List, Optional
from datetime import datetime


class TokenTracker:
    """
    Track input and output tokens for LLM evaluations.
    """
    
    def __init__(self):
        """Initialize token tracker"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.iteration_tokens: Dict[int, Dict] = {}
        self.current_iteration = 0
        self.iteration_call_count: Dict[int, int] = {}
    
    def set_iteration(self, iteration: int):
        """Set current iteration"""
        self.current_iteration = iteration
        if iteration not in self.iteration_tokens:
            self.iteration_tokens[iteration] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'call_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        if iteration not in self.iteration_call_count:
            self.iteration_call_count[iteration] = 0
    
    def add_tokens(self, input_tokens: int, output_tokens: int, iteration: Optional[int] = None):
        """
        Add tokens from an LLM call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            iteration: Iteration number (uses current if not provided)
        """
        if iteration is None:
            iteration = self.current_iteration
        
        # Ensure iteration exists
        if iteration not in self.iteration_tokens:
            self.set_iteration(iteration)
        
        # Update totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Update iteration totals
        self.iteration_tokens[iteration]['input_tokens'] += input_tokens
        self.iteration_tokens[iteration]['output_tokens'] += output_tokens
        self.iteration_tokens[iteration]['total_tokens'] += input_tokens + output_tokens
        self.iteration_tokens[iteration]['call_count'] += 1
        self.iteration_call_count[iteration] += 1
    
    def get_total_tokens(self) -> Dict:
        """Get total token usage"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }
    
    def get_iteration_tokens(self, iteration: int) -> Dict:
        """Get token usage for specific iteration"""
        if iteration not in self.iteration_tokens:
            return {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'call_count': 0
            }
        return self.iteration_tokens[iteration]
    
    def get_all_iterations_tokens(self) -> Dict[int, Dict]:
        """Get token usage for all iterations"""
        return self.iteration_tokens.copy()
    
    def get_summary(self) -> Dict:
        """Get comprehensive token usage summary"""
        total_tokens = self.get_total_tokens()
        
        # Calculate statistics
        iterations_with_data = [it for it in self.iteration_tokens.keys() if it >= 0]
        
        if iterations_with_data:
            avg_tokens_per_iter = total_tokens['total_tokens'] / len(iterations_with_data) if iterations_with_data else 0
            total_calls = sum(it['call_count'] for it in self.iteration_tokens.values())
            avg_tokens_per_call = total_tokens['total_tokens'] / total_calls if total_calls > 0 else 0
        else:
            avg_tokens_per_iter = 0
            avg_tokens_per_call = 0
            total_calls = 0
        
        return {
            'total': total_tokens,
            'total_calls': total_calls,
            'num_iterations': len(iterations_with_data),
            'avg_tokens_per_iteration': avg_tokens_per_iter,
            'avg_tokens_per_call': avg_tokens_per_call,
            'avg_input_per_call': self.total_input_tokens / total_calls if total_calls > 0 else 0,
            'avg_output_per_call': self.total_output_tokens / total_calls if total_calls > 0 else 0
        }
    
    def reset(self):
        """Reset all counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.iteration_tokens = {}
        self.iteration_call_count = {}
        self.current_iteration = 0
    
    def to_dict(self) -> Dict:
        """Convert tracker state to dictionary"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'iterations': self.iteration_tokens.copy(),
            'summary': self.get_summary()
        }
