"""
Token Tracking System - Usage Guide

The token tracking system automatically tracks input and output tokens
for each LLM call throughout the experiment.
"""

# INTEGRATION POINTS:

# 1. TokenTracker is initialized in ExperimentRunner.__init__()
#    self.token_tracker = TokenTracker()

# 2. During optimization, tokens are tracked per iteration:
#    - set_iteration(iteration_num) marks start of iteration
#    - add_tokens(input, output, iteration) adds tokens to tracker
#    - is called automatically when evaluator.evaluate_prompt() is called

# 3. Token information is stored in results['token_tracking'] containing:
#    {
#        'total_input_tokens': int,
#        'total_output_tokens': int,
#        'total_tokens': int,
#        'iterations': {
#            0: {'input_tokens': int, 'output_tokens': int, 'total_tokens': int, 'call_count': int},
#            1: {...},
#            ...
#        },
#        'summary': {
#            'total': {...},
#            'total_calls': int,
#            'num_iterations': int,
#            'avg_tokens_per_iteration': float,
#            'avg_tokens_per_call': float,
#            'avg_input_per_call': float,
#            'avg_output_per_call': float
#        }
#    }

# 4. Token usage is logged during optimization:
#    - Initialization tokens are logged after init_complete phase
#    - Per-iteration tokens are logged via debug messages
#    - Final summary is logged after optimization completes

# 5. Token statistics by iteration are available via:
#    token_tracker.get_all_iterations_tokens()  # Dict of all iterations
#    token_tracker.get_iteration_tokens(iteration_num)  # Specific iteration
#    token_tracker.get_summary()  # High-level statistics


# DATA FLOW:

# LLM Interface (ollama generate_with_tokens)
#   ↓
#   Returns (text, input_tokens, output_tokens)
#   ↓
# Evaluator (evaluate_prompt tracks tokens internally)
#   ↓
#   Stored in self.total_input_tokens, self.total_output_tokens
#   ↓
# ExperimentRunner fitness function
#   ↓
#   Captures token delta and calls:
#   token_tracker.add_tokens(new_tokens_input, new_tokens_output, iteration)
#   ↓
# Stored in results['token_tracking']


# EXAMPLE OUTPUT:

"""
TOKEN USAGE SUMMARY
  Total input tokens: 45230
  Total output tokens: 12340
  Total tokens: 57570
  Total LLM calls: 1000
  Average tokens per call: 57.57
  Average input per call: 45.23
  Average output per call: 12.34
"""
