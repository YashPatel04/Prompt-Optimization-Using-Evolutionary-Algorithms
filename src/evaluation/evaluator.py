from typing import List, Dict
from src.evaluation.llm_interface import OllamaInterface
from src.evaluation.fitness import FitnessCalculator
from src.evaluation.cache import PromptCache
from src.ssa.squirrel import Squirrel
import pandas as pd
import time

class Evaluator:
    """
    Evaluate squirrel prompts on a dataset and calculate fitness.
    """
    def __init__(self, llm_interface, fitness_calculator, cache=None):
        """
        Initialize evaluator.
        
        Args:
            llm_interface: Ollama interface
            fitness_calculator: Fitness calculator
            cache: Prompt cache (optional, handles disk caching separately)
        """
        self.llm = llm_interface
        self.fitness_calc = fitness_calculator
        self.cache = cache
        self.eval_times = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def evaluate_prompt(self, prompt: str, 
                       samples: List[Dict],
                       sample_size: int = None) -> float:
        """
        Evaluate prompt on samples and comparing them later with ground truths.
        Disk caching is handled by PromptCache if provided.
        
        Args:
            prompt: Prompt template with {input} placeholder
            samples: List of evaluation samples with 'text' and 'label' keys
            sample_size: Max samples to evaluate
        
        Returns:
            Fitness score (lower is better)
        """
        if sample_size and sample_size < len(samples):
            samples = samples[:sample_size]
        
        # Extract texts from samples
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        
        start_time = time.time()
        
        responses = []
        for text in texts:
            # Check cache first
            full_prompt = prompt.format(input=text)
            
            if self.cache:
                cached = self.cache.get(full_prompt)
                if cached:
                    responses.append(cached)
                    continue
            
            # Generate response with token tracking
            response, input_tokens, output_tokens = self.llm.generate_with_tokens(full_prompt)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            responses.append(response)
            
            # Cache it
            if self.cache:
                self.cache.set(full_prompt, response)
        
        elapsed = time.time() - start_time
        self.eval_times.append(elapsed)
        
        # Extract labels from responses
        predictions = []
        for response in responses:
            label = self.fitness_calc.extract_label_from_response(response)
            predictions.append(label)
        
        # Calculate fitness
        fitness = self.fitness_calc.calculate_combined_fitness(
            predictions,
            labels,
            accuracy_weight=0.6,
            f1_weight=0.4
        )
        
        return fitness
            
    def get_average_eval_time(self) -> float:
        """Get average evaluation time per prompt"""
        if not self.eval_times:
            return 0.0
        return sum(self.eval_times) / len(self.eval_times)
    
    def get_token_stats(self) -> dict:
        """Get token usage statistics"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }