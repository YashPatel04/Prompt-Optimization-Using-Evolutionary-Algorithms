import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
from src.ssa.population import Population
from src.ssa.movement import Movement
from src.ssa.squirrel import Squirrel
from src.genome.genome import GenomeConfig
import json


class SSAOptimizer:
    """
    Squirrel Search Algorithm (SSA) for prompt optimization.
    Metaheuristic optimization adapted for discrete prompt space.
    """
    
    def __init__(self, 
                 population_size=20,
                 max_iterations=50,
                 Gc=1.9,
                 Pdp=0.1,
                 genome_config=None):
        """
        Initialize SSA optimizer.
        
        Args:
            population_size: Number of squirrels
            max_iterations: Maximum iterations
            Gc: Gravitational coefficient (attraction strength)
            Pdp: Predation presence probability (0-1)
            genome_config: Genome configuration
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.Gc = Gc
        self.Pdp = Pdp
        self.genome_config = genome_config or GenomeConfig()
        
        self.population = None
        self.best_squirrel = None
        self.iteration = 0
        self.history = {
            'best_fitness': [],
            'worst_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'improvements': 0
        }
    
    def initialize(self):
        """Initialize population with random squirrels"""
        self.population = Population(self.population_size, self.genome_config)
        self.population.initialize_random()
        self.iteration = 0
        self.history = {
            'best_fitness': [],
            'worst_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'improvements': 0
        }
    
    def optimize(self, 
                 fitness_function: Callable, 
                 early_stopping_patience: int = 10,
                 progress_callback: Optional[Callable] = None,
                 verbose: bool = True):
        """
        Run SSA optimization.
        
        Args:
            fitness_function: Function that takes squirrel and returns fitness (lower is better)
            early_stopping_patience: Stop if no improvement for N iterations
            progress_callback: Optional callback for progress updates
                              Signature: callback(iteration, best_fitness, improved, phase)
            verbose: Whether to print progress (used if no callback provided)
        
        Returns:
            (best_squirrel, history)
        """
        self.initialize()
        
        # Phase 1: Evaluate initial population
        if progress_callback:
            progress_callback(0, None, False, 'init_start')
        
        for i, squirrel in enumerate(self.population.squirrels):
            fitness = fitness_function(squirrel)
            squirrel.update_fitness(fitness, 0)
            if progress_callback:
                progress_callback(i + 1, None, False, 'init_eval')
        
        self.best_squirrel = self.population.get_best_squirrel()
        self._record_iteration()
        
        if progress_callback:
            progress_callback(0, self.best_squirrel.fitness, False, 'init_complete')
        elif verbose:
            print(f"Initial best fitness: {self.best_squirrel.fitness:.6f}")
        
        # Phase 2: Main optimization loop
        no_improvement_count = 0
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            
            if progress_callback:
                progress_callback(iteration, self.best_squirrel.fitness, False, 'iter_start')
            
            # Update positions using SSA movement
            Movement.update_population(
                self.population,
                self.best_squirrel,
                Gc=self.Gc,
                Pdp=self.Pdp,
                iteration=iteration,
                max_iterations=self.max_iterations
            )
            
            # Evaluate population
            for squirrel in self.population.squirrels:
                if not squirrel.evaluated:
                    fitness = fitness_function(squirrel)
                    squirrel.update_fitness(fitness, iteration)
            
            # Update best squirrel
            current_best = self.population.get_best_squirrel()
            improved = False
            
            if current_best.is_better_than(self.best_squirrel):
                self.best_squirrel = current_best.copy()
                no_improvement_count = 0
                improved = True
                self.history['improvements'] += 1
            else:
                no_improvement_count += 1
            
            self._record_iteration()
            
            if progress_callback:
                progress_callback(iteration, self.best_squirrel.fitness, improved, 'iter_complete')
            elif verbose:
                status = "★ Improved!" if improved else ""
                print(f"Iteration {iteration}/{self.max_iterations} | "
                      f"Best: {self.best_squirrel.fitness:.6f} {status}")
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                if progress_callback:
                    progress_callback(iteration, self.best_squirrel.fitness, False, 'early_stop')
                elif verbose:
                    print(f"\nEarly stopping at iteration {iteration} "
                          f"(no improvement for {early_stopping_patience} iterations)")
                break
        
        if progress_callback:
            progress_callback(self.iteration, self.best_squirrel.fitness, False, 'complete')
        
        return self.best_squirrel, self.history
    
    def _record_iteration(self):
        """Record statistics for current iteration"""
        stats = self.population.get_fitness_stats()
        
        self.history['best_fitness'].append(stats['best'])
        self.history['worst_fitness'].append(stats['worst'])
        self.history['mean_fitness'].append(stats['mean'])
        self.history['std_fitness'].append(stats['std'])
    
    def get_best_prompts(self, decoder, top_k=5):
        """
        Get top K best prompts found.
        
        Args:
            decoder: GenomeDecoder to convert genomes to prompts
            top_k: Number of prompts to return
        
        Returns:
            List of (prompt, fitness) tuples
        """
        evaluated = [s for s in self.population.squirrels if s.evaluated]
        evaluated.sort(key=lambda s: s.fitness)
        
        return [(decoder.decode(s.genome), s.fitness) for s in evaluated[:top_k]]
    
    def save_history(self, filepath):
        """Save evolution history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def __str__(self):
        best_fit = self.best_squirrel.fitness if self.best_squirrel else None
        return (f"SSAOptimizer(pop={self.population_size}, "
                f"max_iter={self.max_iterations}, best={best_fit})")