import numpy as np
from typing import List, Tuple
from src.ssa.squirrel import Squirrel
from src.genome.genome import Genome, GenomeConfig

class Population:
    """
    Manages the squirrel population with different types.
    in SSA: elite (best), exploratory (moderate), foraging (random).
    """

    def __init__(self, size, genome_config=None):
        """
        Initialize the population.

        Args:
            size: total population size.
            genome_config: configuration for genomes.
        """
        self.size = size
        self.genome_config = genome_config or GenomeConfig()
        self.squirrels = []
        self.iteration = 0

    def initialize_random(self):
        """Initialize squirrel population with random squirrels"""
        self.squirrels = []
        for i in range(self.size):
            genome = Genome(config=self.genome_config)
            squirrel_type a = self._assign_squirrel_type(i)
            squirrel = Squirrel(genome, squirrel_type)
            squirrel.iteration_created = self.iteration
            self.squirrels.append(squirrel)

    def _assign_squirrel_type(self, index):
        """
        Assign squirrel type based on position in population.
        Elite: best performers (top 15%)
        Exploratory: moderate performers (middle 35%)
        Foraging: rest (random search)
        """
        elite_count = max(1, int(self.size * 0.15))
        exploratory_count = max(1, int(self.size * 0.35))

        if index < elite_count:
            return 'elite'
        elif index < (elite_count + exploratory_count):
            return 'exploratory'
        else:
            return 'foraging'
        
    def get_best_squirrel(self):
        """Get squirrel with best fitness"""
        evaluated = [s for s in self.squirrels if s.evaluated]
        if not evaluated:
            return self.squirrels[0]
        return min(evaluated, key=lambda s: s.fitness)

    def get_worst_squirrel(self):
        """Get squirrel with worst fitness"""
        evaluated = [s for s in self.squirrels if s.evaluated]
        if not evaluated:
            return self.squirrels[0]
        return max(evaluated, key=lambda s: s.fitness)
    
    def get_squirrels_by_type(self, squirrel_type):
        """Get all squirrels of specefic type"""
        return [s for s in self.squirrels if s.squirrel_type == squirrel_type]
    
    def sort_by_fitness(self):
        """Sort the squirrel population by fitness"""
        evaluated = [s for s in self.squirrels if s.evaluated]
        unevaluated = [s for s in self.squirrels if not s.evaluated]
        evaluated.sort(key=lambda s: s.fitness)
        self.squirrels = evaluated + unevaluated

    def get_fitness_stats(self):
        """Get population fitness statistics"""
        evaluated = [s for s in self.squirrels if s.evaluated]

        if not evaluated:
            return {
                'best': None,
                'worst': None,
                'mean': None,
                'std': None,
                'evaluated_count': 0
            }

        fitnesses = [s.fitness for s in evaluated]

        return {
                'best': min(fitnesses),
                'worst': max(fitnesses),
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'evaluated_count': len(evaluated)
            }
    
    def __len__(self) -> int:
        return len(self.squirrels)
    
    def __getitem__(self, index: int) -> Squirrel:
        return self.squirrels[index]
    
