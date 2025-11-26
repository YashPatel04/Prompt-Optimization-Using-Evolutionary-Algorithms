import numpy as np
from typing import Tuple, List
from src.genome.genome import GenomeConfig

class GenomeValidator:
    """Validate genome constraints and correctness"""
    
    def __init__(self, config: GenomeConfig | None=None):
        """
        Initialize validator.
        
        Args:
            config: GenomeConfig with valid ranges
        """
        self.config = config or GenomeConfig()
        if self.config.ranges is None:
            raise ValueError("GenomeConfig.ranges must be initialized")

    def validate(self, genome):
        """
        Validate genome structure and values.
        
        Args:
            genome: Genome object to validate
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check vector size
        if len(genome.vector) != self.config.dimensions:
            errors.append(
                f"Genome vector size {len(genome.vector)} != "
                f"config dimensions {self.config.dimensions}"
            )
        
        # Check each dimension is in valid range
        dimension_names = list(self.config.ranges.keys())
        for i, dim_name in enumerate(dimension_names[:self.config.dimensions]):
            min_val, max_val = self.config.ranges[dim_name]
            if genome.vector[i] < min_val or genome.vector[i] > max_val:
                errors.append(
                    f"Dimension '{dim_name}' value {genome.vector[i]} "
                    f"outside range [{min_val}, {max_val}]"
                )
        
        # Check for NaN or inf
        if np.any(np.isnan(genome.vector)):
            errors.append("Genome contains NaN values")
        
        if np.any(np.isinf(genome.vector)):
            errors.append("Genome contains infinite values")
        
        return len(errors) == 0, errors
    
    def repair(self, genome) -> None:
        """
        Repair invalid genome by clipping values to valid ranges.
        
        Args:
            genome: Genome object to repair (modified in-place)
        """
        dimension_names = list(self.config.ranges.keys())
        
        for i, dim_name in enumerate(dimension_names[:self.config.dimensions]):
            min_val, max_val = self.config.ranges[dim_name]
            genome.vector[i] = np.clip(genome.vector[i], min_val, max_val)
        
        # Replace NaN with random value
        nan_mask = np.isnan(genome.vector)
        for i in np.where(nan_mask)[0]:
            dim_name = dimension_names[i]
            min_val, max_val = self.config.ranges[dim_name]
            genome.vector[i] = np.random.uniform(min_val, max_val)
    
    def assert_valid(self, genome) -> None:
        """
        Assert genome is valid, raise exception if not.
        
        Args:
            genome: Genome object to validate
        
        Raises:
            ValueError if genome is invalid
        """
        is_valid, errors = self.validate(genome)
        if not is_valid:
            raise ValueError(f"Invalid genome: {', '.join(errors)}")
    
    def compare_genomes(self, genome1, genome2):
        """
        Calculate Euclidean distance between two genomes.
        
        Args:
            genome1: First genome
            genome2: Second genome
        
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(genome1.vector - genome2.vector)
    
    def genome_diversity(self, genomes: list):
        """
        Calculate average pairwise distance (diversity) of population.
        
        Args:
            genomes: List of Genome objects
        
        Returns:
            Average pairwise distance
        """
        if len(genomes) < 2:
            return 0.0
        
        distances = []
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distances.append(self.compare_genomes(genomes[i], genomes[j]))
        
        return np.mean(distances) if distances else 0.0