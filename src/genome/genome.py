import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class GenomeConfig:
    """Configuration for genome structure."""
    dimensions: int = 10
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


    def __post_init__(self):
        if self.ranges=={}:
            self.ranges = {
                'instruction_template': (0, 15.99),     # Will int to 0-15 (16 templates)
                'add_reasoning': (0, 1.99),             # Will int to 0-1 (TRUE BINARY)
                'reasoning_template': (0, 12.99),       # Will int to 0-12 (13 templates)
                'output_format': (0, 14.99),            # Will int to 0-14 (15 templates)
                'constraint_strength': (0, 1),          # Continuous 0-1
                'add_role': (0, 1.99),                  # Will int to 0-1 (TRUE BINARY)
                'role_template': (0, 11.99),            # Will int to 0-11 (12 templates)
                'synonym_intensity': (0, 1),            # Continuous 0-1
                'add_examples': (0, 1.99),              # Will int to 0-1 (TRUE BINARY)
                'example_count': (0, 8.99),             # Will int to 0-8 (9 templates)
            }

class Genome:
    """
    Represents a prompt variant as a fixed-length vector.
    Each dimension controls a specific mutation aspect.
    """
    def __init__(self, vector=None, config=None):
        """
        Initialize genome.

        Args:
            vector: Numpy array of genome values.
            config: GenomeConfig specifying dimensions and ranges.
        """
        self.config = config or GenomeConfig()
        
        assert self.config.ranges is not None

        if vector is None:
            # Random initialization
            self.vector = self._random_initialize()
        else:
            self.vector = np.array(vector, dtype=float)
            self._validate()
        
        self.fitness = None
        self.evaluated = False
        self.mutation_history = []

    def _random_initialize(self):
        """Initialize genome with random values within specified ranges."""
        vector = np.zeros(self.config.dimensions)

        dimension_names = list(self.config.ranges.keys())
        for i, dim_name in enumerate(dimension_names[:self.config.dimensions]):
            min_val, max_val = self.config.ranges[dim_name]
            vector[i] = np.random.uniform(min_val, max_val)
        
        return vector

    def _validate(self):
        """Ensure genome values are within valid ranges"""
        dimension_names = list(self.config.ranges.keys())
        for i, dim_name in enumerate(dimension_names[:self.config.dimensions]):
            min_val, max_val = self.config.ranges[dim_name]
            self.vector[i] = np.clip(self.vector[i], min_val, max_val)

    def get_dimension(self, name):
        """Get value of specific dimension by name"""
        dimension_names = list(self.config.ranges.keys())
        if name in dimension_names:
            idx = dimension_names.index(name)
            return self.vector[idx]
        raise ValueError(f"Dimension '{name}' not found")

    def set_dimension(self, name, value):
        """Set value of specific dimension by name"""
        dimension_names = list(self.config.ranges.keys())
        if name in dimension_names:
            idx = dimension_names.index(name)
            min_val, max_val = self.config.ranges[name]
            self.vector[idx] = np.clip(value, min_val, max_val)
        else:
            raise ValueError(f"Dimension '{name}' not found")
        
    def random_perturbation(self, perturbation_std=0.2):
        """
        Large random perturbation (escape predation in SSA).
        reset step or kick-out-of-local-minima

        Args:
            perturbation_std: Standard deviation of perturbation
        """
        dimension_names = list(self.config.ranges.keys())
        
        for i in range(self.config.dimensions):
            noise = np.random.normal(0, perturbation_std)
            self.vector[i] += noise
            
            dim_name = dimension_names[i]
            min_val, max_val = self.config.ranges[dim_name]
            self.vector[i] = np.clip(self.vector[i], min_val, max_val)
        
        self.mutation_history.append("random_perturbation")
    
    def copy(self):
        """Create a deep copy of this genome"""
        return Genome(self.vector.copy(), self.config)
    
    def to_dict(self):
        """Convert genome to dictionary representation"""
        dimension_names = list(self.config.ranges.keys())
        return {
            'vector': self.vector.tolist(),
            'dimensions': {
                dimension_names[i]: self.vector[i] 
                for i in range(self.config.dimensions)
            },
            'fitness': self.fitness,
            'evaluated': self.evaluated,
            'mutation_history': self.mutation_history
        }
    
    def __str__(self):
        dimension_names = list(self.config.ranges.keys())
        dims_str = ", ".join(
            f"{dimension_names[i]}={self.vector[i]:.2f}"
            for i in range(min(5, len(self.vector)))
        )

        return f"Genome(fitness={self.fitness}, {dims_str}...)"
    
    def __repr__(self):
        return self.__str__()