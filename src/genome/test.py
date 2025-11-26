from src.genome.genome import Genome, GenomeConfig
from src.genome.decoder import GenomeDecoder
from src.genome.validator import GenomeValidator

# Create genome config
config = GenomeConfig(dimensions=10)

# Initialize random genome
genome = Genome(config=config)
print("Initial genome:", genome)
print("Genome vector:", genome.vector)

# Decode to prompt
decoder = GenomeDecoder()
prompt = decoder.decode(genome)
print("\nDecoded prompt:")
print(prompt)

# Get mutation explanation
explanation = decoder.get_mutation_explanation(genome)
print("\nMutation explanation:")
for key, value in explanation.items():
    print(f"  {key}: {value}")

# Validate
validator = GenomeValidator(config)
is_valid, errors = validator.validate(genome)
print(f"\nGenome valid: {is_valid}")

# Mutate
genome.mutate(mutation_rate=0.3)
print("\nAfter mutation:", genome)

# Crossover
genome2 = Genome(config=config)
offspring = genome.crossover(genome2)
print("Offspring from crossover:", offspring)