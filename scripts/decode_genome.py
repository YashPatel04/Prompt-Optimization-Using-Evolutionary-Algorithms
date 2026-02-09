"""
Standalone script to decode a genome vector to a prompt
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genome.genome import Genome, GenomeConfig
from src.genome.decoder import GenomeDecoder


def decode_genome(genome_vector):
    """
    Decode a genome vector to a prompt.
    
    Args:
        genome_vector: List of 10 float values
        
    Returns:
        Decoded prompt string
    """
    try:
        # Create genome config
        config = GenomeConfig(dimensions=10)
        
        # Create genome object
        genome = Genome(vector=genome_vector, config=config)
        
        # Decode to prompt
        decoder = GenomeDecoder()
        prompt = decoder.decode(genome)
        
        return prompt
    
    except Exception as e:
        return f"[ERROR] Failed to decode: {e}"


def main():
    """Main function"""
    
    # Your genome vector
    genome_vector = [
          5.765059296703247,
          0.5010467686924747,
          2.4812700444030034,
          1.8022610758024504,
          0.2848404943774676,
          0.07340502523552027,
          2.4321616925797884,
          0.5026790232288615,
          0.10244271498747881,
          1.6690923207773025
        ]
    
    print("\n" + "="*80)
    print("GENOME DECODER".center(80))
    print("="*80 + "\n")
    
    # Print genome dimensions with their meanings
    print("Genome Vector (10 dimensions):")
    print("-" * 80)
    
    dimension_names = [
        'instruction_template',
        'add_reasoning',
        'reasoning_template',
        'output_format',
        'constraint_strength',
        'add_role',
        'role_template',
        'synonym_intensity',
        'add_examples',
        'example_count'
    ]
    
    for i, (name, val) in enumerate(zip(dimension_names, genome_vector)):
        print(f"  [{i}] {name:25s} = {val:.6f}")
    
    print()
    
    # Decode the genome
    print("Decoding genome to prompt...")
    prompt = decode_genome(genome_vector)
    
    print("\n" + "="*80)
    print("DECODED PROMPT".center(80))
    print("="*80 + "\n")
    print(prompt)
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()