import numpy as np
from typing import Dict
from src.prompt.prompt_parser import PromptParser
from src.prompt.build_prompt import PromptBuilder
from src.prompt.mutation_libraries import MutationLibraries
from src.prompt.base_prompt import BasePrompt

class GenomeDecoder:
    """
    Decode genome vector into prompt mutations.
    Core logic: Genome → Mutations → Final Prompt
    """
    
    def __init__(self, base_prompt: str = None):
        """
        Initialize decoder with base prompt.
        
        Args:
            base_prompt: Starting prompt template
        """
        self.base_prompt = base_prompt or BasePrompt.SENTIMENT_SIMPLE
        self.base_components = PromptParser.parse(self.base_prompt)
    
    def decode(self, genome) -> str:
        """
        Decode genome to final prompt.
        
        Args:
            genome: Genome object with vector representation
        
        Returns:
            Final mutated prompt string
        """
        # Start with fresh copy of base components
        components = {
            'instruction': '',
            'role': '',
            'examples': [],
            'output_spec': '',
            'reasoning': '',
            'input_marker': '{input}',
            'constraints': '',
        }
        
        # Extract genome dimensions (order matters!)
        instruction_template_id = int(genome.get_dimension('instruction_template'))
        reasoning_template_id = int(genome.get_dimension('reasoning_template'))
        output_format_id = int(genome.get_dimension('output_format'))
        constraint_strength = genome.get_dimension('constraint_strength')
        role_template_id = int(genome.get_dimension('role_template'))
        synonym_intensity = genome.get_dimension('synonym_intensity')
        example_count_id = int(genome.get_dimension('example_count'))
        add_reasoning = int(genome.get_dimension('add_reasoning')) == 1
        add_role = int(genome.get_dimension('add_role')) == 1
        add_examples = int(genome.get_dimension('add_examples')) == 1
        # Apply mutations in logical order
        
        # 1. Set instruction (BEFORE synonym replacement)
        base_instruction = MutationLibraries.get_instruction(instruction_template_id)
        
        # 2. Apply synonym replacement to instruction
        if synonym_intensity > 0.1:
            components['instruction'] = MutationLibraries.apply_synonym_replacement(
                base_instruction,
                intensity=synonym_intensity
            )
        else:
            components['instruction'] = base_instruction
        
        # 3. Add role if enabled
        if add_role:
            components['role'] = MutationLibraries.get_role(role_template_id)
        else:
            components['role'] = ''
        
        # 4. Add reasoning if enabled
        if add_reasoning:
            components['reasoning'] = MutationLibraries.get_reasoning(reasoning_template_id)
        else:
            components['reasoning']=''
        
        # 5. Set output format (THIS WAS MISSING IN BUILDER!)
        components['output_spec'] = MutationLibraries.get_output_format(output_format_id)
        
        # 6. Add constraint language based on strength
        components['constraints'] = MutationLibraries.get_constraint_level(constraint_strength)
        
        # 7. Add examples if enabled
        if add_examples:
            example_template = MutationLibraries.EXAMPLE_TEMPLATES[example_count_id % len(MutationLibraries.EXAMPLE_TEMPLATES)]
            components['examples_text'] = example_template  # Store raw template text
        else:
            components['examples_text'] = ''
        
        # Reconstruct prompt
        final_prompt = PromptBuilder.build(components)
        
        return final_prompt
    
    def decode_batch(self, genomes: list) -> list:
        """
        Decode multiple genomes to prompts.
        
        Args:
            genomes: List of Genome objects
        
        Returns:
            List of prompt strings
        """
        return [self.decode(genome) for genome in genomes]
    
    def get_mutation_explanation(self, genome) -> Dict:
        """
        Generate human-readable explanation of mutations applied by genome.
        
        Args:
            genome: Genome object
        
        Returns:
            Dictionary with mutation explanations
        """
        instruction_id = int(genome.get_dimension('instruction_template'))
        reasoning_id = int(genome.get_dimension('reasoning_template'))
        role_id = int(genome.get_dimension('role_template'))
        output_id = int(genome.get_dimension('output_format'))
        example_id = int(genome.get_dimension('example_count'))
        
        add_reasoning = int(genome.get_dimension('add_reasoning')) == 1
        add_role = int(genome.get_dimension('add_role')) == 1
        add_examples = int(genome.get_dimension('add_examples')) == 1
        
        explanation = {
            'instruction_template_id': instruction_id,
            'instruction': MutationLibraries.get_instruction(instruction_id),
            'has_role': add_role,
            'role_id': role_id,
            'role': MutationLibraries.get_role(role_id) if add_role else None,
            'has_reasoning': add_reasoning,
            'reasoning_id': reasoning_id,
            'reasoning': MutationLibraries.get_reasoning(reasoning_id) if add_reasoning else None,
            'output_format_id': output_id,
            'output_format': MutationLibraries.get_output_format(output_id),
            'constraint_strength': round(genome.get_dimension('constraint_strength'), 2),
            'constraint_level': MutationLibraries._map_constraint_strength(genome.get_dimension('constraint_strength')),
            'synonym_intensity': round(genome.get_dimension('synonym_intensity'), 2),
            'has_examples': add_examples,
            'example_id': example_id if add_examples else None,
            'example_template': MutationLibraries.EXAMPLE_TEMPLATES[example_id % len(MutationLibraries.EXAMPLE_TEMPLATES)] if add_examples else None,
        }
        
        return explanation