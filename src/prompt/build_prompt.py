from typing import List, Dict

class PromptBuilder:
    """Build prompt from components (reverse of parser)"""

    @staticmethod
    def build(components):
        """
        Reconstruct prompt from parsed components.

        Args:
            components: Dictionary with the following keys and expected formats:
                - 'role' (str, optional): Role/persona specification.
                  Format: "You are an expert sentiment analyst." or "As a professional text analyzer,"
                  Example: "You are an expert sentiment analyst."
                
                - 'instruction' (str, required): Main task description.
                  Format: Clear, action-oriented instruction string.
                  Example: "Classify the sentiment of this text as positive, negative, or neutral"
                
                - 'reasoning' (str, optional): Indicator for chain-of-thought reasoning.
                  Format: if present adds CoT directive
                  Example: "present" → adds "Think step by step:"
                
                - 'examples' (List[Tuple[str, str]], optional): Few-shot examples.
                  Format: List of tuples where each tuple is (example_text, example_label)
                  Example: [("I love this!", "positive"), ("This is awful.", "negative")]
                
                - 'constraints' (str, optional): Constraints or requirements.
                  Format: String describing constraint(s). Multiple constraints can be space-separated.
                  Example: "Classify with high precision. Double-check your reasoning."
                
                - 'output_spec' (str, optional): Output format specification.
                  Format: One of ['simple', 'json', 'labeled', 'explanation']
                  - 'simple': No specific format instruction
                  - 'json': Return output in JSON format with structured fields
                  - 'labeled': Use format like [LABEL] explanation
                  - 'explanation': Provide explanation with answer
                  Example: "json"
                
                - 'input_marker' (str, optional): Placeholder for actual input text.
                  Format: String placeholder, typically "{input}"
                  Default: "{input}"
                  Example: "{input}" or "{text}" or "[TEXT]"

        Returns:
            str: A reconstructed prompt string with all components assembled in logical order.
                 Structure: Role → Instruction → Reasoning → Examples → Constraints → Output Format → Input Marker
        """

        parts = []

        # Add role if present
        if components.get('role'):
            parts.append(components['role'])
            parts.append('')  # blank line
        
        # Add instruction
        if components.get('instruction'):
            parts.append(components['instruction'])
            parts.append('')
        
        # Add reasoning directive
        if components.get('reasoning'):
            parts.append(components['reasoning'])
            parts.append('')
        
        # Add examples
        if components.get('examples_text'):
            parts.append('Examples:')
            parts.append(components['examples_text'])
            parts.append('')
        
        # Add constraints
        if components.get('constraints'):
            parts.append(f'Constraints: {components["constraints"]}')
            parts.append('')
        
        # Add output format specification
        output_spec = components.get('output_spec', '')
        if output_spec:
            parts.append(f'Output format: {output_spec}')
            parts.append('')
        
        # Add input marker
        input_marker = components.get('input_marker', '{input}')
        parts.append(f'Text: {input_marker}')
        parts.append('Result:')
        
        return '\n'.join(parts)