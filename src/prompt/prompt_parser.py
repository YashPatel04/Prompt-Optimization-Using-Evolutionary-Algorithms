from typing import Dict, List
import re
class PromptParser:
    """Parse prompts into functional components for mutation"""

    @staticmethod
    def parse(prompt):
        """
        Parse prompt into structured components.

        Returns:
            Dict with keys:
            - instruction: main task description
            - role: role/persona specification
            - examples: few-shot examples
            - output_spec: output format specification
            - reasoning: chain-of-thought indicators
            - input_marker: placeholder for actual input
            - contraints: any contraints/requirements
        """

        components = {
            'instruction': '',
            'role': '',
            'examples': [],
            'output_spec': '',
            'reasoning': '',
            'input_marker': '{input}',
            'constraints': '',
            'raw': prompt
        }

        # Extract role (if starts with "You are" or "As a")
        role_match = re.search(r'(You are|As a)[^.\n]+\.', prompt, re.IGNORECASE)
        if role_match:
            components['role'] = role_match.group(0)

        # Extract examples (lines starting with -)
        example_pattern = r'- "([^"]+)"\s*->\s*(\w+)'
        examples = re.findall(example_pattern, prompt)
        components['examples'] = examples

        # Extract main instruction (usually first substantive sentence)
        lines = prompt.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('-') and not line.startswith('You') and len(line) > 10:
                components['instruction'] = line.strip()
                break
        
        # Extract reasoning indicators
        reasoning_keywords = ['step by step', "let's", 'think', 'analyze', 'consider']
        reasoning_found = [kw for kw in reasoning_keywords if kw in prompt.lower()]
        components['reasoning'] = 'present' if reasoning_found else 'absent'

        # Extract constraints (usually contain keywords like "must", "should", "only")
        constraint_pattern = r'(must|should|only|always|never)[^.]*\.'
        constraints = re.findall(constraint_pattern, prompt, re.IGNORECASE)
        components['constraints'] = ' '.join(constraints) if constraints else ''

        # Detect output format specification
        if 'json' in prompt.lower():
            components['output_spec'] = 'json'
        elif 'labeled' in prompt.lower() or '→' in prompt:
            components['output_spec'] = 'labeled'
        elif 'explain' in prompt.lower() or 'reasoning' in prompt.lower():
            components['output_spec'] = 'explanation'
        else:
            components['output_spec'] = 'simple'
        
        return components