# Quick usage example
from base_prompt import BasePrompt
from prompt_parser import PromptParser
from build_prompt import PromptBuilder
from mutation_libraries import MutationLibraries

# Start with base
base = BasePrompt.get_template('structured')
print("Base prompt:\n", base)

# Parse it
components = PromptParser.parse(base)
print("Parsed components:", components.keys())

# Mutate (apply mutations)
components['instruction'] = MutationLibraries.get_instruction(2)
components['output_spec'] = 'json'

# Rebuild
mutated = PromptBuilder.build(components)
print("Mutated prompt:\n", mutated)