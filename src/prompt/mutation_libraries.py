from typing import List, Dict

class MutationLibraries:
    """Predefined mutation templates and transformations"""
    
    # Instruction rephrasings
    INSTRUCTION_TEMPLATES = {
        0: "Classify the sentiment of this text",
        1: "Determine whether this text is positive, negative, or neutral",
        2: "Analyze the emotional tone of the following text",
        3: "Identify the sentiment expressed in this passage",
        4: "What is the sentiment of this text?",
        5: "Categorize the sentiment: positive, negative, or neutral",
    }
    
    # Reasoning/CoT prompts
    REASONING_TEMPLATES = {
        0: "Think step by step before answering.",
        1: "Consider the emotional indicators carefully.",
        2: "First identify key words, then classify.",
        3: "Let's approach this systematically.",
        4: "Analyze the context and tone before deciding.",
    }
    
    # Role specifications
    ROLE_TEMPLATES = {
        0: "You are an expert sentiment analyst.",
        1: "You are a natural language understanding specialist.",
        2: "As a professional text analyst,",
        3: "You are skilled in detecting emotional nuances.",
    }
    
    # Output format specifications
    OUTPUT_FORMATS = {
        0: "",  # no specification
        1: "Respond with only the label: positive, negative, or neutral.",
        2: "Return output in JSON format with 'sentiment' and 'confidence' fields.",
        3: "Provide your classification with a brief explanation.",
        4: "[LABEL] explanation of your reasoning",
        5: "Classify, then explain your reasoning in one sentence.",
    }
    
    # Constraint language intensity
    CONSTRAINT_LEVELS = {
        "none": "",  # no constraint
        "low": "Be careful with your classification.",
        "medium": "Classify accurately and carefully.",
        "high": "Classify with high precision. Double-check your reasoning.",
    }
    
    # Constraint markers for extraction (detect existing constraints in prompts)
    CONSTRAINT_MARKERS = {
        "must": "high",
        "should": "medium",
        "carefully": "medium",
        "accurate": "medium",
        "precision": "high",
        "double-check": "high",
        "ensure": "medium",
        "only": "high",
        "always": "high",
        "never": "high",
    }
    
    # Synonyms for key terms
    SYNONYMS = {
        "classify": ["categorize", "determine", "identify", "label", "sort"],
        "sentiment": ["emotional tone", "feeling", "mood", "attitude", "opinion"],
        "text": ["passage", "content", "statement", "message", "input"],
        "analyze": ["examine", "study", "assess", "evaluate", "review"],
        "positive": ["favorable", "good", "optimistic", "upbeat"],
        "negative": ["unfavorable", "bad", "pessimistic", "critical"],
        "neutral": ["objective", "impartial", "balanced"],
    }
    
    # Example addition templates
    EXAMPLE_TEMPLATES = [
        '- "I love this!" -> positive\n- "This is awful." -> negative\n- "The sky is blue." -> neutral',
        '- "Best day ever!" -> positive\n- "Worst experience." -> negative\n- "Temperature is 72°F." -> neutral',
        '- Positive: "Excellent work!"\n- Negative: "Disappointing."\n- Neutral: "It rained today."',
    ]

    @staticmethod
    def _map_constraint_strength(constraint_strength):
        """Map with more granular thresholds"""
        thresholds = {
            (0.0, 0.1): "none",
            (0.1, 0.33): "low",
            (0.33, 0.66): "medium",
            (0.66, 1.0): "high",
        }
        
        for (lower, upper), level in thresholds.items():
            if lower <= constraint_strength < upper:
                return level
        
        return "high"  # Default for edge case

    @staticmethod
    def get_constraint_level(constraint_intensity):
        """Get constraint template by intensity level"""
        return MutationLibraries.CONSTRAINT_LEVELS.get(
            MutationLibraries._map_constraint_strength(constraint_intensity), 
            MutationLibraries.CONSTRAINT_LEVELS["none"]
        )
    
    @staticmethod
    def get_instruction(template_id):
        """Get instruction template by ID"""
        return MutationLibraries.INSTRUCTION_TEMPLATES.get(
            template_id % len(MutationLibraries.INSTRUCTION_TEMPLATES),
            MutationLibraries.INSTRUCTION_TEMPLATES[0]
        )
    
    @staticmethod
    def get_reasoning(template_id):
        """Get reasoning template by ID"""
        return MutationLibraries.REASONING_TEMPLATES.get(
            template_id % len(MutationLibraries.REASONING_TEMPLATES),
            MutationLibraries.REASONING_TEMPLATES[0]
        )
    
    @staticmethod
    def get_role(template_id):
        """Get role template by ID"""
        return MutationLibraries.ROLE_TEMPLATES.get(
            template_id % len(MutationLibraries.ROLE_TEMPLATES),
            MutationLibraries.ROLE_TEMPLATES[0]
        )
    
    @staticmethod
    def get_output_format(format_id):
        """Get output format specification by ID"""
        return MutationLibraries.OUTPUT_FORMATS.get(
            format_id % len(MutationLibraries.OUTPUT_FORMATS),
            MutationLibraries.OUTPUT_FORMATS[0]
        )
    
    @staticmethod
    def apply_synonym_replacement(text: str, intensity):
        """Replace words with synonyms based on intensity (0-1)"""
        words = text.split()
        num_to_replace = int(len(words) * intensity)
        
        import random
        indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))
        
        for idx in indices_to_replace:
            word = words[idx].lower().strip('.,!?')
            if word in MutationLibraries.SYNONYMS:
                words[idx] = random.choice(MutationLibraries.SYNONYMS[word])
        
        return ' '.join(words)