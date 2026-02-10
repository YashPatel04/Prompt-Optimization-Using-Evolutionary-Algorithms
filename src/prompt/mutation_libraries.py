from typing import List, Dict

class MutationLibraries:
    """Predefined mutation templates and transformations"""
    
    # Instruction rephrasings (16 templates - expanded)
    INSTRUCTION_TEMPLATES = {
        0: "Classify the sentiment of this text",
        1: "Determine whether this text is positive, negative, or neutral",
        2: "Analyze the emotional tone of the following text",
        3: "Identify the sentiment expressed in this passage",
        4: "What is the sentiment of this text?",
        5: "Categorize the sentiment: positive, negative, or neutral",
        6: "Evaluate the polarity of this text as positive, negative, or neutral",
        7: "Assess the emotional valence of the following statement",
        8: "Label the sentiment conveyed in this text",
        9: "Determine the emotional polarity: positive, negative, or neutral",
        10: "What emotional tone does this text convey?",
        11: "Classify whether this sentiment is favorable or unfavorable",
        12: "Categorize the emotional stance expressed here",
        13: "Judge the sentiment: is this text positive, negative, or neutral?",
        14: "Extract the sentiment polarity from this text",
        15: "Determine if this text expresses positive, negative, or neutral sentiment",
    }
    
    # Reasoning/CoT prompts (13 templates - expanded)
    REASONING_TEMPLATES = {
        0: "Think step by step before answering.",
        1: "Consider the emotional indicators carefully.",
        2: "First identify key words, then classify.",
        3: "Let's approach this systematically.",
        4: "Analyze the context and tone before deciding.",
        5: "Break down the text and examine emotional cues.",
        6: "Consider both explicit and implicit emotional markers.",
        7: "Evaluate the language for emotional intensity.",
        8: "Step through your reasoning: what emotions are expressed?",
        9: "Examine the adjectives and verbs for emotional content.",
        10: "Work through the sentiment indicators logically.",
        11: "Identify contrasts between stated and implied sentiment.",
        12: "Consider cultural and linguistic context clues.",
    }
    
    # Role specifications (12 templates - expanded)
    ROLE_TEMPLATES = {
        0: "You are an expert sentiment analyst.",
        1: "You are a natural language understanding specialist.",
        2: "As a professional text analyst,",
        3: "You are skilled in detecting emotional nuances.",
        4: "You are an experienced emotion detection AI.",
        5: "Act as a sentiment classification expert.",
        6: "You are proficient in analyzing emotional expression.",
        7: "As a linguistic emotion expert,",
        8: "You have expertise in identifying emotional polarity.",
        9: "You are an advanced text sentiment classifier.",
        10: "Assume the role of a sentiment analysis specialist.",
        11: "You are trained to understand emotional language.",
    }
    
    # Output format specifications (15 templates - expanded)
    OUTPUT_FORMATS = {
        0: "",  # no specification
        1: "Respond with only the label: positive, negative, or neutral.",
        2: "Return output in JSON format with 'sentiment' and 'confidence' fields.",
        3: "Provide your classification with a brief explanation.",
        4: "[LABEL] explanation of your reasoning",
        5: "Classify, then explain your reasoning in one sentence.",
        6: "Output format: [SENTIMENT] with confidence score.",
        7: "Provide the classification and a brief justification.",
        8: "Answer with just the sentiment label, nothing else.",
        9: "Output: sentiment classification followed by reasoning.",
        10: "Return the sentiment in all caps: POSITIVE, NEGATIVE, or NEUTRAL.",
        11: "Format your response as: Sentiment: [LABEL]",
        12: "Provide structured output: Classification | Confidence | Reasoning",
        13: "Give sentiment label and explanation in bullet format.",
        14: "Output sentiment with supporting evidence from the text.",
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
    
    # Synonyms for key terms (greatly expanded - 16 words with multiple alternatives)
    SYNONYMS = {
        "classify": ["categorize", "determine", "identify", "label", "sort", "tag", "mark", "assign", "organize"],
        "sentiment": ["emotional tone", "feeling", "mood", "attitude", "opinion", "emotion", "polarity", "stance", "expression"],
        "text": ["passage", "content", "statement", "message", "input", "sentence", "phrase", "document", "material"],
        "analyze": ["examine", "study", "assess", "evaluate", "review", "inspect", "scrutinize", "investigate", "dissect"],
        "positive": ["favorable", "good", "optimistic", "upbeat", "affirmative", "constructive", "approving", "praising"],
        "negative": ["unfavorable", "bad", "pessimistic", "critical", "adverse", "disparaging", "disapproving", "condemning"],
        "neutral": ["objective", "impartial", "balanced", "measured", "factual", "unbiased", "detached"],
        "emotion": ["feeling", "sentiment", "affect", "state", "reaction", "response", "impression"],
        "express": ["convey", "communicate", "display", "manifest", "show", "demonstrate", "articulate", "indicate"],
        "understand": ["comprehend", "grasp", "perceive", "recognize", "discern", "interpret", "decode"],
        "carefully": ["attentively", "meticulously", "thoroughly", "diligently", "rigorously", "precisely", "cautiously"],
        "consider": ["examine", "ponder", "reflect", "contemplate", "weigh", "think about", "take into account"],
        "determine": ["decide", "ascertain", "establish", "find", "resolve", "figure out", "conclude"],
        "important": ["critical", "essential", "significant", "vital", "key", "crucial", "paramount"],
        "word": ["term", "expression", "vocabulary", "language", "phrase", "diction"],
        "meaning": ["definition", "sense", "significance", "interpretation", "implication", "connotation"],
    }
    
    # Example addition templates (9 templates - expanded)
    EXAMPLE_TEMPLATES = [
        '- "I love this!" -> positive\n- "This is awful." -> negative\n- "The sky is blue." -> neutral',
        '- "Best day ever!" -> positive\n- "Worst experience." -> negative\n- "Temperature is 72°F." -> neutral',
        '- Positive: "Excellent work!"\n- Negative: "Disappointing."\n- Neutral: "It rained today."',
        '- "Amazing product, highly recommended!" -> positive\n- "Total waste of money." -> negative\n- "The store is open at 9am." -> neutral',
        '- "I\'m so happy!" -> positive\n- "This is terrible." -> negative\n- "Today is Tuesday." -> neutral',
        '- "Great job, I\'m impressed!" -> positive\n- "Horrible experience, never again." -> negative\n- "The meeting is at 3pm." -> neutral',
        '- "Fantastic! Love it!" -> positive\n- "Absolutely dreadful." -> negative\n- "It weighs 2 kilograms." -> neutral',
        '- "Outstanding service!" -> positive\n- "Completely useless." -> negative\n- "The color is blue." -> neutral',
        '- "Perfect! Exactly what I wanted!" -> positive\n- "Completely disappointed." -> negative\n- "The store has 5 employees." -> neutral',
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