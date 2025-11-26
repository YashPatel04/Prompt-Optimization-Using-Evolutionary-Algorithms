class BasePrompt:
    """Container for base prompt templates"""

    # SIMPLE BASELINE
    SENTIMENT_SIMPLE = """Classify the sentiment of this text as positive, negative or neutral.

Text: {input}
Sentiment:"""

    # STRUCTURED BASELINE WITH EXAMPLES
    SENTIMENT_STRUCTURED="""You are a sentiment classification expert.
Classify the statement of the given text as positive, negative, or neutral.

Examples:
- "I love this product!" -> positive
- "I dont like this. it's awful." -> negative
- "The weather is cloudy." -> neutral

Text:{input}
Classification:"""

    # CHAIN OF THOUGHT BASELINE
    SENTIMENT_COT="""Analyze the sentiment of the following text step by step.

1. Identify emotional keywords.
2. Consider context and intensity.
3. Make your classification.

Text: {input}
Analysis:"""

    TEMPLATES = {
        'simple':SENTIMENT_SIMPLE,
        'structured':SENTIMENT_STRUCTURED,
        'cot':SENTIMENT_COT
    }

    @staticmethod
    def get_template(name):
        """Get prompt template by name."""
        return BasePrompt.TEMPLATES.get(name, BasePrompt.SENTIMENT_SIMPLE)
    
    @staticmethod
    def list_templates():
        """List all the available templates."""
        return list(BasePrompt.TEMPLATES.keys())