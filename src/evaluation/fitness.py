from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class FitnessCalculator:
    """
    Calculate fitness scores for prompt variants.
    For sentiment classification: lower fitness is better (error-based).
    """
    
    def __init__(self, task='sentiment_classification'):
        """
        Initialize fitness calculator.
        Args:
            task: Type of task ('sentiment_classification', 'text_classification', etc.)
        """
        self.task = task
    
    def calculate_accuracy(self, predictions, ground_truth):
        """
        Calculate accuracy.
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
        
        Returns:
            Accuracy score based fitness score (0-1). lower is better. 
        """
        # Normalize predictions
        predictions = [self._normalize_label(p) for p in predictions]
        ground_truth = [self._normalize_label(g) for g in ground_truth]
        accuracy = accuracy_score(ground_truth, predictions)
        # Return as fitness (lower is better): 1 - accuracy
        return 1 - accuracy
    
    def calculate_f1(self, predictions, ground_truth, average='weighted'):
        """
        Calculate F1 score.
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            average: 'weighted', 'macro', 'micro'
        
        Returns:
            F1-based fitness score (lower is better)
        """
        predictions = [self._normalize_label(p) for p in predictions]
        ground_truth = [self._normalize_label(g) for g in ground_truth]
        f1 = f1_score(ground_truth, predictions, average=average, zero_division=0)
        # Return as fitness: 1 - F1
        return 1 - f1
    
    def calculate_combined_fitness(self, predictions, ground_truth, accuracy_weight=0.6, f1_weight=0.4):
        """
        Calculate weighted combination of accuracy and F1.
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            accuracy_weight: Weight for accuracy (0-1)
            f1_weight: Weight for F1 (0-1)
        Returns:
            Combined fitness score (lower is better)
        """
        accuracy_fitness = self.calculate_accuracy(predictions, ground_truth)
        f1_fitness = self.calculate_f1(predictions, ground_truth)
        # Normalize weights
        total_weight = accuracy_weight + f1_weight
        accuracy_weight /= total_weight
        f1_weight /= total_weight
        combined = accuracy_weight * accuracy_fitness + f1_weight * f1_fitness
        return combined
    
    def _normalize_label(self, label):
        """
        Normalize label for sentiment classification.
        Handles variations like 'positive', 'pos', 'POSITIVE', etc.
        """
        label = label.lower().strip()
        # Sentiment classification normalization
        if 'positive' in label or 'pos' in label or 'good' in label:
            return 'positive'
        elif 'negative' in label or 'neg' in label or 'bad' in label:
            return 'negative'
        elif 'neutral' in label or 'neut' in label or 'medium' in label:
            return 'neutral'
        else:
            # Try to extract first word
            first_word = label.split()[0] if label else 'unknown'
            return first_word
    
    def extract_label_from_response(self, response):
        """
        Extract predicted label from LLM response.
        Args:
            response: Full LLM response text
        Returns:
            Extracted label
        """
        response = response.lower().strip()
        
        # Pattern 1: Extract after "Result:" or "Classification:" or "Sentiment:"
        import re
        
        result_patterns = [
            r'result\s*:\s*(\w+)',
            r'classification\s*:\s*(\w+)',
            r'sentiment\s*:\s*(\w+)',
            r'answer\s*:\s*(\w+)',
            r'output\s*:\s*(\w+)',
        ]
        
        for pattern in result_patterns:
            match = re.search(pattern, response)
            if match:
                label = match.group(1).strip()
                return self._normalize_label(label)
        
        # Pattern 2: Try common direct keywords
        if 'positive' in response:
            return 'positive'
        elif 'negative' in response:
            return 'negative'
        elif 'neutral' in response:
            return 'neutral'
        
        # Pattern 3: Try labeled format: [LABEL]
        if '[' in response and ']' in response:
            start = response.find('[') + 1
            end = response.find(']')
            label = response[start:end].strip()
            return self._normalize_label(label)
        
        # Pattern 4: Try JSON format
        if '{' in response and '}' in response:
            try:
                # Extract first key-value that looks like a sentiment
                if '"sentiment"' in response or "'sentiment'" in response:
                    match = re.search(r'["\']sentiment["\']\s*:\s*["\'](\w+)["\']', response)
                    if match:
                        return self._normalize_label(match.group(1))
            except:
                pass
        
        # Pattern 5: Try to find after arrow (→)
        if '→' in response:
            parts = response.split('→')
            if len(parts) > 1:
                label = parts[-1].strip().split()[0]
                return self._normalize_label(label)
        
        # Fallback: return first word
        first_word = response.split()[0] if response else 'unknown'
        return self._normalize_label(first_word)