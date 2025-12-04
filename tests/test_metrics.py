"""Unit tests for evaluation metrics."""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import SummarizationMetrics


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SummarizationMetrics()
        
        self.prediction = "The cat sat on the mat."
        self.reference = "A cat was sitting on a mat."
        
        self.predictions = [
            "The cat sat on the mat.",
            "The dog ran in the park."
        ]
        self.references = [
            "A cat was sitting on a mat.",
            "A dog was running in the park."
        ]
    
    def test_rouge_single(self):
        """Test ROUGE with single prediction."""
        scores = self.metrics.compute_rouge(self.prediction, self.reference)
        
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_rouge_batch(self):
        """Test ROUGE with batch predictions."""
        scores = self.metrics.compute_rouge(self.predictions, self.references)
        
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_bleu(self):
        """Test BLEU score."""
        scores = self.metrics.compute_bleu(self.predictions, self.references)
        
        self.assertIn('bleu', scores)
        self.assertGreaterEqual(scores['bleu'], 0.0)
        self.assertLessEqual(scores['bleu'], 100.0)
    
    def test_evaluate_all(self):
        """Test computing all metrics."""
        scores = self.metrics.evaluate_all(
            self.predictions,
            self.references,
            compute_bertscore=False  # Skip BERTScore for speed
        )
        
        # Check all expected metrics are present
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        self.assertIn('bleu', scores)


if __name__ == '__main__':
    unittest.main()
