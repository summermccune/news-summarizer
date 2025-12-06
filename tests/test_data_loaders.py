"""Unit tests for data loaders."""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CNNDailyMailLoader, XSumLoader


class TestDataLoaders(unittest.TestCase):
    """Test cases for dataset loaders."""
    
    def test_cnn_dailymail_loader(self):
        """Test CNN/DailyMail loader."""
        loader = CNNDailyMailLoader(cache_dir="./data/test_cache")
        
        # Load a small subset
        dataset = loader.load(split='validation')
        
        # Check dataset is loaded
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)
        
        # Check sample structure
        sample = loader.get_sample(0)
        self.assertIn('article', sample)
        self.assertIn('summary', sample)
        self.assertIn('id', sample)
        
        # Check preprocessing
        text = "This  is   a   test."
        processed = loader.preprocess(text)
        self.assertEqual(processed, "This is a test.")
    
    def test_xsum_loader(self):
        """Test XSum loader."""
        loader = XSumLoader(cache_dir="./data/test_cache")
        
        # Load a small subset
        dataset = loader.load(split='validation')
        
        # Check dataset is loaded
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)
        
        # Check sample structure
        sample = loader.get_sample(0)
        self.assertIn('article', sample)
        self.assertIn('summary', sample)
        self.assertIn('id', sample)


if __name__ == '__main__':
    unittest.main()
