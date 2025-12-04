"""
Example script for evaluating summarization models.

This script demonstrates how to:
1. Load a test dataset
2. Generate summaries
3. Compute comprehensive evaluation metrics
4. Save results
"""
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CNNDailyMailLoader, XSumLoader
from src.models import TextSummarizer
from src.evaluation import SummarizationMetrics


def evaluate_on_dataset(dataset_name: str = 'cnn_dailymail', num_samples: int = 100):
    """
    Evaluate summarizer on a dataset.
    
    Args:
        dataset_name: Name of dataset ('cnn_dailymail' or 'xsum')
        num_samples: Number of samples to evaluate
    """
    print("=" * 70)
    print(f"Evaluating on {dataset_name.upper()}")
    print("=" * 70)
    
    # 1. Load dataset
    print(f"\n1. Loading {dataset_name} dataset...")
    if dataset_name == 'cnn_dailymail':
        loader = CNNDailyMailLoader(cache_dir=f"./data/{dataset_name}")
    elif dataset_name == 'xsum':
        loader = XSumLoader(cache_dir=f"./data/{dataset_name}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = loader.load(split='test')
    print(f"   Loaded dataset with {len(dataset)} samples")
    print(f"   Evaluating on {num_samples} samples\n")
    
    # 2. Initialize summarizer
    print("2. Initializing summarizer...")
    summarizer = TextSummarizer()
    print()
    
    # 3. Generate summaries
    print("3. Generating summaries...")
    predictions = []
    references = []
    
    for i in range(min(num_samples, len(dataset))):
        if i % 10 == 0:
            print(f"   Progress: {i}/{num_samples}")
        
        sample = loader.get_sample(i)
        article = sample['article']
        reference = sample['summary']
        
        # Generate summary
        prediction = summarizer.summarize(article)
        
        predictions.append(prediction)
        references.append(reference)
    
    print(f"   Generated {len(predictions)} summaries\n")
    
    # 4. Evaluate
    print("4. Computing evaluation metrics...")
    metrics = SummarizationMetrics()
    
    # Compute all metrics
    print("   Computing ROUGE scores...")
    rouge_scores = metrics.compute_rouge(predictions, references)
    
    print("   Computing BLEU scores...")
    bleu_scores = metrics.compute_bleu(predictions, references)
    
    print("   Computing BERTScore (this may take a while)...")
    bert_scores = metrics.compute_bertscore(predictions, references)
    
    # Combine all scores
    all_scores = {**rouge_scores, **bleu_scores, **bert_scores}
    
    # Print results
    metrics.print_scores(all_scores, f"Results on {dataset_name.upper()}")
    
    # 5. Save results
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_{dataset_name}_{timestamp}.json"
    
    results = {
        'dataset': dataset_name,
        'num_samples': num_samples,
        'timestamp': timestamp,
        'metrics': all_scores,
        'samples': [
            {
                'prediction': pred,
                'reference': ref
            }
            for pred, ref in zip(predictions[:5], references[:5])  # Save first 5
        ]
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def main():
    """Run evaluation on multiple datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate summarization model')
    parser.add_argument('--dataset', type=str, default='cnn_dailymail',
                       choices=['cnn_dailymail', 'xsum'],
                       help='Dataset to evaluate on')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    evaluate_on_dataset(args.dataset, args.num_samples)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
