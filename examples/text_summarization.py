import sys
from pathlib import Path

#add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CNNDailyMailLoader
from src.models import TextSummarizer
from src.evaluation import SummarizationMetrics


def main():
    print("=" * 70)
    print("Text Summarization Example")
    print("=" * 70)
    
    #load dataset
    print("\n1. Loading CNN/DailyMail dataset...")
    loader = CNNDailyMailLoader(cache_dir="./data/cnn_dailymail")
    dataset = loader.load(split='validation')
    
    #get a few samples
    num_samples = 3
    print(f"   Using {num_samples} samples from validation set\n")
    
    #initialize summarizer
    print("2. Initializing text summarizer...")
    summarizer = TextSummarizer()
    print()
    
    #generate summaries
    print("3. Generating summaries...")
    print("-" * 70)
    
    predictions = []
    references = []
    
    for i in range(num_samples):
        sample = loader.get_sample(i)
        article = sample['article']
        reference = sample['summary']
        
        #generate summary
        prediction = summarizer.summarize(article)
        
        predictions.append(prediction)
        references.append(reference)
        
        #print results
        print(f"\nSample {i+1}:")
        print(f"\nArticle (first 200 chars):\n{article[:200]}...")
        print(f"\nReference Summary:\n{reference}")
        print(f"\nGenerated Summary:\n{prediction}")
        print("-" * 70)
    
    #evaluate
    print("\n4. Evaluating summaries...")
    metrics = SummarizationMetrics()
    scores = metrics.evaluate_all(predictions, references, compute_bertscore=False)
    metrics.print_scores(scores, "Evaluation Results")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
