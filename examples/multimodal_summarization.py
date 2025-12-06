import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import N24NewsLoader
from src.models import MultimodalSummarizer


def main():
    print("=" * 70)
    print("Multimodal Summarization Example")
    print("=" * 70)
    
    #load dataset
    print("\n1. Loading N24News dataset...")
    print("   Note: This requires the N24News dataset to be downloaded.")
    print("   Download from: https://github.com/billywzh717/N24News")
    
    try:
        loader = N24NewsLoader(data_dir="./data/n24news")
        dataset = loader.load(split='train')
        
        #find samples with images
        samples_with_images = []
        for i in range(len(loader)):
            sample = loader.get_sample(i)
            if loader.has_images(sample):
                samples_with_images.append(sample)
                if len(samples_with_images) >= 3:
                    break
        
        if not samples_with_images:
            print("   No samples with images found in dataset.")
            print("   Using text-only example instead...")
            use_images = False
        else:
            print(f"   Found {len(samples_with_images)} samples with images\n")
            use_images = True
    
    except FileNotFoundError as e:
        print(f"   Dataset not found: {e}")
        print("   Using text-only example instead...\n")
        use_images = False
        samples_with_images = []
    
    #initialize summarizer
    print("2. Initializing multimodal summarizer...")
    summarizer = MultimodalSummarizer()
    print()
    
    #generate summaries
    print("3. Generating summaries...")
    print("-" * 70)
    
    if use_images:
        for i, sample in enumerate(samples_with_images):
            print(f"\nSample {i+1}:")
            print(f"Article (first 200 chars):\n{sample['article'][:200]}...")
            print(f"\nNumber of images: {len(sample['image_ids'])}")
            
            # Generate multimodal summary
            result = summarizer.summarize(
                text=sample['article'],
                images=sample.get('image_paths', [])
            )
            
            print(f"\nImage Captions:")
            for j, caption in enumerate(result['image_captions']):
                print(f"  {j+1}. {caption}")
            
            print(f"\nText-Only Summary:\n{result['text_summary']}")
            print(f"\nMultimodal Summary:\n{result['multimodal_summary']}")
            print("-" * 70)
    else:
        #use a sample article without images
        sample_article = """
        Scientists have discovered a new species of deep-sea fish in the Pacific Ocean.
        The fish, which glows in the dark, was found at a depth of 3,000 meters.
        Researchers believe this discovery could help us better understand deep-sea ecosystems.
        The team used advanced underwater cameras to capture images of the fish in its natural habitat.
        """
        
        print("\nSample Article:")
        print(sample_article)
        
        result = summarizer.summarize(text=sample_article)
        print(f"\nGenerated Summary:\n{result['text_summary']}")
        print("-" * 70)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
