import sys
from pathlib import Path

#add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CNNDailyMailLoader
from src.rag import QASystem


def main():
    """Run RAG Q&A example."""
    print("=" * 70)
    print("RAG-based Question Answering Example")
    print("=" * 70)
    
    #initialize QA system
    print("\n1. Initializing QA system...")
    qa_system = QASystem(backend="faiss")
    print()
    
    #load and index articles
    print("2. Loading and indexing articles...")
    loader = CNNDailyMailLoader(cache_dir="./data/cnn_dailymail")
    dataset = loader.load(split='validation')
    
    #index first 10 articles
    num_articles = 10
    print(f"   Indexing {num_articles} articles...")
    
    for i in range(num_articles):
        sample = loader.get_sample(i)
        qa_system.index_article(
            text=sample['article'],
            article_id=sample['id'],
            metadata={'summary': sample['summary']}
        )
    
    print(f"   Indexed {num_articles} articles\n")
    
    #ask questions
    print("3. Asking questions...")
    print("-" * 70)
    
    questions = [
        "What are the main topics discussed in the articles?",
        "Are there any mentions of political events?",
        "What technology-related news is covered?"
    ]
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        #get answer with sources
        result = qa_system.answer_question(question, top_k=3, return_sources=True)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nRelevant Sources:")
        for j, source in enumerate(result['sources']):
            print(f"\n  Source {j+1}:")
            print(f"  {source['document'][:150]}...")
            if 'article_id' in source['metadata']:
                print(f"  Article ID: {source['metadata']['article_id']}")
        
        print("-" * 70)
    
    #retrieval example
    print("\n4. Retrieval example...")
    print("-" * 70)
    
    query = "sports and athletics"
    print(f"\nQuery: {query}")
    
    retrieved = qa_system.retrieve(query, top_k=3)
    print(f"\nRetrieved {len(retrieved)} relevant passages:")
    
    for i, result in enumerate(retrieved):
        print(f"\n  Passage {i+1}:")
        print(f"  {result['document'][:200]}...")
        print(f"  Distance: {result['distance']:.4f}")
    
    print("-" * 70)
    
    #save index
    print("\n5. Saving vector store...")
    qa_system.save()
    print("   Vector store saved successfully")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
