"""Question Answering system using RAG."""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from .vector_store import VectorStore
from ..models.summarizer import TextSummarizer
from ..vision.image_processor import ImageProcessor
from ..config import load_config


class QASystem:
    """
    Question Answering system using Retrieval-Augmented Generation.
    
    Combines retrieval from vector store with generation to answer questions
    about news articles grounded in source material.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        config_path: str = "config.yaml",
        backend: str = "faiss"
    ):
        """
        Initialize the QA system.
        
        Args:
            embedding_model: Model for embeddings
            llm_model: Language model for generation
            persist_directory: Directory for vector store
            config_path: Path to configuration file
            backend: Vector store backend
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize vector store
        print("Initializing vector store...")
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            config_path=config_path,
            backend=backend
        )
        
        # Initialize text generator (using summarizer for now)
        print("Initializing text generator...")
        self.text_generator = TextSummarizer(
            model_name=llm_model,
            config_path=config_path
        )
        
        # Initialize image processor
        print("Initializing image processor...")
        self.image_processor = ImageProcessor(config_path=config_path)
        
        # Configuration
        self.rag_config = self.config['models']['rag']
        self.chunk_size = self.rag_config.get('chunk_size', 512)
        self.top_k = self.rag_config.get('top_k', 5)
        
        print("QA System initialized.")
    
    def index_article(
        self,
        text: str,
        article_id: Optional[str] = None,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Index a news article for retrieval.
        
        Args:
            text: Article text
            article_id: Unique article identifier
            images: Related images
            metadata: Additional metadata
        """
        # Prepare metadata
        meta = metadata or {}
        if article_id:
            meta['article_id'] = article_id
        
        # Process images if provided
        if images:
            captions = self.image_processor.generate_captions(images)
            meta['image_captions'] = captions
            meta['num_images'] = len(images)
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=[text],
            metadata=[meta],
            chunk_size=self.chunk_size
        )
    
    def index_articles(
        self,
        articles: List[Dict[str, Any]]
    ):
        """
        Index multiple articles.
        
        Args:
            articles: List of article dicts with 'text', 'id', 'images', etc.
        """
        for article in articles:
            self.index_article(
                text=article['text'],
                article_id=article.get('id'),
                images=article.get('images'),
                metadata=article.get('metadata')
            )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of relevant passages with metadata
        """
        top_k = top_k or self.top_k
        return self.vector_store.search(query, top_k=top_k)
    
    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Answer a question using RAG.
        
        Args:
            question: Question to answer
            top_k: Number of passages to retrieve
            return_sources: Whether to return source passages
            
        Returns:
            Answer string or dict with answer and sources
        """
        # Retrieve relevant passages
        retrieved = self.retrieve(question, top_k=top_k)
        
        if not retrieved:
            answer = "I don't have enough information to answer this question."
            if return_sources:
                return {'answer': answer, 'sources': []}
            return answer
        
        # Build context from retrieved passages
        context_parts = []
        for i, result in enumerate(retrieved):
            context_parts.append(f"Passage {i+1}: {result['document']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for generation
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer using summarization model
        # Note: For production, you'd want to use a proper QA or instruction-tuned model
        answer = self.text_generator.summarize(
            prompt,
            max_length=150,
            min_length=20
        )
        
        if return_sources:
            return {
                'answer': answer,
                'sources': retrieved,
                'question': question
            }
        
        return answer
    
    def retrieve_with_images(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve passages and related images.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dict with passages and image information
        """
        retrieved = self.retrieve(query, top_k=top_k)
        
        # Extract unique image captions
        all_captions = []
        for result in retrieved:
            if 'image_captions' in result['metadata']:
                all_captions.extend(result['metadata']['image_captions'])
        
        return {
            'passages': retrieved,
            'image_captions': list(set(all_captions))
        }
    
    def save(self):
        """Save the vector store."""
        self.vector_store.save()
    
    def load(self):
        """Load the vector store."""
        self.vector_store.load()
