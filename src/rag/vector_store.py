"""Vector store for document embeddings and retrieval."""
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from ..config import load_config


class VectorStore:
    """
    Vector store for storing and retrieving document embeddings.
    
    Uses sentence transformers to create embeddings and FAISS/ChromaDB
    for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        config_path: str = "config.yaml",
        backend: str = "faiss"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist the vector store
            config_path: Path to configuration file
            backend: Backend to use ('faiss' or 'chroma')
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set embedding model
        model_name = embedding_model or self.config['models']['rag']['embedding_model']
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Set persist directory
        self.persist_directory = Path(persist_directory or self.config['vector_db']['persist_directory'])
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        self.backend = backend
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        if backend == "faiss":
            import faiss
            self.index = None
            self.faiss = faiss
        elif backend == "chroma":
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
            collection_name = self.config['vector_db'].get('collection_name', 'news_articles')
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print(f"Vector store initialized with backend: {backend}")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_size: Optional[int] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            chunk_size: Size of chunks to split documents into
        """
        # Chunk documents if needed
        if chunk_size:
            documents, metadata = self._chunk_documents(documents, metadata, chunk_size)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Store in backend
        if self.backend == "faiss":
            self._add_to_faiss(documents, embeddings, metadata)
        elif self.backend == "chroma":
            self._add_to_chroma(documents, embeddings, metadata)
    
    def _chunk_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]],
        chunk_size: int
    ):
        """Split documents into chunks."""
        chunked_docs = []
        chunked_metadata = []
        
        for i, doc in enumerate(documents):
            # Simple chunking by words
            words = doc.split()
            for j in range(0, len(words), chunk_size):
                chunk = ' '.join(words[j:j+chunk_size])
                chunked_docs.append(chunk)
                
                # Copy metadata and add chunk info
                if metadata and i < len(metadata):
                    meta = metadata[i].copy()
                    meta['chunk_index'] = j // chunk_size
                else:
                    meta = {'chunk_index': j // chunk_size}
                chunked_metadata.append(meta)
        
        return chunked_docs, chunked_metadata
    
    def _add_to_faiss(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]]
    ):
        """Add documents to FAISS index."""
        # Initialize index if needed
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = self.faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
    
    def _add_to_chroma(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]]
    ):
        """Add documents to ChromaDB collection."""
        # Generate IDs
        start_id = len(self.documents)
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadata if metadata else [{}] * len(documents),
            ids=ids
        )
        
        self.documents.extend(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of results with documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in backend
        if self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k)
        elif self.backend == "chroma":
            return self._search_chroma(query_embedding, top_k)
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def _search_chroma(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        return formatted_results
    
    def save(self, filename: str = "vector_store.pkl"):
        """Save the vector store to disk."""
        if self.backend == "faiss":
            save_path = self.persist_directory / filename
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                }, f)
            
            # Save FAISS index
            if self.index is not None:
                index_path = self.persist_directory / "faiss.index"
                self.faiss.write_index(self.index, str(index_path))
            
            print(f"Vector store saved to {save_path}")
        # ChromaDB persists automatically
    
    def load(self, filename: str = "vector_store.pkl"):
        """Load the vector store from disk."""
        if self.backend == "faiss":
            load_path = self.persist_directory / filename
            if load_path.exists():
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                
                # Load FAISS index
                index_path = self.persist_directory / "faiss.index"
                if index_path.exists():
                    self.index = self.faiss.read_index(str(index_path))
                
                print(f"Vector store loaded from {load_path}")
        # ChromaDB loads automatically
