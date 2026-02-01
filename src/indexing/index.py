"""Embeddings and FAISS indexing functionality"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from pathlib import Path

from ..core.interfaces import Chunk, RetrievalResult
from ..core.io import DataLoader

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Abstract embedding provider interface"""
    
    def __init__(self, embed_function: Callable[[List[str]], np.ndarray]):
        """
        Initialize with embedding function
        
        Args:
            embed_function: Function that takes list of texts and returns embeddings array
        """
        self.embed_function = embed_function
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        return self.embed_function(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for single query
        
        Args:
            query: Query text to embed
            
        Returns:
            Numpy array of embedding with shape (embedding_dim,)
        """
        embeddings = self.embed_function([query])
        return embeddings[0]


class LocalFAISSIndex:
    """Local FAISS-based vector index for retrieval"""
    
    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunk_ids = []
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "flat":
            # Simple L2 distance index
            self.index = self.faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "ivf":
            # IVF (Inverted File) index for larger datasets
            quantizer = self.faiss.IndexFlatL2(self.embedding_dim)
            # Use sqrt(N) clusters as a heuristic, minimum 10
            n_clusters = max(10, int(np.sqrt(1000)))  # Default for 1M docs
            self.index = self.faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) index
            self.index = self.faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created FAISS {self.index_type} index with dimension {self.embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Numpy array of embeddings (n_docs, embedding_dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk IDs")
        
        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {len(self.chunk_ids)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search index for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            
        Returns:
            List of (chunk_id, distance) tuples, sorted by distance (ascending)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []
        
        # Ensure query is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunk_ids)))
        
        # Convert to list of (chunk_id, distance) tuples
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices (FAISS may return -1 for missing results)
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(distance)))
        
        return results
    
    def save(self, file_path: str):
        """
        Save index to file
        
        Args:
            file_path: Path to save index file
        """
        data_loader = DataLoader()
        data_loader.save_faiss_index(self.index, file_path)
        
        # Also save chunk IDs mapping
        chunk_ids_path = str(Path(file_path).with_suffix('.chunk_ids.npy'))
        np.save(chunk_ids_path, np.array(self.chunk_ids))
        
        logger.info(f"Saved FAISS index and chunk IDs to {file_path}")
    
    def load(self, file_path: str):
        """
        Load index from file
        
        Args:
            file_path: Path to index file
        """
        data_loader = DataLoader()
        self.index = data_loader.load_faiss_index(file_path)
        
        # Load chunk IDs mapping
        chunk_ids_path = str(Path(file_path).with_suffix('.chunk_ids.npy'))
        self.chunk_ids = np.load(chunk_ids_path).tolist()
        
        logger.info(f"Loaded FAISS index with {len(self.chunk_ids)} chunks from {file_path}")


class RAGRetriever:
    """Complete RAG retrieval system combining embeddings and FAISS"""
    
    def __init__(
        self, 
        embedding_provider: EmbeddingProvider,
        docstore_path: Optional[str] = None,
        index_path: Optional[str] = None
    ):
        """
        Initialize RAG retriever
        
        Args:
            embedding_provider: Provider for generating embeddings
            docstore_path: Path to docstore.parquet file (optional)
            index_path: Path to existing FAISS index (optional)
        """
        self.embedding_provider = embedding_provider
        self.index = None
        self.docstore = None
        
        # Load docstore if provided
        if docstore_path:
            self.load_docstore(docstore_path)
        
        # Load index if provided
        if index_path:
            self.load_index(index_path)
    
    def load_docstore(self, docstore_path: str):
        """Load docstore mapping chunk_id -> text/metadata"""
        data_loader = DataLoader()
        self.docstore = data_loader.load_docstore(docstore_path)
        logger.info(f"Loaded docstore with {len(self.docstore)} chunks")
    
    def load_index(self, index_path: str):
        """Load pre-built FAISS index"""
        # Determine embedding dimension from first embedding (if docstore available)
        if self.docstore is not None and len(self.docstore) > 0:
            # Get embedding dimension by embedding a sample text
            sample_text = self.docstore.iloc[0]['text']
            sample_embedding = self.embedding_provider.embed_query(sample_text)
            embedding_dim = len(sample_embedding)
        else:
            # Default dimension (will be overridden when index is loaded)
            embedding_dim = 768
        
        self.index = LocalFAISSIndex(embedding_dim)
        self.index.load(index_path)
    
    def build_index(
        self, 
        chunks: List[Chunk], 
        save_path: Optional[str] = None,
        docstore_save_path: Optional[str] = None
    ):
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of text chunks to index
            save_path: Optional path to save index
            docstore_save_path: Optional path to save docstore
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")
        
        # Extract texts and generate embeddings
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_provider.embed_texts(texts)
        
        # Create index
        embedding_dim = embeddings.shape[1]
        self.index = LocalFAISSIndex(embedding_dim)
        self.index.add_embeddings(embeddings, chunk_ids)
        
        # Create docstore
        import pandas as pd
        docstore_data = []
        for chunk in chunks:
            docstore_data.append({
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'document_id': chunk.document_id,
                'metadata': chunk.metadata or {}
            })
        self.docstore = pd.DataFrame(docstore_data)
        
        # Save if paths provided
        if save_path:
            self.index.save(save_path)
        
        if docstore_save_path:
            data_loader = DataLoader()
            data_loader.save_docstore(self.docstore, docstore_save_path)
        
        logger.info(f"Built index with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for query
        
        Args:
            query: Query text
            k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        if self.docstore is None:
            raise ValueError("Docstore not loaded. Call load_docstore() or build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_query(query)
        
        # Search index
        search_results = self.index.search(query_embedding, k)
        
        # Convert to RetrievalResult objects
        results = []
        for chunk_id, distance in search_results:
            # Look up chunk in docstore
            chunk_row = self.docstore[self.docstore['chunk_id'] == chunk_id]
            if len(chunk_row) == 0:
                logger.warning(f"Chunk {chunk_id} not found in docstore, skipping")
                continue
            
            chunk_data = chunk_row.iloc[0]
            
            # Convert distance to similarity score (lower distance = higher similarity)
            # Using negative distance for simplicity
            similarity_score = -distance
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                similarity_score=similarity_score,
                chunk_text=chunk_data['text'],
                metadata=chunk_data.get('metadata', {})
            )
            results.append(result)
        
        logger.debug(f"Retrieved {len(results)} chunks for query")
        return results


# Convenience functions for common operations
def create_simple_retriever(embed_function: Callable[[List[str]], np.ndarray]) -> RAGRetriever:
    """
    Create a simple retriever with provided embedding function
    
    Args:
        embed_function: Function that takes list of texts and returns embeddings
        
    Returns:
        RAGRetriever instance
    """
    embedding_provider = EmbeddingProvider(embed_function)
    return RAGRetriever(embedding_provider)


def build_index_from_documents(
    documents: List[str],
    embed_function: Callable[[List[str]], np.ndarray],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_index_path: Optional[str] = None,
    save_docstore_path: Optional[str] = None
) -> RAGRetriever:
    """
    Build complete RAG index from documents
    
    Args:
        documents: List of document texts
        embed_function: Embedding function
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        save_index_path: Optional path to save FAISS index
        save_docstore_path: Optional path to save docstore
        
    Returns:
        RAGRetriever with built index
    """
    from ..core.chunking import chunk_text
    
    # Chunk all documents
    all_chunks = []
    for doc_id, doc_text in enumerate(documents):
        chunks = chunk_text(doc_text, document_id=f"doc_{doc_id}",
                           chunk_size=chunk_size, overlap=chunk_overlap)
        all_chunks.extend(chunks)
    
    # Build retriever
    retriever = create_simple_retriever(embed_function)
    retriever.build_index(all_chunks, save_index_path, save_docstore_path)
    
    return retriever