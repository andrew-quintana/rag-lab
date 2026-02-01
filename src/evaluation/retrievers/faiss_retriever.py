"""FAISS-based retrieval implementation"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from ..base.base_retriever import VectorRetriever
from ...core.interfaces import RetrievalResult
from ...core.registry import register_retriever


@register_retriever(
    name="faiss_retriever",
    description="FAISS-based vector similarity retrieval",
    index_type="vector"
)
class FAISSRetriever(VectorRetriever):
    """FAISS-based retrieval implementation"""
    
    def __init__(
        self,
        embedding_function: Callable[[List[str]], np.ndarray],
        index_path: Optional[str] = None,
        docstore_path: Optional[str] = None,
        **config
    ):
        """
        Initialize FAISS retriever
        
        Args:
            embedding_function: Function to generate query embeddings
            index_path: Path to FAISS index file
            docstore_path: Path to document store file
            **config: Additional configuration
        """
        super().__init__(
            index_path=index_path,
            docstore_path=docstore_path,
            **config
        )
        self.embedding_function = embedding_function
        self.index = None
        self.docstore = {}
        self.index_path = index_path
        self.docstore_path = docstore_path
        
        if index_path and docstore_path:
            self._load_index()
    
    def get_name(self) -> str:
        return "faiss_retriever"
    
    def get_description(self) -> str:
        return f"FAISS vector retrieval (index_size={self.get_index_size()})"
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using FAISS similarity search
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieval results sorted by similarity score
        """
        if self.index is None:
            return []
        
        query_embedding = self.embedding_function([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx in self.docstore:
                doc_info = self.docstore[idx]
                results.append(RetrievalResult(
                    chunk_id=doc_info.get('chunk_id', f'doc_{idx}'),
                    text=doc_info.get('text', ''),
                    score=float(score),
                    rank=i + 1,
                    document_id=doc_info.get('document_id'),
                    metadata=doc_info.get('metadata', {})
                ))
        
        return results
    
    def add_documents(
        self, 
        documents: List[str], 
        document_ids: List[str],
        **kwargs
    ) -> None:
        """
        Add documents to FAISS index
        
        Args:
            documents: List of document texts
            document_ids: List of document identifiers
            **kwargs: Additional indexing parameters
        """
        import faiss
        
        if not documents:
            return
        
        embeddings = self.embedding_function(documents)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        start_idx = len(self.docstore)
        self.index.add(embeddings.astype('float32'))
        
        for i, (doc, doc_id) in enumerate(zip(documents, document_ids)):
            idx = start_idx + i
            self.docstore[idx] = {
                'chunk_id': doc_id,
                'text': doc,
                'document_id': doc_id,
                'metadata': kwargs.get('metadata', {})
            }
    
    def get_index_size(self) -> int:
        """Get number of documents in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def _load_index(self):
        """Load FAISS index and docstore from disk"""
        try:
            import faiss
            import pandas as pd
            
            if self.index_path:
                self.index = faiss.read_index(self.index_path)
            
            if self.docstore_path:
                df = pd.read_parquet(self.docstore_path)
                self.docstore = {}
                for _, row in df.iterrows():
                    self.docstore[row['index']] = {
                        'chunk_id': row['chunk_id'],
                        'text': row['text'],
                        'document_id': row.get('document_id'),
                        'metadata': row.get('metadata', {})
                    }
        except Exception as e:
            print(f"Warning: Could not load index/docstore: {e}")
    
    def save_index(self, index_path: str, docstore_path: str):
        """Save FAISS index and docstore to disk"""
        try:
            import faiss
            import pandas as pd
            
            if self.index is not None:
                faiss.write_index(self.index, index_path)
            
            if self.docstore:
                rows = []
                for idx, doc_info in self.docstore.items():
                    rows.append({
                        'index': idx,
                        'chunk_id': doc_info['chunk_id'],
                        'text': doc_info['text'],
                        'document_id': doc_info.get('document_id'),
                        'metadata': doc_info.get('metadata', {})
                    })
                
                df = pd.DataFrame(rows)
                df.to_parquet(docstore_path)
                
            self.index_path = index_path
            self.docstore_path = docstore_path
            
        except Exception as e:
            print(f"Error saving index/docstore: {e}")