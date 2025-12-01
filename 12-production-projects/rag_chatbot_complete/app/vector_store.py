"""
Vector store wrapper for ChromaDB.

Provides a clean interface for storing and retrieving document embeddings
using ChromaDB as the vector database.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    """
    Vector database for storing and retrieving document embeddings.
    
    Uses ChromaDB for persistent storage and fast similarity search.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._init_client()
        
        # Get or create collection
        self._init_collection()
        
        logger.info(
            f"Initialized VectorStore: "
            f"collection={collection_name}, "
            f"persist_dir={persist_directory}"
        )
    
    def _init_client(self) -> None:
        """Initialize ChromaDB client with persistence."""
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _init_collection(self) -> None:
        """Initialize or get existing collection."""
        # Create OpenAI embedding function
        # Note: Requires OPENAI_API_KEY environment variable
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=None,  # Will use OPENAI_API_KEY env var
            model_name=self.embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "RAG document chunks"}
        )
    
    def add_documents(
        self,
        chunks: List[DocumentChunk]
    ) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Number of chunks added
            
        Example:
            ```python
            store = VectorStore()
            chunks = processor.process_file(Path("doc.pdf"))
            count = store.add_documents(chunks)
            print(f"Added {count} chunks")
            ```
        """
        if not chunks:
            return 0
        
        # Extract data from chunks
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to collection
        # ChromaDB will automatically generate embeddings
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with content, metadata, and scores
            
        Example:
            ```python
            store = VectorStore()
            results = store.search(
                query="What are the features?",
                top_k=3
            )
            for result in results:
                print(result['content'])
                print(result['similarity_score'])
            ```
        """
        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'chunk_id': results['ids'][0][i]
                })
        
        logger.info(f"Search returned {len(formatted_results)} results")
        
        return formatted_results
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks from a document.
        
        Args:
            document_id: Document identifier (source filename)
            
        Returns:
            Number of chunks deleted
            
        Example:
            ```python
            store = VectorStore()
            deleted = store.delete_document("old_doc.pdf")
            print(f"Deleted {deleted} chunks")
            ```
        """
        # Query to find all chunks from this document
        results = self.collection.get(
            where={"source": document_id}
        )
        
        if not results['ids']:
            return 0
        
        # Delete chunks
        self.collection.delete(ids=results['ids'])
        
        count = len(results['ids'])
        logger.info(f"Deleted {count} chunks from document {document_id}")
        
        return count
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        Warning: This permanently deletes all data!
        
        Example:
            ```python
            store = VectorStore()
            store.delete_collection()
            ```
        """
        self.client.delete_collection(name=self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
        
        # Re-initialize collection
        self._init_collection()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
            
        Example:
            ```python
            store = VectorStore()
            stats = store.get_stats()
            print(f"Total chunks: {stats['total_chunks']}")
            ```
        """
        count = self.collection.count()
        
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'persist_directory': str(self.persist_directory),
            'embedding_model': self.embedding_model
        }
    
    def health_check(self) -> Dict:
        """
        Check health of vector store.
        
        Returns:
            Health status dictionary
            
        Example:
            ```python
            store = VectorStore()
            health = store.health_check()
            if health['status'] == 'healthy':
                print("Vector store is healthy!")
            ```
        """
        try:
            # Try to query (quick operation)
            self.collection.count()
            
            return {
                'status': 'healthy',
                'error': None
            }
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# ============================================================================
# Production Considerations
# ============================================================================

"""
Vector Store Best Practices:

1. **Embedding Models**:
   - text-embedding-3-small: Cheaper, faster, good for most cases
   - text-embedding-3-large: Better quality, more expensive
   - text-embedding-ada-002: Older, still good
   
   Cost comparison (per 1M tokens):
   - text-embedding-3-small: $0.02
   - text-embedding-3-large: $0.13
   - text-embedding-ada-002: $0.10

2. **ChromaDB vs Alternatives**:
   - ChromaDB: Easy to use, great for development and small-medium scale
   - Pinecone: Fully managed, scales to billions, but costs money
   - Weaviate: Open source, feature-rich, more complex
   - Milvus: High performance, complex setup
   
   Choose based on:
   - Scale: How many documents?
   - Budget: Managed vs self-hosted
   - Features: Filtering, multi-tenancy, etc.

3. **Persistence**:
   - ChromaDB PersistentClient saves to disk automatically
   - Back up persist_directory regularly
   - Consider S3/GCS for backups in production
   
4. **Performance**:
   - Batch inserts when possible (faster than one-by-one)
   - Use metadata filtering to reduce search space
   - Monitor query latency
   - Consider caching frequently accessed results

5. **Metadata Filtering**:
   ```python
   results = store.search(
       query="features",
       filter_metadata={"source": {"$eq": "product_docs.pdf"}}
   )
   ```
   
   Useful for:
   - Document-specific searches
   - Date range filtering
   - Category filtering
   - Multi-tenancy

6. **Similarity Scores**:
   - ChromaDB returns distances (lower = more similar)
   - Convert to similarity: 1 - distance
   - Typical threshold: 0.7+ for relevant results
   - Adjust based on your use case

7. **Error Handling**:
   - Handle OpenAI API errors (rate limits, invalid keys)
   - Handle ChromaDB errors (disk space, permissions)
   - Implement retries with exponential backoff
   - Log all errors for debugging

Real-World Tips:
- Start with ChromaDB for simplicity
- Monitor embedding costs (can add up quickly)
- Use metadata filtering to reduce API calls
- Consider caching embeddings for repeated queries
- Test different embedding models to find best balance of cost/quality
- Set up monitoring for vector store health
"""
