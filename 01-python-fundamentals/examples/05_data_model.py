"""
Data Model & Magic Methods for Senior Engineers

Python's data model (__dunder__ methods) allows you to create
objects that behave like built-in types. Essential for creating
intuitive APIs in ML/AI libraries.

Run: python 05_data_model.py
"""

from typing import Optional, Any
from dataclasses import dataclass
import math


# ============================================================================
# 1. Basic Magic Methods (__init__, __str__, __repr__)
# ============================================================================

class Embedding:
    """
    Vector embedding with magic methods.
    Makes embeddings behave like first-class objects.
    """
    
    def __init__(self, vector: list[float], metadata: Optional[dict] = None):
        """Constructor - called when creating instance."""
        self.vector = vector
        self.metadata = metadata or {}
        self.dimension = len(vector)
    
    def __str__(self) -> str:
        """String representation for users (print, str())."""
        preview = self.vector[:3]
        return f"Embedding({preview}..., dim={self.dimension})"
    
    def __repr__(self) -> str:
        """String representation for developers (debugging)."""
        return f"Embedding(vector={self.vector}, metadata={self.metadata})"
    
    def __len__(self) -> int:
        """Make embedding support len()."""
        return self.dimension
    
    def __getitem__(self, index: int) -> float:
        """Make embedding support indexing: emb[0]."""
        return self.vector[index]
    
    def __setitem__(self, index: int, value: float) -> None:
        """Make embedding support assignment: emb[0] = 0.5."""
        self.vector[index] = value
    
    def __iter__(self):
        """Make embedding iterable: for val in emb."""
        return iter(self.vector)
    
    def __eq__(self, other: "Embedding") -> bool:
        """Equality comparison: emb1 == emb2."""
        if not isinstance(other, Embedding):
            return False
        return self.vector == other.vector
    
    def __add__(self, other: "Embedding") -> "Embedding":
        """Add embeddings: emb1 + emb2."""
        if len(self) != len(other):
            raise ValueError("Embeddings must have same dimension")
        
        new_vector = [a + b for a, b in zip(self.vector, other.vector)]
        return Embedding(new_vector)
    
    def __mul__(self, scalar: float) -> "Embedding":
        """Multiply by scalar: emb * 2.0."""
        new_vector = [val * scalar for val in self.vector]
        return Embedding(new_vector)
    
    def __abs__(self) -> float:
        """Compute magnitude: abs(emb)."""
        return math.sqrt(sum(val * val for val in self.vector))
    
    def __contains__(self, value: float) -> bool:
        """Check membership: 0.5 in emb."""
        return value in self.vector


def demo_basic_magic_methods():
    """Demonstrate basic magic methods."""
    
    # Create embeddings
    emb1 = Embedding([0.1, 0.2, 0.3])
    emb2 = Embedding([0.4, 0.5, 0.6])
    
    # __str__ and __repr__
    print(f"str: {str(emb1)}")
    print(f"repr: {repr(emb1)}")
    
    # __len__
    print(f"Length: {len(emb1)}")
    
    # __getitem__ (indexing)
    print(f"First value: {emb1[0]}")
    
    # __setitem__ (assignment)
    emb1[0] = 0.9
    print(f"Modified: {emb1[0]}")
    
    # __iter__ (iteration)
    print(f"Values: {[val for val in emb1]}")
    
    # __eq__ (equality)
    print(f"Equal? {emb1 == emb2}")
    
    # __add__ (addition)
    emb3 = emb1 + emb2
    print(f"Sum: {emb3}")
    
    # __mul__ (multiplication)
    emb4 = emb1 * 2
    print(f"Doubled: {emb4}")
    
    # __abs__ (magnitude)
    print(f"Magnitude: {abs(emb1):.4f}")
    
    # __contains__ (membership)
    print(f"Contains 0.2? {0.2 in emb1}")


# ============================================================================
# 2. Context Manager Protocol (__enter__, __exit__)
# ============================================================================

class VectorDatabase:
    """Vector database with context manager protocol."""
    
    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self.documents = []
    
    def __enter__(self):
        """Setup - called when entering 'with' block."""
        print(f"🔌 Connecting to {self.url}")
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown - called when exiting 'with' block."""
        print(f"🔌 Disconnecting from {self.url}")
        self.connected = False
        return False
    
    def add(self, doc_id: str, embedding: Embedding) -> None:
        """Add document."""
        if not self.connected:
            raise RuntimeError("Not connected!")
        self.documents.append((doc_id, embedding))


def demo_context_manager_protocol():
    """Demonstrate context manager protocol."""
    
    with VectorDatabase("http://localhost:6333") as db:
        db.add("doc1", Embedding([0.1, 0.2, 0.3]))
        print(f"Added {len(db.documents)} documents")


# ============================================================================
# 3. Callable Objects (__call__)
# ============================================================================

class EmbeddingFunction:
    """
    Callable class - behaves like a function.
    Useful for stateful transformations.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0
    
    def __call__(self, text: str) -> Embedding:
        """Make instance callable: embed_fn(text)."""
        self.call_count += 1
        # Simulate embedding generation
        embedding = [hash(text + str(i)) % 100 / 100.0 for i in range(384)]
        return Embedding(embedding, {"model": self.model_name})
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "calls": self.call_count
        }


def demo_callable():
    """Demonstrate callable objects."""
    
    # Create callable embedding function
    embed = EmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")
    
    # Call like a function
    emb1 = embed("Hello world")
    emb2 = embed("Python is great")
    
    print(f"Generated embeddings: {len(emb1)}-dim")
    print(f"Stats: {embed.get_stats()}")


# ============================================================================
# 4. Container Protocol (__len__, __getitem__, __contains__)
# ============================================================================

class DocumentStore:
    """
    Document store that behaves like a collection.
    Supports len(), indexing, iteration, membership testing.
    """
    
    def __init__(self):
        self._documents: dict[str, dict] = {}
    
    def add(self, doc_id: str, content: str, embedding: Embedding) -> None:
        """Add document to store."""
        self._documents[doc_id] = {
            "content": content,
            "embedding": embedding
        }
    
    def __len__(self) -> int:
        """Number of documents."""
        return len(self._documents)
    
    def __getitem__(self, doc_id: str) -> dict:
        """Get document by ID: store[doc_id]."""
        if doc_id not in self._documents:
            raise KeyError(f"Document '{doc_id}' not found")
        return self._documents[doc_id]
    
    def __setitem__(self, doc_id: str, data: dict) -> None:
        """Set document: store[doc_id] = data."""
        self._documents[doc_id] = data
    
    def __delitem__(self, doc_id: str) -> None:
        """Delete document: del store[doc_id]."""
        del self._documents[doc_id]
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if document exists: doc_id in store."""
        return doc_id in self._documents
    
    def __iter__(self):
        """Iterate over document IDs."""
        return iter(self._documents.keys())
    
    def items(self):
        """Iterate over (doc_id, data) pairs."""
        return self._documents.items()


def demo_container_protocol():
    """Demonstrate container protocol."""
    
    store = DocumentStore()
    
    # Add documents
    store.add("doc1", "Python is great", Embedding([0.1, 0.2]))
    store.add("doc2", "AI is fascinating", Embedding([0.3, 0.4]))
    
    # __len__
    print(f"Total documents: {len(store)}")
    
    # __getitem__ (indexing)
    doc = store["doc1"]
    print(f"Retrieved: {doc['content']}")
    
    # __contains__ (membership)
    print(f"Has doc1? {'doc1' in store}")
    
    # __iter__ (iteration)
    print(f"Document IDs: {[doc_id for doc_id in store]}")
    
    # __delitem__ (deletion)
    del store["doc1"]
    print(f"After deletion: {len(store)} documents")


# ============================================================================
# 5. Comparison Protocol (__lt__, __le__, __gt__, __ge__)
# ============================================================================

@dataclass
class SearchResult:
    """Search result with comparison support."""
    doc_id: str
    score: float
    content: str
    
    def __lt__(self, other: "SearchResult") -> bool:
        """Less than comparison for sorting."""
        return self.score < other.score
    
    def __le__(self, other: "SearchResult") -> bool:
        """Less than or equal."""
        return self.score <= other.score
    
    def __gt__(self, other: "SearchResult") -> bool:
        """Greater than."""
        return self.score > other.score
    
    def __ge__(self, other: "SearchResult") -> bool:
        """Greater than or equal."""
        return self.score >= other.score


def demo_comparison_protocol():
    """Demonstrate comparison protocol."""
    
    results = [
        SearchResult("doc1", 0.5, "Content 1"),
        SearchResult("doc2", 0.9, "Content 2"),
        SearchResult("doc3", 0.7, "Content 3"),
    ]
    
    # Sort by score (uses __lt__)
    sorted_results = sorted(results, reverse=True)
    
    print("Sorted results:")
    for result in sorted_results:
        print(f"  {result.doc_id}: {result.score}")


# ============================================================================
# 6. Real-World Example: Complete Vector Store
# ============================================================================

class VectorStore:
    """
    Production-ready vector store using magic methods.
    Intuitive API through Python's data model.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._data: dict[str, tuple[Embedding, dict]] = {}
    
    def __len__(self) -> int:
        """Number of vectors in store."""
        return len(self._data)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self._data
    
    def __getitem__(self, doc_id: str) -> tuple[Embedding, dict]:
        """Get embedding and metadata."""
        return self._data[doc_id]
    
    def __setitem__(self, doc_id: str, value: tuple[Embedding, dict]) -> None:
        """Store embedding and metadata."""
        embedding, metadata = value
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected {self.dimension}-dim embedding")
        self._data[doc_id] = (embedding, metadata)
    
    def __delitem__(self, doc_id: str) -> None:
        """Delete document."""
        del self._data[doc_id]
    
    def __iter__(self):
        """Iterate over document IDs."""
        return iter(self._data.keys())
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"VectorStore(dimension={self.dimension}, size={len(self)})"
    
    def __enter__(self):
        """Context manager support."""
        print(f"📂 Opening vector store")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        print(f"📂 Closing vector store (saved {len(self)} vectors)")
        return False
    
    def search(self, query_embedding: Embedding, top_k: int = 5) -> list[SearchResult]:
        """Search for similar vectors."""
        results = []
        
        for doc_id, (embedding, metadata) in self._data.items():
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
            magnitude = abs(query_embedding) * abs(embedding)
            similarity = dot_product / magnitude if magnitude > 0 else 0.0
            
            results.append(SearchResult(
                doc_id=doc_id,
                score=similarity,
                content=metadata.get("content", "")
            ))
        
        # Sort by score (uses __lt__)
        results.sort(reverse=True)
        return results[:top_k]


def demo_complete_vector_store():
    """Demonstrate complete vector store."""
    
    # Use as context manager
    with VectorStore(dimension=3) as store:
        # Add documents (uses __setitem__)
        store["doc1"] = (
            Embedding([0.1, 0.2, 0.3]),
            {"content": "Python tutorial"}
        )
        store["doc2"] = (
            Embedding([0.4, 0.5, 0.6]),
            {"content": "AI guide"}
        )
        store["doc3"] = (
            Embedding([0.2, 0.3, 0.4]),
            {"content": "ML basics"}
        )
        
        # Check size (uses __len__)
        print(f"Store size: {len(store)}")
        
        # Check membership (uses __contains__)
        print(f"Has doc1? {'doc1' in store}")
        
        # Search (uses comparison protocol in SearchResult)
        query = Embedding([0.15, 0.25, 0.35])
        results = store.search(query, top_k=2)
        
        print("\nSearch results:")
        for result in results:
            print(f"  {result.doc_id}: {result.score:.3f} - {result.content}")
        
        # Iterate (uses __iter__)
        print(f"\nAll IDs: {list(store)}")


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Data Model & Magic Methods Examples ===\n")
    
    print("1. Basic Magic Methods:")
    demo_basic_magic_methods()
    print()
    
    print("2. Context Manager Protocol:")
    demo_context_manager_protocol()
    print()
    
    print("3. Callable Objects:")
    demo_callable()
    print()
    
    print("4. Container Protocol:")
    demo_container_protocol()
    print()
    
    print("5. Comparison Protocol:")
    demo_comparison_protocol()
    print()
    
    print("6. Complete Vector Store:")
    demo_complete_vector_store()
    
    print("\n✅ Magic methods make objects behave like built-in types!")
    print("💡 Use for: custom containers, embeddings, vector stores")


if __name__ == "__main__":
    main()
