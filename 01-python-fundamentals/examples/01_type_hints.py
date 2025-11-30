"""
Type Hints for Senior Engineers

Coming from Go's static typing, Python's type hints provide similar benefits
for tooling and documentation, but without runtime enforcement.

Run: python 01_type_hints.py
Type check: mypy 01_type_hints.py
"""

from typing import (
    Any, Callable, Dict, List, Optional, 
    Union, Tuple, TypeVar, Generic, Protocol
)
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# 1. Basic Type Hints (Similar to Go)
# ============================================================================

def calculate_similarity(text1: str, text2: str) -> float:
    """Basic type hints - straightforward like Go."""
    # Simplified similarity (real ML would use embeddings)
    common = set(text1.split()) & set(text2.split())
    total = set(text1.split()) | set(text2.split())
    return len(common) / len(total) if total else 0.0


def fetch_embeddings(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Python 3.10+ syntax: list[str] instead of List[str]"""
    # Simulate embedding generation
    return [[0.1, 0.2, 0.3] for _ in texts]


# ============================================================================
# 2. Optional Types (vs Go's nil)
# ============================================================================

def get_cached_embedding(text: str) -> Optional[list[float]]:
    """
    Optional[T] is like *T in Go - can be None or the type.
    Python 3.10+: Can also use `list[float] | None`
    """
    cache: dict[str, list[float]] = {}
    return cache.get(text)  # Returns None if not found


def process_with_default(
    text: str, 
    embedding: Optional[list[float]] = None
) -> list[float]:
    """Optional parameter with default None."""
    if embedding is None:
        return fetch_embeddings([text])[0]
    return embedding


# ============================================================================
# 3. Union Types (vs Go's interface{})
# ============================================================================

# Old style
def old_process(value: Union[str, int, float]) -> str:
    """Union for multiple possible types."""
    return str(value)


# Python 3.10+ style (cleaner)
def new_process(value: str | int | float) -> str:
    """Pipe operator for union types (like TypeScript)."""
    return str(value)


# ============================================================================
# 4. Type Aliases (Like Go's type declarations)
# ============================================================================

# Type aliases for clarity
Embedding = list[float]
EmbeddingBatch = list[Embedding]
DocumentID = str
EmbeddingCache = dict[DocumentID, Embedding]

def store_embeddings(
    cache: EmbeddingCache, 
    doc_id: DocumentID, 
    embedding: Embedding
) -> None:
    """Type aliases make code self-documenting."""
    cache[doc_id] = embedding


# ============================================================================
# 5. Generics (Like Go generics)
# ============================================================================

T = TypeVar('T')

def first_element(items: list[T]) -> Optional[T]:
    """Generic function - works with any type."""
    return items[0] if items else None


class VectorStore(Generic[T]):
    """Generic class - like Go's generic structs."""
    
    def __init__(self):
        self.items: list[T] = []
    
    def add(self, item: T) -> None:
        self.items.append(item)
    
    def get(self, index: int) -> Optional[T]:
        return self.items[index] if 0 <= index < len(self.items) else None


# ============================================================================
# 6. Protocols (Like Go interfaces)
# ============================================================================

class Embeddable(Protocol):
    """
    Protocol defines an interface - any class with these methods satisfies it.
    Like Go's implicit interface implementation (duck typing with types).
    """
    def get_text(self) -> str: ...
    def get_embedding(self) -> Embedding: ...


@dataclass
class Document:
    """Document that satisfies Embeddable protocol."""
    content: str
    embedding: Embedding
    
    def get_text(self) -> str:
        return self.content
    
    def get_embedding(self) -> Embedding:
        return self.embedding


def process_embeddable(item: Embeddable) -> Embedding:
    """Works with anything that implements the protocol."""
    text = item.get_text()
    return item.get_embedding()


# ============================================================================
# 7. Callable Types (Function types)
# ============================================================================

# Function that takes a string and returns embedding
EmbeddingFunction = Callable[[str], Embedding]

def apply_embedding_function(
    texts: list[str], 
    embed_fn: EmbeddingFunction
) -> EmbeddingBatch:
    """Higher-order function with typed callback."""
    return [embed_fn(text) for text in texts]


# ============================================================================
# 8. Dataclasses (Like Go structs with tags)
# ============================================================================

@dataclass
class SearchResult:
    """
    Dataclass = auto-generated __init__, __repr__, __eq__
    Similar to Go struct with json tags
    """
    document_id: str
    score: float
    content: str
    metadata: dict[str, Any] = None  # Default value
    
    def __post_init__(self):
        """Called after __init__ - for validation."""
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# 9. Enums (Like Go const iota)
# ============================================================================

class SearchType(Enum):
    """Type-safe enums."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


def search(
    query: str, 
    search_type: SearchType = SearchType.SEMANTIC
) -> list[SearchResult]:
    """Enum ensures only valid search types."""
    print(f"Searching with {search_type.value}")
    return []


# ============================================================================
# 10. Real-World AI/ML Example: Type-Safe RAG Pipeline
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration with type hints."""
    model: str
    temperature: float
    max_tokens: int
    top_k: int = 5
    
    def validate(self) -> None:
        """Validation logic."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")


class RAGPipeline:
    """Type-hinted RAG pipeline."""
    
    def __init__(
        self, 
        config: RAGConfig,
        embedding_fn: EmbeddingFunction
    ):
        self.config = config
        self.embedding_fn = embedding_fn
        self.vector_store: VectorStore[Document] = VectorStore()
    
    def add_documents(self, documents: list[str]) -> None:
        """Add documents to vector store."""
        for doc_text in documents:
            embedding = self.embedding_fn(doc_text)
            doc = Document(content=doc_text, embedding=embedding)
            self.vector_store.add(doc)
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> list[SearchResult]:
        """Search with type-safe results."""
        k = top_k or self.config.top_k
        # Simplified search logic
        return []
    
    def generate_response(
        self, 
        query: str, 
        context: list[str]
    ) -> str:
        """Generate response from query and context."""
        # Would call LLM here
        return f"Response based on {len(context)} documents"


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Type Hints Examples ===\n")
    
    # Basic types
    sim = calculate_similarity("hello world", "hello python")
    print(f"Similarity: {sim:.2f}")
    
    # Optional types
    cached = get_cached_embedding("test")
    print(f"Cached embedding: {cached}")
    
    # Generics
    store: VectorStore[str] = VectorStore()
    store.add("document 1")
    print(f"First item: {store.get(0)}")
    
    # Protocols
    doc = Document(content="AI is fascinating", embedding=[0.1, 0.2, 0.3])
    embedding = process_embeddable(doc)
    print(f"Processed embedding: {embedding}")
    
    # Enums
    results = search("AI tutorials", SearchType.SEMANTIC)
    print(f"Search results: {len(results)}")
    
    # Real-world example
    config = RAGConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )
    
    def dummy_embed(text: str) -> Embedding:
        return [0.1] * 384  # Simulated 384-dim embedding
    
    pipeline = RAGPipeline(config, dummy_embed)
    pipeline.add_documents([
        "Python is great for AI",
        "Machine learning requires data",
        "LangChain simplifies LLM apps"
    ])
    
    print("\n✅ Type hints provide IDE autocomplete and error checking!")
    print("💡 Run 'mypy 01_type_hints.py' to check types statically")


if __name__ == "__main__":
    main()
