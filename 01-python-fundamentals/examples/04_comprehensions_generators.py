"""
Comprehensions & Generators for Senior Engineers

More powerful and memory-efficient than JavaScript's map/filter/reduce.
Essential for processing large datasets in ML/AI without loading everything into memory.

Run: python 04_comprehensions_generators.py
"""

import time
from typing import Generator, Iterator
import sys


# ============================================================================
# 1. List Comprehensions (vs JavaScript map/filter)
# ============================================================================

# JavaScript equivalent:
# const squares = numbers.filter(x => x > 0).map(x => x * x);

def demo_list_comprehensions():
    """List comprehensions - create lists efficiently."""
    
    # Basic comprehension
    numbers = [1, 2, 3, 4, 5]
    squares = [x * x for x in numbers]
    print(f"Squares: {squares}")
    
    # With condition (filter + map in one)
    even_squares = [x * x for x in numbers if x % 2 == 0]
    print(f"Even squares: {even_squares}")
    
    # ML/AI example: Normalize embeddings
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    normalized = [
        [val / sum(emb) for val in emb]  # Nested comprehension
        for emb in embeddings
    ]
    print(f"Normalized: {normalized[0]}")
    
    # Flatten list of lists (common in document processing)
    documents = [
        ["word1", "word2"],
        ["word3", "word4"],
        ["word5"]
    ]
    all_words = [word for doc in documents for word in doc]
    print(f"Flattened: {all_words}")


# ============================================================================
# 2. Dictionary Comprehensions
# ============================================================================

def demo_dict_comprehensions():
    """Dictionary comprehensions - create dicts efficiently."""
    
    # Basic dict comprehension
    words = ["hello", "world", "python"]
    word_lengths = {word: len(word) for word in words}
    print(f"Word lengths: {word_lengths}")
    
    # ML/AI example: Create embedding cache
    documents = ["doc1", "doc2", "doc3"]
    embedding_cache = {
        doc: [0.1 * i, 0.2 * i, 0.3 * i]
        for i, doc in enumerate(documents, 1)
    }
    print(f"Cache: {embedding_cache['doc1']}")
    
    # Invert dictionary (swap keys and values)
    original = {"a": 1, "b": 2, "c": 3}
    inverted = {v: k for k, v in original.items()}
    print(f"Inverted: {inverted}")
    
    # Filter dictionary
    scores = {"doc1": 0.9, "doc2": 0.5, "doc3": 0.8}
    high_scores = {doc: score for doc, score in scores.items() if score > 0.7}
    print(f"High scores: {high_scores}")


# ============================================================================
# 3. Set Comprehensions
# ============================================================================

def demo_set_comprehensions():
    """Set comprehensions - unique values."""
    
    # Basic set comprehension
    numbers = [1, 2, 2, 3, 3, 4]
    unique_squares = {x * x for x in numbers}
    print(f"Unique squares: {unique_squares}")
    
    # ML/AI example: Unique tokens across documents
    documents = [
        "hello world",
        "hello python",
        "world of python"
    ]
    unique_tokens = {
        token
        for doc in documents
        for token in doc.split()
    }
    print(f"Unique tokens: {unique_tokens}")


# ============================================================================
# 4. Generator Expressions (Lazy Evaluation)
# ============================================================================

def demo_generator_expressions():
    """
    Generator expressions - like list comprehensions but lazy.
    Don't create list in memory, generate values on-demand.
    """
    
    # List comprehension - creates entire list immediately
    squares_list = [x * x for x in range(1000000)]
    print(f"List size: {sys.getsizeof(squares_list)} bytes")
    
    # Generator expression - creates generator object (small)
    squares_gen = (x * x for x in range(1000000))
    print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")
    
    # Generator consumed on-demand
    first_five = [next(squares_gen) for _ in range(5)]
    print(f"First five: {first_five}")
    
    # ML/AI example: Process documents without loading all into memory
    def simulate_large_dataset():
        """Simulate large dataset from database/API."""
        for i in range(5):
            yield f"Document {i}: Some content here..."
    
    # Process one at a time
    processed = (
        doc.upper()  # Some processing
        for doc in simulate_large_dataset()
    )
    
    for doc in processed:
        print(doc[:30])


# ============================================================================
# 5. Generator Functions (yield keyword)
# ============================================================================

def chunk_text(text: str, chunk_size: int) -> Generator[str, None, None]:
    """
    Generator function - yields values one at a time.
    Essential for processing large texts in ML/AI.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        yield chunk


def batch_generator(
    items: list, 
    batch_size: int
) -> Generator[list, None, None]:
    """
    Batch generator - critical for ML training/inference.
    Process data in batches without loading everything.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def demo_generator_functions():
    """Generator functions with yield."""
    
    # Text chunking for embeddings
    large_text = "This is a large document with many words " * 10
    
    print("Text chunks:")
    for i, chunk in enumerate(chunk_text(large_text, chunk_size=10), 1):
        print(f"Chunk {i}: {chunk[:40]}...")
        if i >= 3:
            break
    
    # Batch processing
    documents = [f"doc_{i}" for i in range(100)]
    
    print("\nBatch processing:")
    for batch_num, batch in enumerate(batch_generator(documents, batch_size=32), 1):
        print(f"Processing batch {batch_num}: {len(batch)} documents")
        # Process batch (e.g., generate embeddings)
        time.sleep(0.1)
        if batch_num >= 3:
            break


# ============================================================================
# 6. Advanced: Generator Pipeline
# ============================================================================

def read_documents() -> Generator[str, None, None]:
    """Simulate reading documents from source."""
    for i in range(10):
        yield f"Document {i}: " + "word " * 50


def preprocess(docs: Iterator[str]) -> Generator[str, None, None]:
    """Preprocess documents."""
    for doc in docs:
        yield doc.lower().strip()


def tokenize(docs: Iterator[str]) -> Generator[list[str], None, None]:
    """Tokenize documents."""
    for doc in docs:
        yield doc.split()


def embed(tokens: Iterator[list[str]]) -> Generator[list[float], None, None]:
    """Generate embeddings (simulated)."""
    for token_list in tokens:
        # Simulate embedding generation
        yield [len(token_list) * 0.1, len(token_list) * 0.2]


def demo_generator_pipeline():
    """
    Generator pipeline - chain generators for memory-efficient processing.
    Each document flows through pipeline without storing all in memory.
    """
    print("Generator Pipeline:")
    
    # Chain generators - only one document in memory at a time!
    pipeline = embed(tokenize(preprocess(read_documents())))
    
    # Process results
    embeddings = []
    for i, embedding in enumerate(pipeline, 1):
        embeddings.append(embedding)
        print(f"Processed document {i}: {embedding}")
        if i >= 5:
            break
    
    print(f"Total embeddings: {len(embeddings)}")


# ============================================================================
# 7. Real-World Example: Streaming Document Embeddings
# ============================================================================

class DocumentEmbeddingPipeline:
    """
    Production-ready document embedding pipeline using generators.
    Memory-efficient for processing millions of documents.
    """
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def load_documents(self, source: str) -> Generator[dict, None, None]:
        """Simulate loading documents from database/file."""
        for i in range(100):
            yield {
                "id": f"doc_{i}",
                "content": f"Document {i} content with various words " * 10,
                "metadata": {"source": source}
            }
    
    def preprocess_documents(
        self, 
        docs: Iterator[dict]
    ) -> Generator[dict, None, None]:
        """Preprocess documents (clean, normalize)."""
        for doc in docs:
            doc["processed_content"] = doc["content"].lower().strip()
            yield doc
    
    def chunk_documents(
        self, 
        docs: Iterator[dict], 
        chunk_size: int = 100
    ) -> Generator[dict, None, None]:
        """Split documents into chunks for embedding."""
        for doc in docs:
            words = doc["processed_content"].split()
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
            
            for i, chunk in enumerate(chunks):
                yield {
                    "doc_id": doc["id"],
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "content": chunk,
                    "metadata": doc["metadata"]
                }
    
    def batch_documents(
        self, 
        docs: Iterator[dict]
    ) -> Generator[list[dict], None, None]:
        """Batch documents for efficient embedding generation."""
        batch = []
        for doc in docs:
            batch.append(doc)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining
            yield batch
    
    def generate_embeddings(
        self, 
        batches: Iterator[list[dict]]
    ) -> Generator[dict, None, None]:
        """Generate embeddings for batches."""
        for batch in batches:
            # Simulate batch embedding generation
            time.sleep(0.1)  # Simulate API call
            
            for doc in batch:
                doc["embedding"] = [0.1, 0.2, 0.3]  # Simulated
                yield doc
    
    def process(self, source: str) -> Generator[dict, None, None]:
        """
        Complete pipeline - yields processed documents with embeddings.
        Memory-efficient: only one batch in memory at a time.
        """
        pipeline = self.generate_embeddings(
            self.batch_documents(
                self.chunk_documents(
                    self.preprocess_documents(
                        self.load_documents(source)
                    )
                )
            )
        )
        
        yield from pipeline  # Yield all items from pipeline


def demo_real_world_pipeline():
    """Demonstrate production-ready streaming pipeline."""
    print("Real-World Document Embedding Pipeline:")
    
    pipeline = DocumentEmbeddingPipeline(batch_size=10)
    
    # Process documents - memory efficient!
    processed_count = 0
    for doc in pipeline.process("s3://my-bucket/documents"):
        processed_count += 1
        if processed_count <= 5:
            print(f"✅ Processed: {doc['chunk_id']}")
        
        # In production: store to vector database here
        # vector_store.upsert(doc['chunk_id'], doc['embedding'], doc['metadata'])
        
        if processed_count >= 50:
            break
    
    print(f"\n📊 Total processed: {processed_count} document chunks")
    print("💡 Memory usage stays constant regardless of dataset size!")


# ============================================================================
# 8. Performance Comparison
# ============================================================================

def demo_performance():
    """Compare list vs generator performance."""
    import tracemalloc
    
    n = 1_000_000
    
    # List comprehension - loads everything into memory
    print("List comprehension:")
    tracemalloc.start()
    start = time.time()
    
    squares_list = [x * x for x in range(n)]
    total = sum(squares_list)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  Time: {time.time() - start:.4f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    # Generator expression - minimal memory
    print("\nGenerator expression:")
    tracemalloc.start()
    start = time.time()
    
    squares_gen = (x * x for x in range(n))
    total = sum(squares_gen)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  Time: {time.time() - start:.4f}s")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Comprehensions & Generators Examples ===\n")
    
    print("1. List Comprehensions:")
    demo_list_comprehensions()
    print()
    
    print("2. Dictionary Comprehensions:")
    demo_dict_comprehensions()
    print()
    
    print("3. Set Comprehensions:")
    demo_set_comprehensions()
    print()
    
    print("4. Generator Expressions:")
    demo_generator_expressions()
    print()
    
    print("5. Generator Functions:")
    demo_generator_functions()
    print()
    
    print("6. Generator Pipeline:")
    demo_generator_pipeline()
    print()
    
    print("7. Real-World Pipeline:")
    demo_real_world_pipeline()
    print()
    
    print("8. Performance Comparison:")
    demo_performance()
    
    print("\n✅ Generators are essential for memory-efficient ML/AI!")
    print("💡 Use for: large datasets, streaming data, batch processing")


if __name__ == "__main__":
    main()
