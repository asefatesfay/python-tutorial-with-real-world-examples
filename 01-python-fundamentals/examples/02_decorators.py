"""
Decorators for Senior Engineers

More powerful than JavaScript/TypeScript decorators. Essential for:
- Timing ML model inference
- Caching embeddings
- Retry logic for API calls
- Authentication and logging

Run: python 02_decorators.py
"""

import time
import functools
from typing import Callable, Any, TypeVar, ParamSpec
from datetime import datetime


# TypeVar for generic return types
P = ParamSpec('P')  # For preserving function signatures
T = TypeVar('T')


# ============================================================================
# 1. Basic Function Decorator
# ============================================================================

def timer(func: Callable[P, T]) -> Callable[P, T]:
    """
    Basic decorator - measures execution time.
    Essential for profiling ML inference.
    """
    @functools.wraps(func)  # Preserves original function metadata
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"⏱️  {func.__name__} took {duration:.4f}s")
        return result
    return wrapper


@timer
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Simulate embedding generation."""
    time.sleep(0.5)  # Simulate API call
    return [[0.1, 0.2, 0.3] for _ in texts]


# ============================================================================
# 2. Decorator with Arguments (Parameterized)
# ============================================================================

def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator factory - creates decorator with custom parameters.
    Essential for resilient API calls.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"⚠️  Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


@retry(max_attempts=3, delay=0.5)
def call_llm_api(prompt: str) -> str:
    """Simulate flaky API call."""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("API unavailable")
    return f"Response to: {prompt}"


# ============================================================================
# 3. Caching Decorator (Essential for ML/AI)
# ============================================================================

def simple_cache(func: Callable[P, T]) -> Callable[P, T]:
    """
    Memoization decorator - caches function results.
    Critical for caching embeddings during development.
    """
    cache: dict[str, T] = {}
    
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            print(f"💾 Cache miss - computing result")
            cache[key] = func(*args, **kwargs)
        else:
            print(f"⚡ Cache hit - returning cached result")
        
        return cache[key]
    return wrapper


@simple_cache
@timer
def expensive_embedding(text: str) -> list[float]:
    """Simulate expensive embedding computation."""
    time.sleep(1.0)
    return [hash(text) % 100 / 100.0 for _ in range(384)]


# ============================================================================
# 4. Class Decorators
# ============================================================================

def singleton(cls):
    """
    Class decorator - ensures only one instance exists.
    Useful for vector store clients, model loaders.
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


@singleton
class VectorStoreClient:
    """Only one instance will ever exist."""
    
    def __init__(self, url: str):
        print(f"🔌 Connecting to vector store at {url}")
        self.url = url
    
    def search(self, query: str) -> list[str]:
        return [f"Result for {query}"]


# ============================================================================
# 5. Method Decorators (@property, @staticmethod, @classmethod)
# ============================================================================

class MLModel:
    """Demonstrates built-in decorators."""
    
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._loaded = False
    
    @property
    def model_name(self) -> str:
        """Property - access like attribute, not method."""
        return self._model_name
    
    @property
    def is_loaded(self) -> bool:
        """Computed property."""
        return self._loaded
    
    @staticmethod
    def supported_models() -> list[str]:
        """Static method - doesn't need instance or class."""
        return ["gpt-4", "claude-3", "llama-2"]
    
    @classmethod
    def from_config(cls, config: dict) -> "MLModel":
        """Class method - alternative constructor."""
        return cls(config["model_name"])
    
    def load(self) -> None:
        """Regular instance method."""
        print(f"Loading {self._model_name}...")
        self._loaded = True


# ============================================================================
# 6. Decorator Stacking (Combining Multiple Decorators)
# ============================================================================

@timer
@retry(max_attempts=2, delay=0.3)
@simple_cache
def fetch_and_embed(text: str) -> list[float]:
    """
    Multiple decorators are applied bottom-to-top:
    1. simple_cache (innermost - caches results)
    2. retry (retries on failure)
    3. timer (outermost - times everything)
    """
    print(f"🌐 Fetching embedding for: {text[:30]}...")
    time.sleep(0.3)
    return [0.1, 0.2, 0.3]


# ============================================================================
# 7. Real-World Example: Rate Limiting Decorator
# ============================================================================

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls: list[float] = []
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Make the class instance callable as a decorator."""
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            # Check rate limit
            if len(self.calls) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.calls[0])
                print(f"⏸️  Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            self.calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper


# Rate limit: max 10 calls per minute
rate_limit = RateLimiter(calls_per_minute=10)

@rate_limit
def call_embedding_api(text: str) -> list[float]:
    """API call with rate limiting."""
    print(f"📡 Calling API for: {text[:30]}")
    return [0.1, 0.2, 0.3]


# ============================================================================
# 8. Context-Aware Decorator (for logging with metadata)
# ============================================================================

def log_with_context(context: str):
    """Decorator that logs with additional context."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            timestamp = datetime.now().isoformat()
            print(f"[{timestamp}] [{context}] Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                print(f"[{timestamp}] [{context}] ✅ Success")
                return result
            except Exception as e:
                print(f"[{timestamp}] [{context}] ❌ Error: {e}")
                raise
        return wrapper
    return decorator


@log_with_context("RAG Pipeline")
def retrieve_documents(query: str, top_k: int = 5) -> list[str]:
    """Retrieve relevant documents."""
    return [f"Doc {i}" for i in range(top_k)]


# ============================================================================
# 9. Real-World RAG Example: Complete Pipeline with Decorators
# ============================================================================

class RAGPipelineDecorated:
    """RAG pipeline using multiple decorators."""
    
    @timer
    @simple_cache
    def embed_query(self, query: str) -> list[float]:
        """Cache embeddings to avoid repeated API calls."""
        print(f"🔢 Embedding query: {query}")
        time.sleep(0.2)  # Simulate API call
        return [0.1] * 384
    
    @timer
    @retry(max_attempts=3, delay=0.5)
    def search_vector_store(
        self, 
        embedding: list[float], 
        top_k: int = 5
    ) -> list[str]:
        """Retry on vector store failures."""
        print(f"🔍 Searching for top {top_k} results")
        return [f"Document {i}" for i in range(top_k)]
    
    @timer
    @rate_limit
    @retry(max_attempts=2, delay=1.0)
    def generate_response(
        self, 
        query: str, 
        context: list[str]
    ) -> str:
        """Rate-limited, retryable LLM call."""
        print(f"🤖 Generating response with {len(context)} docs")
        time.sleep(0.3)
        return f"Response based on {len(context)} documents"
    
    def query(self, user_query: str) -> str:
        """Complete RAG pipeline."""
        # All methods benefit from decorators
        embedding = self.embed_query(user_query)
        docs = self.search_vector_store(embedding, top_k=3)
        response = self.generate_response(user_query, docs)
        return response


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("=== Decorator Examples ===\n")
    
    # 1. Timer
    print("1. Timer Decorator:")
    embeddings = generate_embeddings(["hello", "world"])
    print()
    
    # 2. Retry
    print("2. Retry Decorator:")
    try:
        response = call_llm_api("What is AI?")
        print(f"✅ Got response: {response}")
    except Exception as e:
        print(f"❌ All attempts failed: {e}")
    print()
    
    # 3. Cache
    print("3. Cache Decorator:")
    emb1 = expensive_embedding("Hello AI")
    emb2 = expensive_embedding("Hello AI")  # Cached
    print()
    
    # 4. Singleton
    print("4. Singleton Decorator:")
    client1 = VectorStoreClient("http://localhost:6333")
    client2 = VectorStoreClient("http://localhost:6333")  # Same instance
    print(f"Same instance? {client1 is client2}")
    print()
    
    # 5. Property decorators
    print("5. Property Decorators:")
    model = MLModel("gpt-4")
    print(f"Model name: {model.model_name}")  # Access like attribute
    print(f"Is loaded: {model.is_loaded}")
    print(f"Supported: {MLModel.supported_models()}")
    print()
    
    # 6. Stacked decorators
    print("6. Stacked Decorators:")
    result1 = fetch_and_embed("Python is awesome")
    result2 = fetch_and_embed("Python is awesome")  # Cached
    print()
    
    # 7. Context logging
    print("7. Context Logging:")
    docs = retrieve_documents("What is RAG?")
    print()
    
    # 8. Complete RAG example
    print("8. Complete RAG Pipeline:")
    pipeline = RAGPipelineDecorated()
    answer = pipeline.query("Explain decorators")
    print(f"📝 Final answer: {answer}")
    print()
    
    print("✅ Decorators are essential for production ML/AI code!")
    print("💡 Use for: timing, caching, retries, rate limiting, logging")


if __name__ == "__main__":
    main()
