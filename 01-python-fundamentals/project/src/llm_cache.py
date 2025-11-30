"""
LLM Response Cache - Production-Ready Implementation

A decorator-based caching system for LLM API calls that demonstrates:
- Decorators with parameters
- Context managers for resource management
- Type hints for type safety
- File I/O for persistence
- Statistics tracking

Usage:
    from llm_cache import llm_cache, CacheManager
    
    @llm_cache(ttl=3600)
    def call_llm(prompt: str) -> str:
        return expensive_api_call(prompt)
"""

import functools
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Any, Optional, TypeVar, ParamSpec, Generator
from dataclasses import dataclass, asdict
from datetime import datetime


# Type variables for generic function signatures
P = ParamSpec('P')  # Parameters
T = TypeVar('T')    # Return type


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    ttl: Optional[int]
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        age = time.time() - self.created_at
        return age > self.ttl
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)


class Cache:
    """
    In-memory cache with disk persistence.
    Demonstrates: magic methods, type hints, context managers.
    """
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self._entries: dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0
        }
    
    def __len__(self) -> int:
        """Number of entries in cache."""
        return len(self._entries)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._entries:
            return False
        entry = self._entries[key]
        if entry.is_expired():
            del self._entries[key]
            return False
        return True
    
    def __getitem__(self, key: str) -> Any:
        """Get value from cache."""
        if key not in self:
            raise KeyError(f"Key '{key}' not in cache")
        
        entry = self._entries[key]
        entry.hit_count += 1
        self._stats["hits"] += 1
        return entry.value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value in cache (simple interface)."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Cache(entries={len(self)}, hits={self._stats['hits']}, misses={self._stats['misses']})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        try:
            return self[key]
        except KeyError:
            self._stats["misses"] += 1
            return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with optional TTL."""
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
        self._entries[key] = entry
        self._stats["writes"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests * 100 
            if total_requests > 0 
            else 0
        )
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "writes": self._stats["writes"],
            "entries": len(self),
            "hit_rate": hit_rate
        }
    
    def save(self) -> None:
        """Save cache to disk."""
        if not self.cache_file:
            return
        
        # Ensure directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = {
            key: entry.to_dict()
            for key, entry in self._entries.items()
            if not entry.is_expired()
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Cache saved: {len(data)} entries to {self.cache_file}")
    
    def load(self) -> None:
        """Load cache from disk."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        
        # Convert from JSON format
        self._entries = {
            key: CacheEntry.from_dict(entry_data)
            for key, entry_data in data.items()
        }
        
        # Remove expired entries
        expired = [
            key for key, entry in self._entries.items()
            if entry.is_expired()
        ]
        for key in expired:
            del self._entries[key]
        
        print(f"📂 Cache loaded: {len(self)} entries from {self.cache_file}")


def make_cache_key(*args, **kwargs) -> str:
    """
    Create cache key from function arguments.
    Uses hash for consistent key generation.
    """
    # Create string representation of arguments
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    
    # Hash for consistent key (handles long arguments)
    return hashlib.md5(key_str.encode()).hexdigest()


def llm_cache(
    ttl: Optional[int] = 3600,
    cache_instance: Optional[Cache] = None
):
    """
    Decorator for caching LLM responses.
    
    Args:
        ttl: Time-to-live in seconds (None = never expire)
        cache_instance: Cache instance to use (creates new if None)
    
    Example:
        @llm_cache(ttl=3600)
        def call_llm(prompt: str) -> str:
            return expensive_api_call(prompt)
    """
    # Use provided cache or create new one
    cache = cache_instance or Cache()
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Create cache key from arguments
            cache_key = f"{func.__name__}:{make_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                print(f"⚡ Cache hit for {func.__name__}")
                return cached_value
            
            # Cache miss - call function
            print(f"💾 Cache miss for {func.__name__} - executing")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        # Attach cache to function for external access
        wrapper.cache = cache  # type: ignore
        
        return wrapper
    
    return decorator


class CacheManager:
    """
    Context manager for cache lifecycle.
    Demonstrates: __enter__, __exit__, resource management.
    
    Example:
        with CacheManager("./cache") as cache:
            @cache.cached(ttl=1800)
            def my_function(x):
                return expensive_operation(x)
    """
    
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "llm_cache.json"
        self.cache = Cache(cache_file=self.cache_file)
    
    def __enter__(self) -> "CacheManager":
        """Load cache when entering context."""
        print(f"📂 Opening cache: {self.cache_dir}")
        self.cache.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Save cache when exiting context."""
        if exc_type is not None:
            print(f"⚠️  Exception occurred: {exc_val}")
        
        self.cache.save()
        print(f"📂 Cache context closed")
        return False  # Don't suppress exceptions
    
    def cached(
        self, 
        ttl: Optional[int] = 3600
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Create decorator using this cache instance."""
        return llm_cache(ttl=ttl, cache_instance=self.cache)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.stats()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# ============================================================================
# Demo Functions
# ============================================================================

def simulate_llm_call(prompt: str, delay: float = 1.5) -> str:
    """Simulate expensive LLM API call."""
    time.sleep(delay)
    return f"Response to: {prompt[:50]}..."


def demo_basic_caching():
    """Demonstrate basic decorator usage."""
    print("=" * 60)
    print("Demo 1: Basic Caching")
    print("=" * 60)
    
    @llm_cache(ttl=300)
    def call_llm(prompt: str) -> str:
        return simulate_llm_call(prompt)
    
    # First call - cache miss
    print("\n1️⃣ First call (cache miss):")
    start = time.time()
    result1 = call_llm("What is Python?")
    time1 = time.time() - start
    print(f"⏱️  Time: {time1:.4f}s")
    print(f"📝 Result: {result1}")
    
    # Second call - cache hit
    print("\n2️⃣ Second call (cache hit):")
    start = time.time()
    result2 = call_llm("What is Python?")
    time2 = time.time() - start
    print(f"⏱️  Time: {time2:.4f}s")
    print(f"📝 Result: {result2}")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n🚀 Speedup: {speedup:.0f}x faster!")
    
    # Show stats
    print(f"\n📊 Stats: {call_llm.cache.stats()}")


def demo_cache_manager():
    """Demonstrate context manager usage."""
    print("\n" + "=" * 60)
    print("Demo 2: Cache Manager (Context Manager)")
    print("=" * 60)
    
    cache_dir = Path("./temp_cache")
    
    # First session
    print("\n1️⃣ First session:")
    with CacheManager(cache_dir) as manager:
        @manager.cached(ttl=600)
        def process_data(data: str) -> str:
            return f"Processed: {simulate_llm_call(data, delay=0.5)}"
        
        result1 = process_data("test data")
        print(f"📝 Result: {result1}")
        print(f"📊 Stats: {manager.stats()}")
    
    # Second session - cache persisted
    print("\n2️⃣ Second session (cache loaded from disk):")
    with CacheManager(cache_dir) as manager:
        @manager.cached(ttl=600)
        def process_data(data: str) -> str:
            return f"Processed: {simulate_llm_call(data, delay=0.5)}"
        
        result2 = process_data("test data")  # Cache hit!
        print(f"📝 Result: {result2}")
        print(f"📊 Stats: {manager.stats()}")
    
    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"\n🗑️  Cleaned up: {cache_dir}")


def demo_multiple_functions():
    """Demonstrate caching multiple functions."""
    print("\n" + "=" * 60)
    print("Demo 3: Multiple Cached Functions")
    print("=" * 60)
    
    cache = Cache()
    
    @llm_cache(ttl=300, cache_instance=cache)
    def generate_summary(text: str) -> str:
        print(f"  🤖 Generating summary...")
        return f"Summary: {text[:20]}..."
    
    @llm_cache(ttl=300, cache_instance=cache)
    def generate_embedding(text: str) -> list[float]:
        print(f"  🔢 Generating embedding...")
        return [0.1, 0.2, 0.3]
    
    # Use both functions
    print("\n1️⃣ First calls:")
    summary = generate_summary("Long article about Python")
    embedding = generate_embedding("Python is great")
    
    print("\n2️⃣ Second calls (both cached):")
    summary2 = generate_summary("Long article about Python")
    embedding2 = generate_embedding("Python is great")
    
    print(f"\n📊 Shared cache stats: {cache.stats()}")


def main():
    """Run all demos."""
    print("\n🚀 LLM Response Cache Demo\n")
    
    demo_basic_caching()
    demo_cache_manager()
    demo_multiple_functions()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)
    print("\n💡 Key Takeaways:")
    print("  - Decorators simplify caching logic")
    print("  - Context managers ensure proper cleanup")
    print("  - Type hints improve code quality")
    print("  - Persistence makes cache survive restarts")
    print("  - Shared cache instances save memory")


if __name__ == "__main__":
    main()
