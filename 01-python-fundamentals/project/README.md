# Module 1 Mini-Project: LLM Response Cache

Build a production-ready caching system for LLM API calls using decorators, context managers, and type hints.

## 🎯 Project Goals

Learn by building:
- Function decorators for caching
- Context managers for cache lifecycle
- Type hints for type safety
- Generator functions for streaming
- Magic methods for intuitive API

## 📋 Requirements

Build a caching system that:
1. Caches LLM responses to avoid duplicate API calls
2. Supports TTL (time-to-live) for cache entries
3. Works as a decorator
4. Provides cache statistics
5. Can persist cache to disk
6. Supports streaming responses

## 🏗️ Implementation

See the complete implementation in `src/llm_cache.py`

## 🧪 Usage Examples

```python
from llm_cache import llm_cache, CacheManager

# Example 1: Basic decorator usage
@llm_cache(ttl=3600)  # Cache for 1 hour
def call_openai(prompt: str, model: str = "gpt-4") -> str:
    """Call OpenAI API (expensive)."""
    # Make actual API call
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# First call - hits API
result1 = call_openai("What is Python?")

# Second call - cached (instant!)
result2 = call_openai("What is Python?")


# Example 2: Context manager for cache lifecycle
with CacheManager("./cache") as cache:
    # Cache automatically loaded
    
    @cache.cached(ttl=1800)
    def expensive_operation(data: str) -> str:
        return f"Processed: {data}"
    
    result = expensive_operation("test")
    print(cache.stats())  # View cache statistics

# Cache automatically saved on exit


# Example 3: Streaming responses
@llm_cache(ttl=7200)
def stream_openai(prompt: str):
    """Stream LLM responses."""
    for chunk in openai_stream(prompt):
        yield chunk

# Stream and cache
for chunk in stream_openai("Explain decorators"):
    print(chunk, end="")
```

## 🎓 What You'll Learn

### Decorators
- Creating decorators with parameters
- Preserving function metadata with `functools.wraps`
- Stacking multiple decorators
- Class-based decorators for state

### Context Managers
- Implementing `__enter__` and `__exit__`
- Using `@contextmanager` decorator
- Exception handling in context managers
- Resource cleanup patterns

### Type Hints
- Generic types with `TypeVar`
- `Callable` types for functions
- `Optional` and `Union` types
- Type aliases for clarity

### Generators
- Generator functions with `yield`
- Generator expressions
- Streaming data efficiently
- Memory-efficient iteration

### Magic Methods
- `__call__` for callable classes
- `__enter__` and `__exit__` for context managers
- `__repr__` for debugging
- Custom data structures

## 🏃 Running the Project

```bash
# Navigate to project directory
cd 01-python-fundamentals/project

# Run the implementation
python src/llm_cache.py

# Run tests
pytest tests/ -v

# Check types
mypy src/

# Run example
python example.py
```

## 🎯 Exercises

Extend the cache system:

1. **Add cache size limits** - Implement LRU eviction
2. **Add cache warming** - Pre-populate cache from common queries
3. **Add cache invalidation** - Clear specific entries or patterns
4. **Add Redis backend** - Use Redis instead of disk
5. **Add metrics** - Track hit rate, latency, cost savings

## 📊 Expected Output

```
=== LLM Response Cache Demo ===

1. Basic Caching:
💾 Cache miss - making API call
⏱️  API call took 1.5023s
💾 Response: "Python is a high-level programming language..."

⚡ Cache hit - returning cached result
⏱️  Cached call took 0.0001s

Speedup: 15,023x faster!

2. Cache Statistics:
📊 Cache Stats:
   Hits: 1
   Misses: 1
   Hit Rate: 50.00%
   Size: 245 bytes
   Entries: 1

3. Cache Persistence:
💾 Saving cache to disk: ./cache/llm_cache.json
✅ Cache saved (1 entries)

4. Context Manager:
📂 Loading cache from: ./cache
✅ Cache loaded (1 entries)
⚡ Cache hit from loaded cache
📂 Saving cache: 2 entries
✅ Cache saved successfully

✅ All examples completed!
```

## 🔑 Key Takeaways

1. **Decorators reduce boilerplate** - Cache logic in one place
2. **Context managers ensure cleanup** - Cache always saved
3. **Type hints catch bugs early** - IDE knows types
4. **Generators stream efficiently** - Low memory for large responses
5. **Magic methods make intuitive APIs** - Feels like built-in Python

## 🚀 Next Steps

After completing this project:
1. Move to Module 2: Async Python
2. Try integrating with real OpenAI API
3. Build a CLI tool using this cache
4. Deploy as a reusable package

## 💡 Real-World Applications

This pattern is used in:
- **Development environments** - Cache LLM responses to save costs
- **Testing** - Mock LLM responses deterministically
- **Rate limiting** - Avoid hitting API limits
- **Cost optimization** - Reduce API costs significantly
- **Performance** - Sub-millisecond response times

---

**Start building:** Check out `src/llm_cache.py` for the complete implementation!
