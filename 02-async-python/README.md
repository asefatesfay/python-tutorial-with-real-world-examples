# Module 2: Async Python for AI/ML Engineers

> Master asyncio patterns for parallel API calls, data fetching, and concurrent processing

## 🎯 Learning Objectives

By the end of this module, you'll understand:
- `asyncio` vs Node.js event loop (similarities and differences)
- `async/await` syntax and patterns
- Concurrent API calls (OpenAI, Anthropic, embeddings)
- Async context managers and generators
- Running async code in sync contexts
- Common pitfalls and best practices

## 📋 Table of Contents

1. [Async Basics vs Node.js](#1-async-basics-vs-nodejs)
2. [Async Functions & Await](#2-async-functions--await)
3. [Concurrent Execution](#3-concurrent-execution)
4. [Async Context Managers](#4-async-context-managers)
5. [Async Generators](#5-async-generators)
6. [Real-World Patterns](#6-real-world-patterns)

---

## 1. Async Basics vs Node.js

### Coming from Node.js

**Node.js:**
```javascript
// JavaScript is async by default for I/O
fetch('https://api.openai.com/v1/embeddings')
  .then(res => res.json())
  .then(data => console.log(data));

// Or with async/await
async function getEmbedding() {
  const res = await fetch('https://api.openai.com/v1/embeddings');
  return await res.json();
}
```

**Python:**
```python
# Python is sync by default - must opt-in to async
import asyncio
import aiohttp

async def get_embedding():
    async with aiohttp.ClientSession() as session:
        async with session.post('https://api.openai.com/v1/embeddings') as resp:
            return await resp.json()

# Must run in event loop
asyncio.run(get_embedding())
```

### Key Differences

| Feature | Node.js | Python asyncio |
|---------|---------|----------------|
| **Default** | Async I/O everywhere | Sync by default |
| **Event Loop** | Built-in, always running | Must create explicitly |
| **Syntax** | `async/await` | `async/await` (similar!) |
| **HTTP** | `fetch` built-in | Need `aiohttp` or `httpx` |
| **Concurrency** | Single-threaded | Single-threaded (similar) |
| **Parallelism** | Worker threads | `multiprocessing` |

### Examples

See: [`examples/01_async_basics.py`](./examples/01_async_basics.py)

---

## 2. Async Functions & Await

### Defining Async Functions

```python
# Regular function (sync)
def sync_function():
    return "result"

# Async function (coroutine)
async def async_function():
    return "result"

# Calling async functions
result = sync_function()  # Direct call
result = await async_function()  # Must await
```

### Key Concepts

- **Coroutine** - Function defined with `async def`
- **Await** - Suspend execution until coroutine completes
- **Event Loop** - Manages and schedules coroutines
- **Task** - Wrapper around coroutine for concurrent execution

### Examples

See: [`examples/02_async_await.py`](./examples/02_async_await.py)

---

## 3. Concurrent Execution

### Running Multiple Tasks Concurrently

**Node.js:**
```javascript
// Promise.all for concurrent execution
const [emb1, emb2, emb3] = await Promise.all([
  getEmbedding(text1),
  getEmbedding(text2),
  getEmbedding(text3)
]);
```

**Python:**
```python
# asyncio.gather for concurrent execution
emb1, emb2, emb3 = await asyncio.gather(
    get_embedding(text1),
    get_embedding(text2),
    get_embedding(text3)
)
```

### Real-World Use Cases for AI/ML

- **Parallel embedding generation** - Process multiple texts simultaneously
- **Concurrent vector search** - Query multiple vector stores
- **Batch API calls** - Call OpenAI/Anthropic in parallel
- **Multi-model inference** - Query multiple LLMs concurrently
- **Data fetching** - Download datasets from multiple sources

### Examples

See: [`examples/03_concurrent_execution.py`](./examples/03_concurrent_execution.py)

---

## 4. Async Context Managers

### Async with Statement

```python
# Sync context manager
with open('file.txt') as f:
    data = f.read()

# Async context manager
async with aiohttp.ClientSession() as session:
    async with session.get('https://api.example.com') as resp:
        data = await resp.json()
```

### Real-World Examples

- **HTTP clients** - `aiohttp.ClientSession`
- **Database connections** - `asyncpg.connect()`
- **Vector store clients** - Async ChromaDB, Pinecone
- **File I/O** - `aiofiles.open()`

### Examples

See: [`examples/04_async_context_managers.py`](./examples/04_async_context_managers.py)

---

## 5. Async Generators

### Streaming Data

```python
# Sync generator
def stream_data():
    for i in range(10):
        yield i

# Async generator
async def stream_data_async():
    for i in range(10):
        await asyncio.sleep(0.1)
        yield i

# Usage
async for item in stream_data_async():
    print(item)
```

### Real-World Use Cases

- **Streaming LLM responses** - OpenAI streaming API
- **Processing large datasets** - Read files asynchronously
- **Real-time data feeds** - WebSocket streams
- **Batch processing** - Async batch generators

### Examples

See: [`examples/05_async_generators.py`](./examples/05_async_generators.py)

---

## 6. Real-World Patterns

### Pattern 1: Parallel Embedding Generation

```python
async def batch_embed_documents(documents: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple documents in parallel."""
    tasks = [get_embedding(doc) for doc in documents]
    return await asyncio.gather(*tasks)
```

### Pattern 2: Rate-Limited API Calls

```python
async def rate_limited_calls(items: list[str], rpm: int = 60):
    """Process items with rate limiting."""
    semaphore = asyncio.Semaphore(rpm // 60)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)
    
    return await asyncio.gather(*[process_with_limit(item) for item in items])
```

### Pattern 3: Concurrent Vector Search

```python
async def search_multiple_stores(query: str) -> dict:
    """Search multiple vector stores concurrently."""
    results = await asyncio.gather(
        chroma_search(query),
        pinecone_search(query),
        weaviate_search(query),
        return_exceptions=True  # Don't fail if one store fails
    )
    return combine_results(results)
```

### Examples

See: [`examples/06_real_world_patterns.py`](./examples/06_real_world_patterns.py)

---

## 🏗️ Mini Project: Async Document Processor

Build a high-performance document processing pipeline that:
- Fetches documents from multiple sources concurrently
- Generates embeddings in parallel
- Stores to vector database asynchronously
- Handles rate limiting and retries
- Streams progress updates

**Location:** [`project/`](./project/)

---

## 🎯 Exercises

1. **Parallel API Calls** - Fetch data from 10 APIs concurrently
2. **Async Rate Limiter** - Implement semaphore-based rate limiting
3. **Streaming Processor** - Process large file with async generator
4. **Multi-Model Query** - Query GPT-4 and Claude simultaneously
5. **Resilient Pipeline** - Handle failures gracefully with retry logic

**Location:** [`exercises/`](./exercises/)

---

## 🔑 Key Takeaways

1. **Python is sync by default** - Must explicitly use `async/await`
2. **Similar to Node.js** - But need to manage event loop
3. **Use for I/O-bound tasks** - Not CPU-bound (use multiprocessing)
4. **Concurrent != Parallel** - Still single-threaded
5. **Perfect for AI/ML APIs** - Multiple LLM calls, embedding generation
6. **Use `asyncio.gather()`** - Like `Promise.all()` in JS

---

## ⏭️ Next Module

[Module 3: Data Structures for ML](../03-data-structures/) - Learn NumPy and Pandas for data manipulation

---

## 📚 Additional Resources

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python - Async IO](https://realpython.com/async-io-python/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [asyncio Cheatsheet](https://www.pythonsheets.com/notes/python-asyncio.html)
