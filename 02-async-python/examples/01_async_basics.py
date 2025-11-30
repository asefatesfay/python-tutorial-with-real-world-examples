"""
Async Basics for Senior Engineers

Python is synchronous by default, unlike Node.js where I/O is async by default.
This example shows the fundamental differences and similarities.

Run: poetry run python 01_async_basics.py
"""

import asyncio
import time
from typing import Any


# ============================================================================
# 1. Sync vs Async - The Fundamental Difference
# ============================================================================

def sync_sleep(duration: float, name: str) -> str:
    """Synchronous function - blocks the entire thread."""
    print(f"[{name}] Starting (sync)")
    time.sleep(duration)  # Blocks!
    print(f"[{name}] Finished (sync)")
    return f"Result from {name}"


async def async_sleep(duration: float, name: str) -> str:
    """Asynchronous function - yields control to event loop."""
    print(f"[{name}] Starting (async)")
    await asyncio.sleep(duration)  # Yields control, non-blocking!
    print(f"[{name}] Finished (async)")
    return f"Result from {name}"


def demo_sync_vs_async():
    """Compare sync and async execution."""
    print("=" * 60)
    print("Sync vs Async Comparison")
    print("=" * 60)
    
    # Synchronous execution - sequential
    print("\n1️⃣ Synchronous (blocks):")
    start = time.time()
    sync_sleep(1.0, "Task 1")
    sync_sleep(1.0, "Task 2")
    sync_sleep(1.0, "Task 3")
    print(f"⏱️  Total time: {time.time() - start:.2f}s\n")
    
    # Asynchronous execution - concurrent
    print("2️⃣ Asynchronous (concurrent):")
    start = time.time()
    asyncio.run(async_demo())
    print(f"⏱️  Total time: {time.time() - start:.2f}s")


async def async_demo():
    """Run async tasks concurrently."""
    # Create tasks that run concurrently
    task1 = asyncio.create_task(async_sleep(1.0, "Task 1"))
    task2 = asyncio.create_task(async_sleep(1.0, "Task 2"))
    task3 = asyncio.create_task(async_sleep(1.0, "Task 3"))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(task1, task2, task3)
    return results


# ============================================================================
# 2. Event Loop - The Heart of Async Python
# ============================================================================

async def simple_coroutine(n: int) -> int:
    """A simple coroutine."""
    await asyncio.sleep(0.5)
    return n * 2


def demo_event_loop():
    """Demonstrate event loop usage."""
    print("\n" + "=" * 60)
    print("Event Loop Management")
    print("=" * 60)
    
    # Method 1: asyncio.run() - Most common (Python 3.7+)
    print("\n1️⃣ Using asyncio.run():")
    result = asyncio.run(simple_coroutine(5))
    print(f"Result: {result}")
    
    # Method 2: Get event loop manually (advanced)
    print("\n2️⃣ Manual event loop:")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(simple_coroutine(10))
        print(f"Result: {result}")
    finally:
        loop.close()
    
    print("\n💡 Use asyncio.run() for most cases!")


# ============================================================================
# 3. Async vs Node.js - Side by Side
# ============================================================================

async def fetch_data(url: str, delay: float) -> dict:
    """
    Simulate HTTP request.
    
    Node.js equivalent:
        async function fetchData(url, delay) {
            await new Promise(resolve => setTimeout(resolve, delay * 1000));
            return { url, data: 'response' };
        }
    """
    await asyncio.sleep(delay)
    return {"url": url, "data": "response"}


async def demo_nodejs_comparison():
    """
    Compare Python asyncio patterns to Node.js.
    
    Node.js:
        const results = await Promise.all([
            fetchData('url1', 0.5),
            fetchData('url2', 0.5),
            fetchData('url3', 0.5)
        ]);
    
    Python:
        results = await asyncio.gather(
            fetch_data('url1', 0.5),
            fetch_data('url2', 0.5),
            fetch_data('url3', 0.5)
        )
    """
    print("\n" + "=" * 60)
    print("Python asyncio vs Node.js")
    print("=" * 60)
    
    print("\n🔄 Concurrent requests (like Promise.all):")
    start = time.time()
    
    results = await asyncio.gather(
        fetch_data("https://api1.com", 0.5),
        fetch_data("https://api2.com", 0.5),
        fetch_data("https://api3.com", 0.5)
    )
    
    duration = time.time() - start
    print(f"✅ Fetched {len(results)} URLs in {duration:.2f}s")
    print(f"💡 Sequential would take {0.5 * 3:.2f}s")


# ============================================================================
# 4. Real-World Example: Parallel Embedding Generation
# ============================================================================

async def generate_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Simulate OpenAI embedding API call.
    In real code, would use aiohttp or httpx.
    """
    # Simulate API latency
    await asyncio.sleep(0.3)
    
    # Simulate embedding (in reality, would call OpenAI)
    embedding_dim = 384
    embedding = [hash(text + str(i)) % 100 / 100.0 for i in range(embedding_dim)]
    
    return embedding


async def batch_embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple documents in parallel.
    
    This is MUCH faster than sequential calls!
    """
    print(f"\n🔢 Generating embeddings for {len(documents)} documents...")
    
    start = time.time()
    
    # Create tasks for parallel execution
    tasks = [generate_embedding(doc) for doc in documents]
    
    # Wait for all embeddings
    embeddings = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    print(f"✅ Generated {len(embeddings)} embeddings in {duration:.2f}s")
    print(f"💡 Sequential would take {0.3 * len(documents):.2f}s")
    print(f"🚀 Speedup: {(0.3 * len(documents)) / duration:.1f}x faster!")
    
    return embeddings


async def demo_real_world_embeddings():
    """Demonstrate real-world async pattern for AI/ML."""
    print("\n" + "=" * 60)
    print("Real-World: Parallel Embedding Generation")
    print("=" * 60)
    
    documents = [
        "Python is a powerful programming language",
        "Machine learning requires large datasets",
        "LangChain simplifies LLM applications",
        "Vector databases enable semantic search",
        "Async programming improves performance",
        "FastAPI is great for ML model serving",
        "RAG combines retrieval and generation",
        "Embeddings capture semantic meaning",
    ]
    
    embeddings = await batch_embed_documents(documents)
    
    print(f"\n📊 Embedding shape: {len(embeddings)} x {len(embeddings[0])}")


# ============================================================================
# 5. Common Pitfall: Calling Async Functions Incorrectly
# ============================================================================

def demo_common_mistakes():
    """Show common async mistakes and how to fix them."""
    print("\n" + "=" * 60)
    print("Common Async Mistakes")
    print("=" * 60)
    
    # ❌ WRONG: Calling async function without await
    print("\n❌ Wrong: Calling async function directly")
    try:
        result = async_sleep(0.1, "test")
        print(f"Result type: {type(result)}")  # It's a coroutine, not the result!
        print("⚠️  This doesn't execute the function!")
    except Exception as e:
        print(f"Error: {e}")
    
    # ✅ CORRECT: Use asyncio.run()
    print("\n✅ Correct: Use asyncio.run()")
    result = asyncio.run(async_sleep(0.1, "test"))
    print(f"Result: {result}")
    
    # ❌ WRONG: Using await outside async function
    print("\n❌ Wrong: Can't use await in sync function")
    print("# await async_sleep(0.1, 'test')  # SyntaxError!")
    
    # ✅ CORRECT: Define async function first
    print("\n✅ Correct: Define async wrapper")
    async def wrapper():
        return await async_sleep(0.1, "test")
    
    result = asyncio.run(wrapper())
    print(f"Result: {result}")


# ============================================================================
# 6. When to Use Async vs Sync
# ============================================================================

def demo_when_to_use_async():
    """Guidelines for when to use async."""
    print("\n" + "=" * 60)
    print("When to Use Async")
    print("=" * 60)
    
    print("""
✅ Use Async For (I/O-bound):
  - HTTP API calls (OpenAI, Anthropic, etc.)
  - Database queries
  - File I/O (with aiofiles)
  - Network operations
  - Multiple concurrent operations

❌ Don't Use Async For (CPU-bound):
  - Heavy computations
  - ML model training
  - Data processing (NumPy operations)
  - Image/video processing
  
  👉 Use multiprocessing for CPU-bound tasks!

💡 Rule of Thumb:
  - If it waits for I/O → Use async
  - If it uses CPU → Use sync or multiprocessing
  - AI/ML: Async for API calls, sync for local inference
""")


# ============================================================================
# Demonstrations
# ============================================================================

def main():
    print("\n🐍 Async Python Basics\n")
    
    # 1. Sync vs Async
    demo_sync_vs_async()
    
    # 2. Event Loop
    demo_event_loop()
    
    # 3. Node.js comparison
    asyncio.run(demo_nodejs_comparison())
    
    # 4. Real-world embeddings
    asyncio.run(demo_real_world_embeddings())
    
    # 5. Common mistakes
    demo_common_mistakes()
    
    # 6. When to use async
    demo_when_to_use_async()
    
    print("\n" + "=" * 60)
    print("✅ Key Takeaways:")
    print("=" * 60)
    print("""
1. Python is sync by default - must opt-in to async
2. Use async/await for I/O-bound operations
3. asyncio.gather() = Promise.all() in Node.js
4. Perfect for parallel API calls (LLMs, embeddings)
5. Use asyncio.run() to execute async functions
6. Don't use async for CPU-bound operations
""")


if __name__ == "__main__":
    main()
