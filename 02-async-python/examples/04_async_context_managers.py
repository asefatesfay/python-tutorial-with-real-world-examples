"""
Async Context Managers for AI/ML Engineers

Real-world patterns for managing resources in async AI/ML applications:
- HTTP client sessions for API calls
- Database connection pools
- Vector store client lifecycle
- Streaming LLM responses
- Resource cleanup and error handling

Run: poetry run python 04_async_context_managers.py
"""

import asyncio
import time
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager


# ============================================================================
# 1. Real-World: HTTP Client Session Management
# ============================================================================

class AsyncHTTPClient:
    """
    Async HTTP client for AI/ML APIs.
    
    Real implementation would use httpx or aiohttp:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data)
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session_id = id(self)
        self._connected = False
    
    async def __aenter__(self):
        """Setup - create connection pool."""
        print(f"🔌 Opening HTTP session {self.session_id}")
        await asyncio.sleep(0.1)  # Simulate connection setup
        self._connected = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup - close connection pool."""
        print(f"🔌 Closing HTTP session {self.session_id}")
        await asyncio.sleep(0.05)  # Simulate cleanup
        self._connected = False
        return False
    
    async def post(self, endpoint: str, data: dict) -> dict:
        """Make POST request."""
        if not self._connected:
            raise RuntimeError("Session not connected!")
        
        await asyncio.sleep(0.2)  # Simulate API call
        return {"status": "success", "data": data}


async def demo_http_session():
    """
    Real-world: Reusing HTTP session for multiple API calls.
    
    Important: Reusing session is MUCH faster than creating new one each time!
    """
    print("=" * 70)
    print("Real-World: HTTP Session Management for API Calls")
    print("=" * 70)
    
    # ❌ BAD: Creating new session for each request
    print("\n❌ Bad: New session per request")
    start = time.time()
    for i in range(3):
        async with AsyncHTTPClient("https://api.openai.com", "sk-test") as client:
            await client.post("/v1/embeddings", {"text": f"doc {i}"})
    bad_time = time.time() - start
    print(f"   Time: {bad_time:.2f}s")
    
    # ✅ GOOD: Reusing session for multiple requests
    print("\n✅ Good: Reuse session for multiple requests")
    start = time.time()
    async with AsyncHTTPClient("https://api.openai.com", "sk-test") as client:
        # Make multiple requests with same session
        tasks = [
            client.post("/v1/embeddings", {"text": f"doc {i}"})
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
    good_time = time.time() - start
    print(f"   Time: {good_time:.2f}s")
    
    print(f"\n🚀 Speedup: {bad_time/good_time:.1f}x faster with session reuse!")
    print("\n💡 Key insight: Connection pooling is critical for performance")


# ============================================================================
# 2. Real-World: Vector Database Client with Connection Pool
# ============================================================================

class AsyncVectorStoreClient:
    """
    Async vector store client (ChromaDB, Pinecone, etc.).
    
    Real implementation:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(allow_reset=True)
        )
    """
    
    def __init__(self, url: str, collection: str):
        self.url = url
        self.collection = collection
        self._connection = None
        self._pool_size = 10
    
    async def __aenter__(self):
        """Initialize connection pool."""
        print(f"📚 Connecting to vector store: {self.url}/{self.collection}")
        await asyncio.sleep(0.2)  # Simulate connection
        self._connection = {"pool": self._pool_size, "active": True}
        print(f"✅ Connection pool established (size: {self._pool_size})")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close connections gracefully."""
        if exc_type:
            print(f"⚠️  Exception occurred: {exc_val}")
            # Could implement rollback logic here
        
        print(f"📚 Closing vector store connections...")
        await asyncio.sleep(0.1)
        self._connection = None
        print("✅ All connections closed")
        return False
    
    async def upsert(self, doc_id: str, embedding: list[float], metadata: dict) -> None:
        """Insert or update document."""
        if not self._connection:
            raise RuntimeError("Not connected!")
        
        await asyncio.sleep(0.05)  # Simulate write
        print(f"   💾 Upserted {doc_id}: {len(embedding)}-dim embedding")
    
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar vectors."""
        if not self._connection:
            raise RuntimeError("Not connected!")
        
        await asyncio.sleep(0.1)  # Simulate search
        return [{"id": f"doc_{i}", "score": 0.9 - i*0.1} for i in range(top_k)]


async def demo_vector_store_lifecycle():
    """
    Real-world: Managing vector store connections in RAG pipeline.
    """
    print("\n" + "=" * 70)
    print("Real-World: Vector Store Connection Lifecycle")
    print("=" * 70)
    
    # Typical RAG ingestion workflow
    documents = [
        ("doc_1", [0.1] * 384, {"source": "manual"}),
        ("doc_2", [0.2] * 384, {"source": "api"}),
        ("doc_3", [0.3] * 384, {"source": "crawl"}),
    ]
    
    # Connection automatically managed
    async with AsyncVectorStoreClient("http://localhost:8000", "knowledge_base") as store:
        print("\n1️⃣ Ingesting documents...")
        
        # Batch upsert with concurrency
        tasks = [
            store.upsert(doc_id, embedding, metadata)
            for doc_id, embedding, metadata in documents
        ]
        await asyncio.gather(*tasks)
        
        print("\n2️⃣ Searching...")
        query_emb = [0.15] * 384
        results = await store.search(query_emb, top_k=3)
        print(f"   🔍 Found {len(results)} results")
    
    # Connection automatically closed even if exception occurs
    print("\n💡 Connection automatically closed - no memory leaks!")


# ============================================================================
# 3. Real-World: Streaming LLM Responses with Context Manager
# ============================================================================

class StreamingLLMClient:
    """
    Streaming LLM client for real-time responses.
    
    Real implementation with OpenAI:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            print(chunk.choices[0].delta.content, end="")
    """
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self._stream_active = False
    
    async def __aenter__(self):
        """Initialize streaming connection."""
        print(f"🌊 Opening stream for {self.model}...")
        await asyncio.sleep(0.1)
        self._stream_active = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close stream gracefully."""
        print(f"\n🌊 Closing stream...")
        self._stream_active = False
        return False
    
    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        if not self._stream_active:
            raise RuntimeError("Stream not active!")
        
        # Simulate streaming tokens
        response = f"Here's a detailed answer about {prompt[:20]}..."
        words = response.split()
        
        for word in words:
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield word + " "


async def demo_streaming_llm():
    """
    Real-world: Streaming LLM responses for better UX.
    
    Benefits:
    - Lower perceived latency
    - Better user experience
    - Can process tokens as they arrive
    """
    print("\n" + "=" * 70)
    print("Real-World: Streaming LLM Responses")
    print("=" * 70)
    
    prompt = "Explain async context managers in Python"
    
    print(f"\n📝 Prompt: {prompt}")
    print("💬 Response: ", end="")
    
    async with StreamingLLMClient("gpt-4", "sk-test") as client:
        async for token in client.stream_response(prompt):
            print(token, end="", flush=True)
    
    print("\n\n💡 Streaming gives immediate feedback to users!")


# ============================================================================
# 4. Real-World: Multiple Resources with Nested Context Managers
# ============================================================================

@asynccontextmanager
async def rag_pipeline_context(
    vector_store_url: str,
    llm_api_key: str
):
    """
    Context manager for complete RAG pipeline resources.
    
    Manages:
    - HTTP client for API calls
    - Vector store connection
    - LLM client
    - Proper cleanup order
    """
    print("🚀 Initializing RAG pipeline...")
    
    # Setup phase - initialize all resources
    http_client = AsyncHTTPClient("https://api.openai.com", llm_api_key)
    vector_store = AsyncVectorStoreClient(vector_store_url, "docs")
    llm_client = StreamingLLMClient("gpt-4", llm_api_key)
    
    try:
        # Enter all contexts
        async with http_client, vector_store, llm_client:
            print("✅ All resources initialized\n")
            
            # Yield resources to caller
            yield {
                "http": http_client,
                "vector_store": vector_store,
                "llm": llm_client
            }
    
    finally:
        # Cleanup automatically handled by context managers
        print("\n🧹 Cleaning up all resources...")


async def demo_nested_contexts():
    """
    Real-world: Managing multiple resources in production RAG system.
    """
    print("\n" + "=" * 70)
    print("Real-World: Multiple Resources in RAG Pipeline")
    print("=" * 70)
    
    async with rag_pipeline_context("http://localhost:8000", "sk-test") as pipeline:
        # All resources available
        http = pipeline["http"]
        vector_store = pipeline["vector_store"]
        llm = pipeline["llm"]
        
        # Execute RAG workflow
        print("1️⃣ Fetch external data...")
        data = await http.post("/fetch", {"url": "https://docs.example.com"})
        
        print("2️⃣ Store in vector database...")
        await vector_store.upsert("doc_1", [0.5]*384, {"source": "api"})
        
        print("3️⃣ Search and generate...")
        results = await vector_store.search([0.5]*384)
        
        print("4️⃣ Stream response...")
        async for token in llm.stream_response("summarize"):
            pass  # Would print tokens
    
    # All resources automatically cleaned up
    print("\n💡 Clean shutdown - no resource leaks!")


# ============================================================================
# 5. Real-World: Error Handling with Async Context Managers
# ============================================================================

class ResilientVectorStore:
    """Vector store with automatic retry and fallback."""
    
    def __init__(self, primary_url: str, fallback_url: str):
        self.primary_url = primary_url
        self.fallback_url = fallback_url
        self._using_fallback = False
    
    async def __aenter__(self):
        """Try primary, fallback to secondary if needed."""
        try:
            print(f"🔌 Connecting to primary: {self.primary_url}")
            await asyncio.sleep(0.1)
            # Simulate connection failure
            if "fail" in self.primary_url:
                raise ConnectionError("Primary unavailable")
            print("✅ Connected to primary")
            
        except ConnectionError as e:
            print(f"⚠️  Primary failed: {e}")
            print(f"🔄 Falling back to: {self.fallback_url}")
            await asyncio.sleep(0.1)
            self._using_fallback = True
            print("✅ Connected to fallback")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup with error reporting."""
        if exc_type:
            print(f"❌ Operation failed: {exc_val}")
            # Could send alerts, log errors, etc.
        
        where = "fallback" if self._using_fallback else "primary"
        print(f"🔌 Disconnecting from {where}")
        return False
    
    async def query(self, text: str) -> list:
        """Query with automatic failover."""
        source = "fallback" if self._using_fallback else "primary"
        print(f"   🔍 Querying {source}...")
        await asyncio.sleep(0.1)
        return ["result"]


async def demo_error_handling():
    """
    Real-world: Resilient systems with automatic failover.
    """
    print("\n" + "=" * 70)
    print("Real-World: Error Handling and Failover")
    print("=" * 70)
    
    # Scenario 1: Primary works
    print("\n📊 Scenario 1: Primary available")
    async with ResilientVectorStore("http://primary:8000", "http://backup:8000") as store:
        results = await store.query("test query")
    
    # Scenario 2: Primary fails, use fallback
    print("\n📊 Scenario 2: Primary fails, use fallback")
    async with ResilientVectorStore("http://primary-fail:8000", "http://backup:8000") as store:
        results = await store.query("test query")
    
    print("\n💡 Production systems need failover and error handling!")


# ============================================================================
# 6. Real-World: Resource Pool with Async Context Manager
# ============================================================================

class LLMClientPool:
    """
    Pool of LLM clients for high-throughput applications.
    
    Real-world use case:
    - Processing thousands of requests/minute
    - Need connection pooling to manage resources
    - Implement rate limiting and backpressure
    """
    
    def __init__(self, pool_size: int = 10, rpm_limit: int = 3500):
        self.pool_size = pool_size
        self.rpm_limit = rpm_limit
        self._semaphore = None
        self._clients_active = 0
    
    async def __aenter__(self):
        """Initialize pool."""
        print(f"🏊 Initializing LLM client pool (size: {self.pool_size})")
        self._semaphore = asyncio.Semaphore(self.pool_size)
        self._clients_active = self.pool_size
        print(f"✅ Pool ready - {self._clients_active} clients available")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Drain pool gracefully."""
        print(f"🏊 Draining client pool...")
        await asyncio.sleep(0.2)  # Wait for in-flight requests
        self._clients_active = 0
        print("✅ Pool drained")
        return False
    
    async def call(self, prompt: str) -> str:
        """Make LLM call with pooling."""
        async with self._semaphore:  # Acquire from pool
            await asyncio.sleep(0.2)
            return f"Response to: {prompt[:30]}"


async def demo_client_pool():
    """
    Real-world: High-throughput LLM processing with connection pooling.
    """
    print("\n" + "=" * 70)
    print("Real-World: Connection Pooling for High Throughput")
    print("=" * 70)
    
    prompts = [f"Process document {i}" for i in range(20)]
    
    async with LLMClientPool(pool_size=5, rpm_limit=3500) as pool:
        print(f"\n📊 Processing {len(prompts)} prompts with pool...")
        
        # Process all prompts - pool manages concurrency
        tasks = [pool.call(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        print(f"✅ Processed {len(results)} prompts")
    
    print("\n💡 Pool prevents overwhelming API with too many connections!")


# ============================================================================
# Demonstrations
# ============================================================================

async def main():
    """Run all real-world examples."""
    print("\n🔧 Async Context Managers for Production AI/ML\n")
    
    # 1. HTTP session management
    await demo_http_session()
    
    # 2. Vector store lifecycle
    await demo_vector_store_lifecycle()
    
    # 3. Streaming LLM
    await demo_streaming_llm()
    
    # 4. Nested contexts (complete pipeline)
    await demo_nested_contexts()
    
    # 5. Error handling and failover
    await demo_error_handling()
    
    # 6. Connection pooling
    await demo_client_pool()
    
    print("\n" + "=" * 70)
    print("✅ Key Production Patterns")
    print("=" * 70)
    print("""
1. Reuse HTTP sessions - MUCH faster than creating new ones
2. Use context managers for resource cleanup
3. Implement failover for resilience
4. Connection pooling for high throughput
5. Streaming for better UX
6. Proper error handling is critical
7. Always clean up resources in production
""")


if __name__ == "__main__":
    asyncio.run(main())
