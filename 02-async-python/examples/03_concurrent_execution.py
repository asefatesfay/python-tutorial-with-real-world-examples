"""
Concurrent Execution for AI/ML Engineers

Real-world patterns for parallel processing in AI/ML applications:
- Batch embedding generation with OpenAI
- Multi-provider LLM queries (GPT-4 + Claude simultaneously)
- Parallel vector database searches
- Concurrent document processing

Run: poetry run python 03_concurrent_execution.py
"""

import asyncio
import time
from typing import Any
from dataclasses import dataclass
import json


# ============================================================================
# 1. Real-World: Parallel Embedding Generation with Rate Limiting
# ============================================================================

@dataclass
class EmbeddingResponse:
    """OpenAI-style embedding response."""
    text: str
    embedding: list[float]
    model: str
    tokens: int


class OpenAIEmbeddingClient:
    """Simulated OpenAI embedding client with realistic behavior."""
    
    def __init__(self, api_key: str, rpm_limit: int = 3500):
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self.requests_made = 0
    
    async def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> EmbeddingResponse:
        """
        Simulate OpenAI embedding API call.
        Real version would use httpx or aiohttp.
        
        In production:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"input": text, "model": model}
                )
                return response.json()
        """
        # Simulate API latency (200-500ms typical)
        await asyncio.sleep(0.3)
        
        self.requests_made += 1
        
        # Simulate realistic embedding
        embedding_dim = 1536 if "3-small" in model else 3072
        embedding = [hash(text + str(i)) % 100 / 100.0 for i in range(embedding_dim)]
        
        return EmbeddingResponse(
            text=text,
            embedding=embedding,
            model=model,
            tokens=len(text.split())
        )


async def batch_embed_with_rate_limit(
    documents: list[str],
    client: OpenAIEmbeddingClient,
    batch_size: int = 100,
    rpm: int = 3500
) -> list[EmbeddingResponse]:
    """
    Generate embeddings with rate limiting.
    
    Real-world scenario:
    - OpenAI has 3,500 RPM limit on tier 1
    - Need to process 10,000 documents
    - Must batch and rate-limit to avoid 429 errors
    """
    print(f"\n🔢 Processing {len(documents)} documents")
    print(f"📊 Batch size: {batch_size}, Rate limit: {rpm} RPM")
    
    # Semaphore limits concurrent requests
    semaphore = asyncio.Semaphore(rpm // 60)  # Convert RPM to concurrent limit
    
    async def embed_with_semaphore(text: str) -> EmbeddingResponse:
        async with semaphore:
            return await client.create_embedding(text)
    
    # Process in batches to manage memory
    all_embeddings = []
    start_time = time.time()
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"⚡ Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # Process batch concurrently
        tasks = [embed_with_semaphore(doc) for doc in batch]
        embeddings = await asyncio.gather(*tasks)
        all_embeddings.extend(embeddings)
        
        # Show progress
        elapsed = time.time() - start_time
        rate = len(all_embeddings) / elapsed if elapsed > 0 else 0
        print(f"  ✅ {len(all_embeddings)}/{len(documents)} complete ({rate:.1f} docs/sec)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Complete! Processed {len(documents)} docs in {total_time:.2f}s")
    print(f"📈 Average: {len(documents)/total_time:.1f} documents/second")
    
    return all_embeddings


async def demo_parallel_embeddings():
    """Real-world example: RAG system ingestion."""
    print("=" * 70)
    print("Real-World: Parallel Document Embedding for RAG")
    print("=" * 70)
    
    # Simulate ingesting documentation for RAG
    documents = [
        "Python async programming enables concurrent I/O operations",
        "FastAPI is a modern web framework for building APIs",
        "LangChain provides abstractions for LLM applications",
        "Vector databases store embeddings for semantic search",
        "ChromaDB is an open-source embedding database",
        "Retrieval Augmented Generation improves LLM responses",
        "Prompt engineering is crucial for LLM performance",
        "Token limits constrain context window size",
        "Fine-tuning adapts models to specific domains",
        "Few-shot learning provides examples in prompts",
        "Chain-of-thought prompting improves reasoning",
        "RAG systems combine retrieval with generation",
        "Embeddings capture semantic meaning of text",
        "Cosine similarity measures vector similarity",
        "Semantic search finds relevant documents",
    ] * 2  # 30 documents
    
    client = OpenAIEmbeddingClient(api_key="sk-test-key")
    
    # Compare sequential vs parallel
    print("\n📊 Comparison:")
    print(f"   Sequential time: {len(documents) * 0.3:.1f}s")
    print(f"   Parallel time: ~{max(len(documents) * 0.3 / 50, 0.3):.1f}s")
    
    embeddings = await batch_embed_with_rate_limit(
        documents,
        client,
        batch_size=10,
        rpm=3500
    )
    
    print(f"\n💾 Generated {len(embeddings)} embeddings")
    print(f"📐 Dimension: {len(embeddings[0].embedding)}")


# ============================================================================
# 2. Real-World: Multi-Provider LLM Queries
# ============================================================================

@dataclass
class LLMResponse:
    """Standard LLM response format."""
    provider: str
    model: str
    content: str
    latency: float
    tokens: int


class LLMClient:
    """Multi-provider LLM client."""
    
    async def call_openai(self, prompt: str, model: str = "gpt-4") -> LLMResponse:
        """
        Call OpenAI API.
        
        Real implementation:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
        """
        start = time.time()
        await asyncio.sleep(1.2)  # GPT-4 typical latency
        latency = time.time() - start
        
        return LLMResponse(
            provider="OpenAI",
            model=model,
            content=f"OpenAI {model} response to: {prompt[:50]}...",
            latency=latency,
            tokens=100
        )
    
    async def call_anthropic(self, prompt: str, model: str = "claude-3-5-sonnet-20241022") -> LLMResponse:
        """
        Call Anthropic API.
        
        Real implementation:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024
                    }
                )
        """
        start = time.time()
        await asyncio.sleep(0.8)  # Claude typical latency
        latency = time.time() - start
        
        return LLMResponse(
            provider="Anthropic",
            model=model,
            content=f"Claude response to: {prompt[:50]}...",
            latency=latency,
            tokens=120
        )
    
    async def call_local_llm(self, prompt: str, model: str = "llama-3.1-8b") -> LLMResponse:
        """
        Call local LLM (Ollama, vLLM, etc.).
        
        Real implementation:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt}
                )
        """
        start = time.time()
        await asyncio.sleep(0.5)  # Local model faster
        latency = time.time() - start
        
        return LLMResponse(
            provider="Local",
            model=model,
            content=f"Local LLM response to: {prompt[:50]}...",
            latency=latency,
            tokens=90
        )


async def query_multiple_llms(prompt: str, client: LLMClient) -> dict[str, LLMResponse]:
    """
    Query multiple LLMs simultaneously for comparison or voting.
    
    Real-world use cases:
    - Compare responses from different models
    - Implement LLM voting/consensus
    - Failover if one provider is down
    - Choose fastest response
    """
    print(f"\n🤖 Querying multiple LLMs concurrently")
    print(f"📝 Prompt: {prompt[:60]}...")
    
    start = time.time()
    
    # Launch all queries simultaneously
    results = await asyncio.gather(
        client.call_openai(prompt, "gpt-4"),
        client.call_anthropic(prompt, "claude-3-5-sonnet-20241022"),
        client.call_local_llm(prompt, "llama-3.1-8b"),
        return_exceptions=True  # Don't fail if one provider fails
    )
    
    total_time = time.time() - start
    
    # Process results
    responses = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Provider failed: {result}")
            continue
        responses[result.provider] = result
        print(f"✅ {result.provider} ({result.model}): {result.latency:.2f}s")
    
    print(f"\n⏱️  Total time: {total_time:.2f}s (parallel)")
    print(f"💡 Sequential would take: {sum(r.latency for r in responses.values()):.2f}s")
    
    return responses


async def demo_multi_provider():
    """Real-world: LLM comparison or voting system."""
    print("\n" + "=" * 70)
    print("Real-World: Multi-Provider LLM Query (Comparison/Voting)")
    print("=" * 70)
    
    client = LLMClient()
    prompt = "Explain the concept of retrieval augmented generation (RAG) in 2 sentences."
    
    responses = await query_multiple_llms(prompt, client)
    
    # Find fastest response
    fastest = min(responses.values(), key=lambda r: r.latency)
    print(f"\n🏆 Fastest: {fastest.provider} ({fastest.latency:.2f}s)")
    
    # Could implement voting, averaging, etc.
    print("\n💡 Use cases:")
    print("  - Compare model outputs for quality")
    print("  - Implement consensus/voting system")
    print("  - Failover if primary provider down")
    print("  - Choose fastest response (race condition)")


# ============================================================================
# 3. Real-World: Parallel Vector Database Search
# ============================================================================

@dataclass
class SearchResult:
    """Vector search result."""
    document_id: str
    score: float
    content: str
    metadata: dict


class VectorStoreClient:
    """Multi-store vector search client."""
    
    async def search_chroma(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search ChromaDB."""
        await asyncio.sleep(0.2)
        return [
            SearchResult(
                document_id=f"chroma_{i}",
                score=0.9 - i * 0.1,
                content=f"ChromaDB result {i}",
                metadata={"source": "chroma"}
            )
            for i in range(top_k)
        ]
    
    async def search_pinecone(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search Pinecone."""
        await asyncio.sleep(0.3)
        return [
            SearchResult(
                document_id=f"pinecone_{i}",
                score=0.85 - i * 0.1,
                content=f"Pinecone result {i}",
                metadata={"source": "pinecone"}
            )
            for i in range(top_k)
        ]
    
    async def search_weaviate(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search Weaviate."""
        await asyncio.sleep(0.25)
        return [
            SearchResult(
                document_id=f"weaviate_{i}",
                score=0.88 - i * 0.1,
                content=f"Weaviate result {i}",
                metadata={"source": "weaviate"}
            )
            for i in range(top_k)
        ]


async def search_all_stores(
    query: str,
    client: VectorStoreClient,
    top_k: int = 5
) -> dict[str, list[SearchResult]]:
    """
    Search multiple vector stores concurrently.
    
    Real-world use cases:
    - Federated search across multiple databases
    - Redundancy (multiple stores with same data)
    - Different embedding models per store
    - Merge results from multiple sources
    """
    print(f"\n🔍 Searching multiple vector stores")
    print(f"📝 Query: {query}")
    
    start = time.time()
    
    # Search all stores concurrently
    chroma, pinecone, weaviate = await asyncio.gather(
        client.search_chroma(query, top_k),
        client.search_pinecone(query, top_k),
        client.search_weaviate(query, top_k),
        return_exceptions=True
    )
    
    total_time = time.time() - start
    
    results = {}
    if not isinstance(chroma, Exception):
        results["chroma"] = chroma
        print(f"✅ ChromaDB: {len(chroma)} results")
    if not isinstance(pinecone, Exception):
        results["pinecone"] = pinecone
        print(f"✅ Pinecone: {len(pinecone)} results")
    if not isinstance(weaviate, Exception):
        results["weaviate"] = weaviate
        print(f"✅ Weaviate: {len(weaviate)} results")
    
    print(f"\n⏱️  Total time: {total_time:.2f}s (parallel)")
    print(f"💡 Sequential would take: {0.2 + 0.3 + 0.25:.2f}s")
    
    return results


def merge_results(all_results: dict[str, list[SearchResult]], top_k: int = 5) -> list[SearchResult]:
    """
    Merge results from multiple stores using reciprocal rank fusion.
    
    This is a common technique in RAG systems with multiple sources.
    """
    # Reciprocal rank fusion
    score_map: dict[str, float] = {}
    
    for source, results in all_results.items():
        for rank, result in enumerate(results, 1):
            # RRF: 1 / (rank + 60)
            rrf_score = 1.0 / (rank + 60)
            doc_key = result.content  # In reality, use document_id
            score_map[doc_key] = score_map.get(doc_key, 0) + rrf_score
    
    # Sort by merged score
    merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [
        SearchResult(
            document_id=f"merged_{i}",
            score=score,
            content=content,
            metadata={"fusion": "rrf"}
        )
        for i, (content, score) in enumerate(merged)
    ]


async def demo_federated_search():
    """Real-world: Federated vector search for RAG."""
    print("\n" + "=" * 70)
    print("Real-World: Federated Vector Store Search")
    print("=" * 70)
    
    client = VectorStoreClient()
    query = "How do I implement async patterns in Python?"
    
    results = await search_all_stores(query, client, top_k=3)
    
    # Merge results
    merged = merge_results(results, top_k=5)
    print(f"\n🔗 Merged top {len(merged)} results using RRF")
    for i, result in enumerate(merged, 1):
        print(f"   {i}. Score: {result.score:.4f}")
    
    print("\n💡 Use cases:")
    print("  - Search multiple vector stores simultaneously")
    print("  - Implement redundancy/failover")
    print("  - Different embedding models per store")
    print("  - Merge results using RRF or other algorithms")


# ============================================================================
# 4. Real-World: Complete RAG Pipeline with Async
# ============================================================================

async def rag_pipeline(
    user_query: str,
    embedding_client: OpenAIEmbeddingClient,
    vector_client: VectorStoreClient,
    llm_client: LLMClient
) -> dict:
    """
    Complete async RAG pipeline:
    1. Embed query
    2. Search vector stores (parallel)
    3. Generate response with LLM
    
    All steps optimized with async.
    """
    print("\n" + "=" * 70)
    print("Complete Async RAG Pipeline")
    print("=" * 70)
    print(f"\n❓ User Query: {user_query}")
    
    start_time = time.time()
    
    # Step 1: Embed query
    print("\n1️⃣ Embedding query...")
    query_embedding = await embedding_client.create_embedding(user_query)
    print(f"   ✅ Embedded ({len(query_embedding.embedding)} dims)")
    
    # Step 2: Search vector stores (parallel)
    print("\n2️⃣ Searching vector stores...")
    search_results = await search_all_stores(user_query, vector_client, top_k=3)
    merged_results = merge_results(search_results, top_k=5)
    print(f"   ✅ Found {len(merged_results)} relevant documents")
    
    # Step 3: Build context from results
    context = "\n".join([r.content for r in merged_results[:3]])
    augmented_prompt = f"""Context from knowledge base:
{context}

User question: {user_query}

Please answer based on the context provided."""
    
    # Step 4: Generate response (could query multiple models)
    print("\n3️⃣ Generating response...")
    responses = await asyncio.gather(
        llm_client.call_openai(augmented_prompt),
        llm_client.call_anthropic(augmented_prompt),
        return_exceptions=True
    )
    
    # Choose best response (or merge)
    best_response = responses[0] if not isinstance(responses[0], Exception) else responses[1]
    print(f"   ✅ Generated response ({best_response.tokens} tokens)")
    
    total_time = time.time() - start_time
    
    result = {
        "query": user_query,
        "context_docs": len(merged_results),
        "response": best_response.content,
        "latency": total_time,
        "sources": [r.document_id for r in merged_results[:3]]
    }
    
    print(f"\n⏱️  Total pipeline latency: {total_time:.2f}s")
    print(f"📊 Response: {best_response.content[:80]}...")
    
    return result


async def demo_complete_rag():
    """Real-world: Production RAG pipeline."""
    embedding_client = OpenAIEmbeddingClient("sk-test")
    vector_client = VectorStoreClient()
    llm_client = LLMClient()
    
    result = await rag_pipeline(
        "What are best practices for async Python in production?",
        embedding_client,
        vector_client,
        llm_client
    )
    
    print("\n💡 Pipeline Benefits:")
    print("  - Parallel vector store searches")
    print("  - Concurrent LLM calls for redundancy")
    print("  - Non-blocking I/O throughout")
    print("  - Easy to add caching, retries, timeouts")


# ============================================================================
# Demonstrations
# ============================================================================

async def main():
    """Run all real-world examples."""
    print("\n🚀 Real-World Async Patterns for AI/ML\n")
    
    # 1. Parallel embeddings with rate limiting
    await demo_parallel_embeddings()
    
    # 2. Multi-provider LLM queries
    await demo_multi_provider()
    
    # 3. Federated vector search
    await demo_federated_search()
    
    # 4. Complete RAG pipeline
    await demo_complete_rag()
    
    print("\n" + "=" * 70)
    print("✅ Key Real-World Patterns")
    print("=" * 70)
    print("""
1. Rate-limited batch processing (embeddings)
2. Multi-provider queries (comparison, voting, failover)
3. Federated search (multiple vector stores)
4. Complete async RAG pipeline
5. Always handle exceptions gracefully
6. Use semaphores for rate limiting
7. return_exceptions=True for resilience
""")


if __name__ == "__main__":
    asyncio.run(main())
