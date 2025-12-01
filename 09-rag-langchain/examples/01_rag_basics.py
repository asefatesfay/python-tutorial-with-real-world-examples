"""
RAG (Retrieval Augmented Generation) System

Build a RAG system that combines search with LLM generation.
Focus: Understanding how to ground LLM responses in your own data.

Install: poetry add numpy
Run: poetry run python 09-rag-langchain/examples/01_rag_basics.py
"""

import math
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ============================================================================
# 1. What is RAG?
# ============================================================================

def demo_rag_intuition():
    """
    RAG: Retrieval Augmented Generation
    
    INTUITION - The Open-Book Exam Analogy:
    
    Closed-book exam (Regular LLM):
    - Student relies only on memorized knowledge
    - May hallucinate answers if unsure
    - Knowledge cutoff date (outdated info)
    - Can't access private/new information
    
    Open-book exam (RAG):
    - Student looks up relevant info in textbook
    - Cites sources, less hallucination
    - Always has latest information
    - Can access your private documents!
    
    HOW RAG WORKS:
    
    Step 1: Index Phase (Do Once)
    ┌─────────────────┐
    │ Your Documents  │
    │ - PDFs          │
    │ - Web pages     │
    │ - Databases     │
    └────────┬────────┘
             │
             ↓
    Split into chunks (each ~500 words)
             │
             ↓
    Convert to embeddings
             │
             ↓
    ┌────────┴─────────┐
    │ Vector Database  │
    │ (Searchable)     │
    └──────────────────┘
    
    Step 2: Query Phase (Every Request)
    ┌─────────────────┐
    │ User Question   │
    │ "What is our    │
    │ refund policy?" │
    └────────┬────────┘
             │
             ↓
    Convert question to embedding
             │
             ↓
    Search vector DB for similar chunks
             │
             ↓
    ┌────────┴────────┐
    │ Top 3-5 Chunks  │
    │ (Most relevant) │
    └────────┬────────┘
             │
             ↓
    Combine: Question + Retrieved Context
             │
             ↓
    ┌────────┴────────┐
    │ Prompt to LLM:  │
    │                 │
    │ Context: [...]  │
    │ Question: [...]?│
    │ Answer based on │
    │ the context:    │
    └────────┬────────┘
             │
             ↓
    ┌────────┴────────┐
    │ LLM Response    │
    │ (Grounded in    │
    │  your docs!)    │
    └─────────────────┘
    
    WHY RAG IS POWERFUL:
    
    1. No Training Required
       - Don't retrain LLM on your data (expensive!)
       - Just index documents (cheap!)
    
    2. Always Up-to-Date
       - Update documents → Instantly reflected
       - No model retraining needed
    
    3. Source Attribution
       - LLM cites which documents it used
       - User can verify answers
    
    4. Private Data
       - Keep data local, don't send to LLM training
       - Perfect for enterprises
    
    5. Reduces Hallucination
       - LLM has context, less likely to make things up
       - Can say "not in the provided context"
    
    Real Example:
    
    Without RAG:
    User: "What's our company's vacation policy?"
    LLM: "Most companies offer 10-15 days..." (generic, maybe wrong!)
    
    With RAG:
    User: "What's our company's vacation policy?"
    System:
      1. Searches your HR docs
      2. Finds: "Employees receive 20 days vacation..."
      3. LLM responds with YOUR specific policy ✅
      4. Cites the HR document
    """
    print("=" * 70)
    print("1. What is RAG?")
    print("=" * 70)
    print()
    print("💭 INTUITION: Open-Book vs Closed-Book Exam")
    print()
    print("   ❌ Closed-Book (Regular LLM):")
    print("      Student: 'What's the capital of Atlantis?'")
    print("      Answer: 'Poseidonia' (hallucinated! Atlantis is fictional)")
    print("      Problem: Relies only on memorized knowledge")
    print()
    print("   ✅ Open-Book (RAG):")
    print("      Student: 'What's the capital of Atlantis?'")
    print("      *Checks textbook*")
    print("      Answer: 'Atlantis is a fictional place, no capital exists'")
    print("      Benefit: Looks up facts, accurate answers!")
    print()
    
    print("🔄 RAG WORKFLOW:")
    print()
    print("   Step 1: Index Your Documents (Once)")
    print("   ┌────────────────────┐")
    print("   │ Your Documents     │")
    print("   │ • HR policies      │")
    print("   │ • Product docs     │")
    print("   │ • Support tickets  │")
    print("   └─────────┬──────────┘")
    print("             │")
    print("             ↓ Split into chunks")
    print("             ↓ Convert to embeddings")
    print("             ↓")
    print("   ┌─────────┴──────────┐")
    print("   │ Vector Database    │")
    print("   │ (Searchable index) │")
    print("   └────────────────────┘")
    print()
    
    print("   Step 2: Answer Questions (Every Query)")
    print("   ┌────────────────────┐")
    print("   │ User Question      │")
    print("   │ 'Vacation policy?' │")
    print("   └─────────┬──────────┘")
    print("             │")
    print("             ↓ Convert to embedding")
    print("             ↓ Search vector DB")
    print("             ↓")
    print("   ┌─────────┴──────────┐")
    print("   │ Retrieved Context  │")
    print("   │ (Top 3 chunks)     │")
    print("   └─────────┬──────────┘")
    print("             │")
    print("             ↓ Combine: Context + Question")
    print("             ↓")
    print("   ┌─────────┴──────────┐")
    print("   │ LLM                │")
    print("   │ (Generates answer) │")
    print("   └─────────┬──────────┘")
    print("             │")
    print("             ↓")
    print("   ┌─────────┴──────────┐")
    print("   │ Answer             │")
    print("   │ (Based on docs!)   │")
    print("   └────────────────────┘")
    print()
    
    print("🎯 REAL EXAMPLE:")
    print()
    print("   Company Knowledge Base (3 documents):")
    print()
    print("   Doc 1 (HR Policy):")
    print("   'Employees receive 20 days of vacation per year.'")
    print()
    print("   Doc 2 (Office Info):")
    print("   'The main office is located in San Francisco.'")
    print()
    print("   Doc 3 (Product):")
    print("   'Our app supports iOS, Android, and web platforms.'")
    print()
    
    # Simulate RAG process
    documents = {
        "HR Policy": {
            "text": "Employees receive 20 days of vacation per year.",
            "embedding": [0.9, 0.1, 0.1]  # Simplified 3D embedding
        },
        "Office Info": {
            "text": "The main office is located in San Francisco.",
            "embedding": [0.1, 0.9, 0.1]
        },
        "Product Info": {
            "text": "Our app supports iOS, Android, and web platforms.",
            "embedding": [0.1, 0.1, 0.9]
        }
    }
    
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x ** 2 for x in a))
        mag_b = math.sqrt(sum(y ** 2 for y in b))
        return dot / (mag_a * mag_b)
    
    # User question
    question = "How many vacation days do employees get?"
    question_embedding = [0.9, 0.1, 0.0]  # Similar to HR Policy
    
    print(f"   User Question: '{question}'")
    print(f"   Question embedding: {question_embedding}")
    print()
    
    # Retrieve relevant documents
    print("   Step 1: Retrieve Relevant Documents")
    print()
    
    similarities = []
    for doc_name, doc_data in documents.items():
        sim = cosine_similarity(question_embedding, doc_data["embedding"])
        similarities.append((doc_name, sim, doc_data["text"]))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (doc_name, sim, text) in enumerate(similarities, 1):
        bar = "█" * int(sim * 50)
        print(f"   {rank}. [{sim:.3f}] {bar} {doc_name}")
        if rank == 1:
            print(f"      ✅ Most relevant!")
    print()
    
    # Get top result as context
    top_doc = similarities[0][2]
    
    print("   Step 2: Construct Prompt to LLM")
    print()
    print("   ┌" + "─" * 66 + "┐")
    print(f"   │ Context: {top_doc:<54} │")
    print("   │                                                                  │")
    print(f"   │ Question: {question:<51} │")
    print("   │                                                                  │")
    print("   │ Answer based on the context above:                              │")
    print("   └" + "─" * 66 + "┘")
    print()
    
    print("   Step 3: LLM Generates Answer")
    print()
    print("   ┌" + "─" * 66 + "┐")
    print("   │ Based on the HR Policy, employees receive 20 days of vacation   │")
    print("   │ per year.                                                        │")
    print("   │                                                                  │")
    print("   │ Source: HR Policy document                                      │")
    print("   └" + "─" * 66 + "┘")
    print()
    
    print("💡 KEY BENEFITS:")
    print()
    print("   1️⃣  Accurate: Answer comes from YOUR documents")
    print("   2️⃣  Verifiable: Source is cited")
    print("   3️⃣  Up-to-date: Change doc → Answer changes instantly")
    print("   4️⃣  Private: Your data stays in your control")
    print("   5️⃣  No training: Just index and search!")
    print()
    
    print("🆚 COMPARISON:")
    print()
    print("   Traditional LLM:")
    print("   ❌ May hallucinate policy details")
    print("   ❌ Knowledge cutoff (outdated info)")
    print("   ❌ Can't know your specific policies")
    print("   ❌ No source attribution")
    print()
    print("   RAG System:")
    print("   ✅ Answers from your actual documents")
    print("   ✅ Always current (updates instantly)")
    print("   ✅ Knows your specific information")
    print("   ✅ Cites sources")
    print()
    
    print("🌍 REAL-WORLD USE CASES:")
    print()
    print("   • Customer Support: Answer using help docs")
    print("   • Enterprise Search: Search internal knowledge")
    print("   • Legal: Find relevant case law")
    print("   • Medical: Reference medical literature")
    print("   • E-commerce: Product recommendations with specs")
    print("   • Education: Tutor using textbook content")


# ============================================================================
# 2. Building a Simple RAG System
# ============================================================================

class SimpleRAG:
    """
    Simple RAG system implementation.
    
    Components:
    1. Document store (chunks + embeddings)
    2. Retriever (semantic search)
    3. Generator (LLM simulation)
    """
    
    def __init__(self):
        """Initialize empty RAG system."""
        self.documents = []  # List of {text, embedding, metadata}
        
    def chunk_document(self, text: str, chunk_size: int = 200) -> List[str]:
        """
        Split document into chunks.
        
        Why chunking?
        - LLMs have token limits (e.g., 4096 tokens)
        - Smaller chunks = more precise retrieval
        - Better to retrieve 3 relevant chunks than 1 huge doc
        
        Args:
            text: Document text
            chunk_size: Characters per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text (simplified).
        
        In reality, use:
        - sentence-transformers
        - OpenAI embeddings
        - Cohere embeddings
        
        Here: Simplified for demonstration
        """
        # Simplified: Use word presence as features
        keywords = {
            "vacation": 0,
            "holiday": 0,
            "office": 1,
            "location": 1,
            "app": 2,
            "platform": 2,
            "software": 2,
            "policy": 0,
            "benefit": 0,
            "employee": 0,
        }
        
        text_lower = text.lower()
        
        # Create 3D embedding based on keyword presence
        embedding = [0.0, 0.0, 0.0]
        
        for keyword, dimension in keywords.items():
            if keyword in text_lower:
                embedding[dimension] += 1.0
        
        # Normalize
        magnitude = math.sqrt(sum(x ** 2 for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
            embedding = [0.33, 0.33, 0.34]  # Default
        
        return embedding
    
    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """
        Add document to RAG system.
        
        Args:
            text: Document text
            metadata: Optional metadata (source, date, etc.)
        """
        # Chunk document
        chunks = self.chunk_document(text)
        
        # Create embedding for each chunk
        for i, chunk in enumerate(chunks):
            embedding = self.create_embedding(chunk)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_id"] = i
            
            self.documents.append({
                "text": chunk,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for query.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Create query embedding
        query_embedding = self.create_embedding(query)
        
        # Calculate similarity with all documents
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x ** 2 for x in a))
            mag_b = math.sqrt(sum(y ** 2 for y in b))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)
        
        results = []
        for doc in self.documents:
            similarity = cosine_similarity(query_embedding, doc["embedding"])
            results.append({
                "text": doc["text"],
                "similarity": similarity,
                "metadata": doc["metadata"]
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k]
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate answer using retrieved context (LLM simulation).
        
        In reality, this calls an LLM API:
        - OpenAI GPT-4
        - Anthropic Claude
        - Open-source: Llama, Mistral
        
        Args:
            query: User question
            context_docs: Retrieved documents
            
        Returns:
            Generated answer
        """
        # In reality, construct prompt and call LLM
        # Here: Simple simulation
        
        if not context_docs:
            return "I don't have enough information to answer that question."
        
        # Simulate answer based on top document
        context = context_docs[0]["text"]
        source = context_docs[0]["metadata"].get("source", "Unknown")
        
        # Simple answer generation (in reality, LLM does this)
        answer = f"Based on the retrieved information: {context}\n\nSource: {source}"
        
        return answer
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Full RAG query: retrieve + generate.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
            "num_sources": len(retrieved_docs)
        }


def demo_rag_system():
    """
    Demonstrate RAG system in action.
    """
    print("\n" + "=" * 70)
    print("2. Building a Simple RAG System")
    print("=" * 70)
    print()
    
    print("🏗️  STEP 1: Initialize RAG System")
    print()
    
    rag = SimpleRAG()
    
    print("   ✅ Created SimpleRAG instance")
    print()
    
    print("📄 STEP 2: Add Documents to Knowledge Base")
    print()
    
    documents = [
        {
            "text": """Employee Benefits Policy: All full-time employees receive 20 days 
            of paid vacation per year. Vacation days can be used at any time with 
            manager approval. Unused days do not roll over to the next year. 
            Part-time employees receive vacation days prorated based on hours worked.""",
            "metadata": {"source": "HR Policy", "date": "2024-01-15"}
        },
        {
            "text": """Office Locations: Our main headquarters is located in San Francisco, 
            California. We also have satellite offices in New York City, Austin, Texas, 
            and Seattle, Washington. The San Francisco office is open Monday through 
            Friday from 9 AM to 6 PM.""",
            "metadata": {"source": "Office Guide", "date": "2024-02-01"}
        },
        {
            "text": """Product Platform Support: Our application is available on multiple 
            platforms including iOS, Android, and web browsers. The mobile apps support 
            iOS 14+ and Android 10+. The web version works on Chrome, Firefox, Safari, 
            and Edge. We also offer a desktop application for Windows and macOS.""",
            "metadata": {"source": "Product Documentation", "date": "2024-03-10"}
        }
    ]
    
    for i, doc in enumerate(documents, 1):
        rag.add_document(doc["text"], doc["metadata"])
        print(f"   {i}. Added: {doc['metadata']['source']}")
        # Show chunks
        chunks = rag.chunk_document(doc["text"])
        print(f"      Split into {len(chunks)} chunks")
    
    print()
    print(f"   ✅ Total documents indexed: {len(rag.documents)}")
    print()
    
    print("🔍 STEP 3: Query the RAG System")
    print()
    
    # Test queries
    test_queries = [
        "How many vacation days do employees get?",
        "Where is the main office located?",
        "What platforms does the app support?",
        "What is the company's revenue?",  # Not in docs
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: '{query}'")
        print()
        
        # Retrieve relevant docs
        retrieved = rag.retrieve(query, top_k=2)
        
        print("   Retrieved Documents:")
        for rank, doc in enumerate(retrieved, 1):
            print(f"   {rank}. [Similarity: {doc['similarity']:.3f}]")
            print(f"      Text: {doc['text'][:100]}...")
            print(f"      Source: {doc['metadata']['source']}")
            print()
        
        # Generate answer
        result = rag.query(query, top_k=2)
        
        print("   Generated Answer:")
        print(f"   {result['answer'][:200]}...")
        print()
        print("-" * 70)
        print()
    
    print("💡 OBSERVATIONS:")
    print()
    print("   Query 1 (vacation days):")
    print("   ✅ Retrieved HR Policy document")
    print("   ✅ Answer based on company policy")
    print()
    print("   Query 2 (office location):")
    print("   ✅ Retrieved Office Guide")
    print("   ✅ Specific location provided")
    print()
    print("   Query 3 (app platforms):")
    print("   ✅ Retrieved Product Documentation")
    print("   ✅ Listed all platforms")
    print()
    print("   Query 4 (revenue - not in docs):")
    print("   ⚠️  No relevant documents found")
    print("   ✅ System correctly says insufficient information")
    print("      (Better than hallucinating!)")


# ============================================================================
# 3. Advanced RAG Techniques
# ============================================================================

def demo_advanced_rag():
    """
    Advanced RAG techniques for production systems.
    """
    print("\n" + "=" * 70)
    print("3. Advanced RAG Techniques")
    print("=" * 70)
    print()
    
    print("🚀 TECHNIQUE 1: Hybrid Search")
    print()
    print("   Problem: Semantic search alone misses exact keyword matches")
    print()
    print("   Example:")
    print("   Query: 'What is GPT-4?'")
    print("   Semantic only: Might retrieve docs about 'language models'")
    print("   Missing: Docs that specifically mention 'GPT-4'")
    print()
    print("   Solution: Combine semantic + keyword search")
    print()
    print("   Score = 0.7 × semantic_score + 0.3 × keyword_score")
    print()
    print("   Benefits:")
    print("   ✅ Finds conceptually similar docs (semantic)")
    print("   ✅ Finds exact matches (keyword)")
    print("   ✅ Best of both worlds!")
    print()
    
    print("🚀 TECHNIQUE 2: Re-ranking")
    print()
    print("   Problem: Initial retrieval returns 100 docs, but only top 5 matter")
    print()
    print("   Solution: Two-stage retrieval")
    print()
    print("   Stage 1: Fast retrieval")
    print("   • Retrieve top 100 candidates (fast, approximate)")
    print("   • Use simple embeddings")
    print()
    print("   Stage 2: Re-rank top candidates")
    print("   • Use better model on just these 100 docs")
    print("   • More expensive but only on small set")
    print("   • Return best 5")
    print()
    print("   Example: Cohere Re-rank API")
    print("   • Stage 1: 10ms for 1M docs → 100 candidates")
    print("   • Stage 2: 100ms to re-rank 100 docs → 5 best")
    print("   • Total: 110ms for highly accurate results!")
    print()
    
    print("🚀 TECHNIQUE 3: Query Expansion")
    print()
    print("   Problem: User query is vague or uses different terminology")
    print()
    print("   Solution: Generate multiple query variations")
    print()
    print("   Original query: 'car issues'")
    print()
    print("   Expanded queries:")
    print("   • 'automobile problems'")
    print("   • 'vehicle troubleshooting'")
    print("   • 'car repair and maintenance'")
    print("   • 'automotive defects'")
    print()
    print("   Search with all variations, combine results")
    print("   → More comprehensive retrieval!")
    print()
    
    print("🚀 TECHNIQUE 4: Contextual Compression")
    print()
    print("   Problem: Retrieved chunks contain irrelevant information")
    print()
    print("   Retrieved chunk (500 words):")
    print("   'Company history... [irrelevant]... vacation policy is 20 days...")
    print("   [more irrelevant]...'")
    print()
    print("   Solution: Extract only relevant sentences")
    print()
    print("   Compressed chunk (50 words):")
    print("   'Vacation policy is 20 days per year. Employees can use days")
    print("   anytime with approval.'")
    print()
    print("   Benefits:")
    print("   ✅ Less noise in LLM input")
    print("   ✅ Better focus on relevant info")
    print("   ✅ Cheaper (fewer tokens)")
    print()
    
    print("🚀 TECHNIQUE 5: Multi-Query Retrieval")
    print()
    print("   Problem: Single query might not capture user intent")
    print()
    print("   Solution: Generate multiple queries from user question")
    print()
    print("   User: 'How do I reset my password?'")
    print()
    print("   Generated queries:")
    print("   1. 'password reset instructions'")
    print("   2. 'forgot password recovery'")
    print("   3. 'change account password'")
    print("   4. 'reset login credentials'")
    print()
    print("   Retrieve docs for each query, combine results")
    print("   → More robust retrieval!")
    print()
    
    print("🚀 TECHNIQUE 6: Hierarchical Retrieval")
    print()
    print("   Problem: Long documents, hard to find specific info")
    print()
    print("   Solution: Two-level hierarchy")
    print()
    print("   Level 1: Document summaries")
    print("   • Retrieve: 'Which documents are relevant?'")
    print("   • Fast: Compare query to 1000 document summaries")
    print()
    print("   Level 2: Chunk-level search")
    print("   • In selected documents, find specific chunks")
    print("   • Detailed: Compare query to chunks in 10 relevant docs")
    print()
    print("   Example:")
    print("   Query: 'Python async/await syntax'")
    print("   Level 1: Find 'Python Documentation' (relevant doc)")
    print("   Level 2: Find 'Async Programming' chapter chunks")
    print()
    
    print("💡 PRODUCTION CHECKLIST:")
    print()
    print("   ✅ Hybrid search (semantic + keyword)")
    print("   ✅ Re-ranking (two-stage retrieval)")
    print("   ✅ Metadata filtering (date, source, tags)")
    print("   ✅ Query expansion (synonyms, variations)")
    print("   ✅ Contextual compression (remove noise)")
    print("   ✅ Source citation (link to original docs)")
    print("   ✅ Feedback loop (learn from user ratings)")
    print("   ✅ Caching (cache frequent queries)")


# ============================================================================
# 4. Real-World RAG Architecture
# ============================================================================

def demo_production_rag():
    """
    Production-ready RAG system architecture.
    """
    print("\n" + "=" * 70)
    print("4. Production RAG Architecture")
    print("=" * 70)
    print()
    
    print("🏗️  FULL RAG SYSTEM ARCHITECTURE:")
    print()
    print("   ┌─────────────────────────────────────────────────────┐")
    print("   │                   DATA SOURCES                       │")
    print("   │  • PDFs  • Websites  • APIs  • Databases            │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │              DATA INGESTION PIPELINE                 │")
    print("   │  1. Extract text (PDFs, HTML, etc.)                 │")
    print("   │  2. Clean & preprocess                              │")
    print("   │  3. Chunk documents (overlap for context)           │")
    print("   │  4. Generate embeddings (batch process)             │")
    print("   │  5. Store in vector DB                              │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │                 VECTOR DATABASE                      │")
    print("   │  • Pinecone / Weaviate / Qdrant                     │")
    print("   │  • Millions of embeddings indexed                   │")
    print("   │  • Metadata filtering (date, source, etc.)          │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │                  USER QUERY                          │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │              RETRIEVAL LAYER                         │")
    print("   │  1. Convert query to embedding                      │")
    print("   │  2. Search vector DB (hybrid search)                │")
    print("   │  3. Re-rank results                                 │")
    print("   │  4. Return top-k chunks                             │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │              GENERATION LAYER                        │")
    print("   │  1. Construct prompt (context + query)              │")
    print("   │  2. Call LLM API (GPT-4, Claude, etc.)             │")
    print("   │  3. Parse response                                  │")
    print("   │  4. Add source citations                            │")
    print("   └──────────────────┬──────────────────────────────────┘")
    print("                      │")
    print("                      ↓")
    print("   ┌──────────────────┴──────────────────────────────────┐")
    print("   │                RESPONSE TO USER                      │")
    print("   │  • Answer with citations                            │")
    print("   │  • Links to source documents                        │")
    print("   │  • Confidence score                                 │")
    print("   └─────────────────────────────────────────────────────┘")
    print()
    
    print("🔧 KEY COMPONENTS:")
    print()
    print("   1️⃣  Document Loader:")
    print("      • LangChain: PDFLoader, WebBaseLoader, etc.")
    print("      • Unstructured: Parse complex docs")
    print("      • Custom: API integrations")
    print()
    print("   2️⃣  Text Splitter:")
    print("      • RecursiveCharacterTextSplitter (smart chunking)")
    print("      • Chunk size: 500-1000 tokens")
    print("      • Overlap: 50-100 tokens (preserve context)")
    print()
    print("   3️⃣  Embedding Model:")
    print("      • OpenAI: text-embedding-3-small (fast, cheap)")
    print("      • Sentence Transformers: all-MiniLM-L6-v2 (local)")
    print("      • Cohere: embed-multilingual-v3.0 (best quality)")
    print()
    print("   4️⃣  Vector Database:")
    print("      • Pinecone: Managed, easy scaling")
    print("      • Weaviate: Open-source, feature-rich")
    print("      • Qdrant: Fast, production-ready")
    print("      • Chroma: Simple, embedded")
    print()
    print("   5️⃣  LLM:")
    print("      • OpenAI GPT-4: Best quality")
    print("      • Anthropic Claude: Long context (100k tokens)")
    print("      • Open-source: Llama 2, Mistral (self-hosted)")
    print()
    
    print("💰 COST OPTIMIZATION:")
    print()
    print("   Typical Costs per 1000 queries:")
    print()
    print("   Embeddings:")
    print("   • OpenAI: $0.13 (text-embedding-3-small)")
    print("   • Cohere: $0.10 (embed-english-v3.0)")
    print("   • Local: $0 (sentence-transformers)")
    print()
    print("   Vector DB:")
    print("   • Pinecone: ~$70/month (starter)")
    print("   • Self-hosted: Server costs only")
    print()
    print("   LLM:")
    print("   • GPT-4: $0.01-0.03 per query")
    print("   • GPT-3.5: $0.001-0.002 per query")
    print("   • Local: GPU costs only")
    print()
    print("   Total: ~$0.01-0.05 per query")
    print("   (vs $0.10+ for fine-tuning approach)")
    print()
    
    print("⚡ PERFORMANCE OPTIMIZATION:")
    print()
    print("   Latency Breakdown (typical):")
    print("   • Embedding query: 50ms")
    print("   • Vector search: 10-50ms")
    print("   • LLM generation: 500-2000ms ← Bottleneck!")
    print("   • Total: ~600-2100ms")
    print()
    print("   Optimizations:")
    print("   1. Cache frequent queries (90% hit rate → 10x faster)")
    print("   2. Streaming LLM responses (perceived speed ↑)")
    print("   3. Async processing (handle multiple queries)")
    print("   4. Pre-compute embeddings (don't regenerate)")
    print("   5. Use faster LLM for simple queries")
    print()
    
    print("📊 MONITORING & EVALUATION:")
    print()
    print("   Track These Metrics:")
    print()
    print("   Retrieval Quality:")
    print("   • Precision@k: % of retrieved docs that are relevant")
    print("   • Recall@k: % of relevant docs that are retrieved")
    print("   • MRR (Mean Reciprocal Rank): Position of first relevant doc")
    print()
    print("   Generation Quality:")
    print("   • Answer correctness (human eval)")
    print("   • Faithfulness (answer based on context?)")
    print("   • Hallucination rate (made up facts?)")
    print()
    print("   User Satisfaction:")
    print("   • Thumbs up/down on answers")
    print("   • Click-through rate on sources")
    print("   • Time to find answer")
    print()
    
    print("🔒 SECURITY & PRIVACY:")
    print()
    print("   ✅ Data access control (user permissions)")
    print("   ✅ PII detection & redaction")
    print("   ✅ Audit logs (who accessed what)")
    print("   ✅ Encryption at rest and in transit")
    print("   ✅ Rate limiting (prevent abuse)")
    print("   ✅ Content filtering (block inappropriate)")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🔍 RAG (Retrieval Augmented Generation)\n")
    print("Learn how to ground LLM responses in your own data!")
    print()
    
    demo_rag_intuition()
    demo_rag_system()
    demo_advanced_rag()
    demo_production_rag()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. RAG = Retrieval + Generation
   - Retrieve relevant context from your documents
   - Generate answer using LLM + context
   - Reduces hallucination, always up-to-date

2. Core Components:
   - Document ingestion: Load, chunk, embed
   - Vector database: Store and search embeddings
   - Retrieval: Find relevant chunks
   - Generation: LLM creates answer

3. Advanced Techniques:
   - Hybrid search (semantic + keyword)
   - Re-ranking (two-stage retrieval)
   - Query expansion (multiple variations)
   - Contextual compression (remove noise)

4. Production Considerations:
   - Vector DB: Pinecone, Weaviate, Qdrant
   - LLM: GPT-4, Claude, open-source
   - Cost: ~$0.01-0.05 per query
   - Latency: ~600-2100ms total

Real Code Example (LangChain):
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load documents
loader = PyPDFLoader("company_docs.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(chunks, embeddings)

# 4. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 5. Query
result = qa_chain("What is our vacation policy?")
print(result['result'])
print("Sources:", result['source_documents'])
```

Popular Frameworks:
- LangChain: Most features, active community
- LlamaIndex: Specialized for RAG, great docs
- Haystack: Production-ready, enterprise focus

Next Steps:
- Try LangChain/LlamaIndex tutorials
- Build RAG for your own documents
- Experiment with different chunking strategies
- Test various embedding models
""")


if __name__ == "__main__":
    main()
