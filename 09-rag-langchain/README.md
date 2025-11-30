# Module 9: RAG and LangChain

**Goal**: Build Retrieval Augmented Generation (RAG) systems for Q&A over your own data.

**Why RAG?** LLMs don't know your proprietary data. RAG connects LLMs to your documents.

## 📚 What You'll Learn

### RAG Fundamentals
- What is RAG? (Retrieval + Generation)
- Why use RAG? (Fresh data, reduced hallucinations)
- RAG architecture (document → chunks → embeddings → retrieval → generation)
- Document loaders (PDF, Markdown, web, APIs)
- Text chunking strategies
- Retrieval methods

### LangChain Framework
- Core concepts (chains, agents, memory)
- Document loaders and splitters
- Vector stores integration
- Prompt templates
- Output parsers
- Memory (conversation history)
- Agents and tools

### Vector Store Operations
- Ingestion pipeline (load → split → embed → store)
- Retrieval strategies (similarity, MMR, threshold)
- Metadata filtering
- Hybrid search (keyword + semantic)
- Re-ranking results

### Advanced RAG
- Query expansion
- Hypothetical document embeddings (HyDE)
- Multi-query retrieval
- Parent document retrieval
- Ensemble retrievers
- RAG evaluation (faithfulness, relevancy)

### Production Considerations
- Caching strategies
- Error handling
- Monitoring and logging
- Cost optimization
- Scaling vector databases
- A/B testing RAG pipelines

## 🎯 Real-World Applications

- **Documentation Q&A**: Ask questions about docs
- **Customer Support**: Answer from knowledge base
- **Research Assistant**: Query research papers
- **Code Q&A**: Ask about codebases
- **Legal/Compliance**: Query regulations
- **Internal Knowledge**: Company wiki Q&A

## 📂 Module Structure

```
09-rag-langchain/
├── README.md (you are here)
├── fundamentals/
│   ├── 01_langchain_basics.py       # First chain
│   ├── 02_document_loaders.py       # Load various formats
│   ├── 03_text_splitting.py         # Chunk strategies
│   ├── 04_embeddings.py             # Generate embeddings
│   └── 05_vector_stores.py          # Store & retrieve
├── rag_basics/
│   ├── 01_simple_rag.py             # Basic RAG pipeline
│   ├── 02_conversational_rag.py     # With memory
│   ├── 03_metadata_filtering.py     # Filter by metadata
│   └── 04_hybrid_search.py          # Keyword + semantic
├── advanced_rag/
│   ├── 01_query_expansion.py        # Multiple queries
│   ├── 02_hyde.py                   # Hypothetical docs
│   ├── 03_reranking.py              # Improve results
│   ├── 04_parent_retriever.py       # Hierarchical chunks
│   └── 05_agent_rag.py              # Agents for RAG
├── evaluation/
│   ├── 01_faithfulness.py           # Answer accuracy
│   ├── 02_relevancy.py              # Retrieval quality
│   ├── 03_groundedness.py           # Source attribution
│   └── 04_ragas_framework.py        # Full evaluation
└── projects/
    ├── documentation_qa/            # Ask about docs
    ├── research_assistant/          # Query papers
    └── customer_support/            # Knowledge base Q&A
```

## 💡 RAG Architecture

```
User Query
   ↓
Query Embedding
   ↓
Vector Similarity Search
   ↓
Retrieve Top K Documents
   ↓
Construct Prompt (query + retrieved docs)
   ↓
LLM Generation
   ↓
Answer + Sources
```

## 🔧 RAG Pipeline Steps

**1. Ingestion (offline)**
```python
# Load documents
docs = load_documents("data/")

# Split into chunks
chunks = text_splitter.split_documents(docs)

# Generate embeddings
embeddings = embed_documents(chunks)

# Store in vector DB
vector_store.add_documents(chunks)
```

**2. Retrieval (online)**
```python
# User asks question
query = "What is RAG?"

# Retrieve relevant chunks
docs = vector_store.similarity_search(query, k=4)

# Generate answer
answer = llm.generate(query, context=docs)
```

## 📊 Chunking Strategies

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| Fixed Size | General purpose | Simple | May split sentences |
| Sentence | Semantic units | Preserves meaning | Variable size |
| Paragraph | Natural boundaries | Contextual | May be too large |
| Recursive | Hierarchical | Smart splitting | More complex |
| Semantic | Similar content | Coherent chunks | Slow |

**Recommended**: Start with Recursive (500 tokens, 50 overlap)

## 🎓 LangChain Components

**Chains**: Sequence of operations
```python
chain = prompt | llm | parser
result = chain.invoke({"question": "What is RAG?"})
```

**Agents**: Autonomous decision-makers
```python
agent = create_agent(llm, tools=[search, calculator])
agent.run("What's 20% of the GDP of France?")
```

**Memory**: Conversation history
```python
memory = ConversationBufferMemory()
chain = ConversationalRetrievalChain(llm, retriever, memory)
```

## 🚀 Quick Start

```bash
# Install dependencies
poetry add langchain langchain-openai chromadb pypdf

# Set API key
export OPENAI_API_KEY="your-key-here"

# Simple RAG
poetry run python 09-rag-langchain/rag_basics/01_simple_rag.py

# With conversation memory
poetry run python 09-rag-langchain/rag_basics/02_conversational_rag.py

# Full project
poetry run python 09-rag-langchain/projects/documentation_qa/app.py
```

## 💰 Cost Optimization

**1. Cache Embeddings**
```python
# Generate once, use many times
embeddings = cache.get_or_create(document)
```

**2. Smaller Chunks**
```python
# Less context to LLM = lower cost
# But maintain semantic coherence
```

**3. Use Cheaper LLM for Retrieval**
```python
# GPT-3.5 for initial filtering
# GPT-4 for final generation
```

**4. Implement Caching**
```python
# Cache common questions
if question in cache:
    return cache[question]
```

## 🎯 RAG Best Practices

**1. Chunk Size Matters**
- Too small: Loss of context
- Too large: Irrelevant info, high cost
- Sweet spot: 500-1000 tokens

**2. Include Overlap**
- 10-20% overlap between chunks
- Prevents cutting sentences/concepts

**3. Use Metadata**
- Source, date, author, section
- Filter retrieval by metadata

**4. Re-rank Results**
- Initial retrieval: Top 20
- Re-rank to Top 5
- Send only best to LLM

**5. Cite Sources**
- Always return source documents
- Users can verify answers

**6. Handle "I don't know"**
- If no relevant docs, admit it
- Don't hallucinate

## ⚠️ Common Pitfalls

1. **Poor chunking** → Irrelevant context
2. **No metadata filtering** → Wrong documents
3. **Not citing sources** → Can't verify
4. **Too many chunks to LLM** → High cost, slow
5. **No conversation memory** → Loses context
6. **Not handling edge cases** → Bad UX

## 🎯 Expected Outcomes

After this module:
- ✅ Build RAG systems with LangChain
- ✅ Load and process documents
- ✅ Implement effective chunking
- ✅ Use vector stores efficiently
- ✅ Handle conversations with memory
- ✅ Evaluate RAG performance
- ✅ Deploy production RAG apps

---

**Next**: Module 10 - Full-Stack AI Applications 🚀
