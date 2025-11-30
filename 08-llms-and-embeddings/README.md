# Module 8: LLMs and Embeddings

**Goal**: Work with Large Language Models and vector embeddings for modern AI applications.

**Focus**: Practical usage of OpenAI, Anthropic, and open-source LLMs.

## 📚 What You'll Learn

### LLM Fundamentals
- How LLMs work (transformers, attention)
- Prompting techniques (zero-shot, few-shot, CoT)
- Temperature, top-p, and other parameters
- Token management and costs
- Context windows and limitations

### Working with APIs
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic API (Claude)
- Open-source models (Llama, Mistral)
- Streaming responses
- Function calling
- Error handling and retries

### Embeddings
- What are embeddings? (vector representations)
- Text embeddings (OpenAI, Sentence Transformers)
- Similarity search (cosine similarity)
- Semantic search
- Clustering documents
- Recommendation systems

### Vector Databases
- ChromaDB (local, easy)
- Pinecone (cloud, scalable)
- Weaviate (hybrid search)
- FAISS (Facebook's similarity search)
- Storing and querying embeddings

### Advanced Techniques
- Prompt engineering
- Prompt templates
- Output parsing
- Chaining LLM calls
- Agents and tools
- Fine-tuning (basics)

## 🎯 Real-World Applications

- **Semantic Search**: Find similar documents
- **Question Answering**: Over your own data
- **Chatbots**: Customer support, assistants
- **Content Generation**: Articles, emails, code
- **Summarization**: Long documents → summaries
- **Classification**: Zero-shot text classification
- **Recommendation**: Content/product recommendations

## 📂 Module Structure

```
08-llms-and-embeddings/
├── README.md (you are here)
├── fundamentals/
│   ├── 01_openai_basics.py          # First API call
│   ├── 02_prompt_engineering.py     # Effective prompts
│   ├── 03_function_calling.py       # Tool usage
│   ├── 04_streaming.py              # Real-time responses
│   └── 05_error_handling.py         # Robust API calls
├── embeddings/
│   ├── 01_text_embeddings.py        # Generate embeddings
│   ├── 02_similarity_search.py      # Find similar text
│   ├── 03_clustering.py             # Group documents
│   └── 04_semantic_search.py        # Better than keyword search
├── vector_databases/
│   ├── 01_chromadb_basics.py        # Local vector DB
│   ├── 02_pinecone_basics.py        # Cloud vector DB
│   ├── 03_qdrant_basics.py          # Alternative
│   └── 04_comparison.py             # When to use which
├── advanced/
│   ├── 01_prompt_templates.py       # Reusable prompts
│   ├── 02_output_parsing.py         # Structured outputs
│   ├── 03_chaining_calls.py         # Multi-step reasoning
│   └── 04_agents.py                 # Autonomous agents
└── projects/
    ├── semantic_search_engine/      # Search your docs
    ├── chatbot/                     # Q&A over docs
    └── content_generator/           # Automated writing
```

## 💡 Key Concepts

### Embeddings Intuition
```
Text → Vector (list of numbers)

"cat" → [0.2, 0.8, 0.1, ...]
"dog" → [0.3, 0.7, 0.1, ...]  ← Similar to "cat"
"car" → [0.8, 0.1, 0.9, ...]  ← Different from "cat"

Similar meaning → Similar vectors
```

### Similarity Search
```
1. Convert all documents to embeddings
2. Store in vector database
3. Convert query to embedding
4. Find nearest neighbors
5. Return most similar documents
```

### LLM vs Traditional ML

| Task | Traditional ML | LLM |
|------|---------------|-----|
| Training Data | Thousands of examples | Zero/few examples |
| Training Time | Hours/days | Already trained |
| Task Specific | Need to retrain | Just change prompt |
| Cost | Training cost | API call cost |

## 🔧 API Keys Setup

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PINECONE_API_KEY="..."

# Or use .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

## 📊 Cost Considerations

**OpenAI Pricing (as of 2024)**:
- GPT-4: $0.03/1K input tokens, $0.06/1K output tokens
- GPT-3.5 Turbo: $0.0005/1K input tokens, $0.0015/1K output tokens
- Embeddings: $0.0001/1K tokens

**Tips**:
- Use GPT-3.5 for simple tasks
- Cache embeddings (generate once, use many times)
- Set max_tokens to limit costs
- Use streaming for better UX (doesn't save cost)

## 🎓 Prompting Best Practices

**1. Be Specific**
```
❌ "Write about AI"
✅ "Write a 200-word introduction to neural networks for beginners"
```

**2. Provide Context**
```
✅ "You are an expert Python developer. Review this code for bugs:
    [code here]"
```

**3. Use Examples (Few-shot)**
```
✅ "Classify sentiment:
    'I love this!' → Positive
    'This is terrible' → Negative
    'It was okay' → Neutral
    'This is amazing!' → ?"
```

**4. Chain of Thought**
```
✅ "Let's think step by step:
    1. First, identify the problem
    2. Then, list possible solutions
    3. Finally, recommend the best approach"
```

## 🚀 Quick Start

```bash
# Install dependencies
poetry add openai anthropic chromadb sentence-transformers

# Set API key
export OPENAI_API_KEY="your-key-here"

# First API call
poetry run python 08-llms-and-embeddings/fundamentals/01_openai_basics.py

# Semantic search
poetry run python 08-llms-and-embeddings/embeddings/04_semantic_search.py

# Vector database
poetry run python 08-llms-and-embeddings/vector_databases/01_chromadb_basics.py
```

## 🎯 Learning Path

1. **LLM Basics** → Make API calls, understand parameters
2. **Prompt Engineering** → Get better results
3. **Embeddings** → Convert text to vectors
4. **Similarity Search** → Find related content
5. **Vector Databases** → Store and query at scale
6. **Advanced** → Agents, chaining, structured outputs
7. **Projects** → Build real applications

## ⚠️ Common Pitfalls

1. **Not handling rate limits** → Add retries
2. **Ignoring costs** → Monitor token usage
3. **Poor prompt engineering** → Iterate on prompts
4. **Not caching embeddings** → Expensive!
5. **Hallucinations** → Verify important facts

## 🎯 Expected Outcomes

After this module:
- ✅ Call LLM APIs effectively
- ✅ Write effective prompts
- ✅ Generate and use embeddings
- ✅ Build semantic search
- ✅ Use vector databases
- ✅ Create LLM-powered applications
- ✅ Understand costs and limitations

---

**Next**: Module 9 - RAG (Retrieval Augmented Generation) 🚀
