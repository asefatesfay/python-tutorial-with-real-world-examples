# RAG Chatbot - Production-Ready Implementation 🤖

## 🎯 Overview

A **complete, production-ready** Retrieval-Augmented Generation (RAG) chatbot that demonstrates professional ML engineering practices. This isn't a tutorial example—it's a real application you can deploy to production today.

## ✨ Features

### Core Functionality
- ✅ **Document Processing**: Upload and process documents into vector embeddings
- ✅ **Intelligent Search**: Semantic search using vector similarity
- ✅ **Context-Aware Responses**: Generate answers using retrieved context
- ✅ **Streaming Responses**: Real-time token streaming for better UX
- ✅ **Multi-Format Support**: PDF, TXT, MD, DOCX document support

### Production Features
- ✅ **Async Operations**: Non-blocking API for high concurrency
- ✅ **Error Handling**: Comprehensive error handling with retries
- ✅ **Input Validation**: Pydantic models for request/response validation
- ✅ **Rate Limiting**: Prevent abuse and control costs
- ✅ **Health Checks**: Kubernetes-ready liveness/readiness probes
- ✅ **Observability**: Structured logging, metrics, and tracing
- ✅ **Testing**: 90%+ coverage with unit, integration, and E2E tests
- ✅ **Documentation**: OpenAPI/Swagger auto-generated docs

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Backend             │
│  ┌─────────────────────────────┐   │
│  │    API Endpoints            │   │
│  │  - POST /chat               │   │
│  │  - POST /chat/stream        │   │
│  │  - POST /documents          │   │
│  │  - GET /health              │   │
│  └─────────────┬───────────────┘   │
│                │                     │
│  ┌─────────────▼───────────────┐   │
│  │    RAG Service              │   │
│  │  - Document Processing      │   │
│  │  - Embedding Generation     │   │
│  │  - Context Retrieval        │   │
│  │  - Response Generation      │   │
│  └─────────────┬───────────────┘   │
│                │                     │
└────────────────┼─────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌──────────────┐
│   ChromaDB    │  │  OpenAI API  │
│ Vector Store  │  │   GPT-4 +    │
│               │  │  Embeddings  │
└───────────────┘  └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Poetry (package manager)
- OpenAI API key
- Docker (optional, for containerization)

### Local Development

```bash
# 1. Navigate to project
cd 12-production-projects/rag_chatbot_complete/

# 2. Install dependencies
poetry install

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the application
poetry run uvicorn app.main:app --reload

# 5. Access the API
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods
kubectl get services

# Access via service
kubectl port-forward service/rag-chatbot 8000:80
```

## 📦 Project Structure

```
rag_chatbot_complete/
├── app/                          # Application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration management
│   ├── models.py                 # Pydantic request/response models
│   ├── rag_service.py            # RAG pipeline implementation
│   ├── vector_store.py           # ChromaDB vector store wrapper
│   ├── document_processor.py     # Document processing utilities
│   ├── logging_config.py         # Structured logging setup
│   └── metrics.py                # Prometheus metrics
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Shared pytest fixtures
│   ├── unit/                     # Unit tests
│   │   ├── test_models.py
│   │   ├── test_rag_service.py
│   │   ├── test_vector_store.py
│   │   └── test_document_processor.py
│   ├── integration/              # Integration tests
│   │   ├── test_api.py
│   │   └── test_rag_pipeline.py
│   └── e2e/                      # End-to-end tests
│       └── test_chat_flow.py
├── docker/                       # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/                          # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── hpa.yaml
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── API.md
│   └── TROUBLESHOOTING.md
├── .env.example                  # Example environment variables
├── pyproject.toml                # Poetry dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...                    # Your OpenAI API key
OPENAI_MODEL=gpt-4-turbo-preview         # Model to use for generation
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Model for embeddings

# Application Configuration
APP_NAME=RAG Chatbot
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma   # Where to store vector DB
CHROMA_COLLECTION_NAME=documents         # Collection name

# RAG Configuration
CHUNK_SIZE=1000                          # Document chunk size
CHUNK_OVERLAP=200                        # Overlap between chunks
TOP_K=3                                  # Number of chunks to retrieve
MAX_TOKENS=1000                          # Max response tokens
TEMPERATURE=0.7                          # Response temperature

# API Configuration
MAX_REQUEST_SIZE=10485760                # 10MB max request size
RATE_LIMIT_PER_MINUTE=60                 # Rate limit
CORS_ORIGINS=["http://localhost:3000"]   # Allowed CORS origins

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 📡 API Endpoints

### Chat (Non-Streaming)

```bash
POST /chat
Content-Type: application/json

{
  "query": "What are the key features of this product?",
  "conversation_history": []  # Optional
}

Response:
{
  "response": "Based on the documentation...",
  "sources": [
    {
      "content": "...",
      "metadata": {"source": "doc1.pdf", "page": 5}
    }
  ],
  "model": "gpt-4-turbo-preview",
  "tokens_used": 450
}
```

### Chat (Streaming)

```bash
POST /chat/stream
Content-Type: application/json

{
  "query": "Explain the architecture",
  "conversation_history": []
}

Response: Server-Sent Events (SSE)
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " architecture"}
data: {"type": "sources", "sources": [...]}
data: {"type": "done"}
```

### Upload Document

```bash
POST /documents
Content-Type: multipart/form-data

file: document.pdf

Response:
{
  "document_id": "doc_abc123",
  "filename": "document.pdf",
  "chunks": 42,
  "status": "processed"
}
```

### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "vector_store": "healthy",
    "openai": "healthy"
  }
}
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test types
poetry run pytest tests/unit/           # Unit tests only
poetry run pytest tests/integration/    # Integration tests only
poetry run pytest tests/e2e/            # E2E tests only

# Run with markers
poetry run pytest -m "not slow"         # Skip slow tests
poetry run pytest -m integration        # Integration tests only
```

### Test Coverage

Current coverage: **92%**

- Unit tests: 150+ tests
- Integration tests: 50+ tests
- E2E tests: 20+ tests

## 📊 Monitoring

### Logs

Structured JSON logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "message": "Chat request processed",
  "correlation_id": "abc-123-def",
  "query_length": 45,
  "response_time_ms": 1234,
  "tokens_used": 450,
  "model": "gpt-4-turbo-preview"
}
```

### Metrics (Prometheus)

Available metrics:
- `rag_requests_total`: Total requests
- `rag_request_duration_seconds`: Request latency
- `rag_errors_total`: Total errors
- `rag_tokens_used_total`: Tokens consumed
- `rag_vector_search_duration_seconds`: Vector search latency
- `rag_openai_call_duration_seconds`: OpenAI API latency

Access metrics at: `http://localhost:9090/metrics`

### Health Checks

- **Liveness**: `/health/live` - Is the app running?
- **Readiness**: `/health/ready` - Is the app ready to serve traffic?

## 🔒 Security

### API Key Management
- ✅ Environment variables (never hardcoded)
- ✅ Secret management in Kubernetes
- ✅ Rotation support

### Input Validation
- ✅ Pydantic models for all inputs
- ✅ File type validation
- ✅ Size limits
- ✅ Sanitization

### Rate Limiting
- ✅ Per-IP rate limiting
- ✅ Configurable limits
- ✅ Cost control

### CORS
- ✅ Configurable origins
- ✅ Secure defaults
- ✅ Production-ready

## 🚢 Deployment

### Docker Deployment

```bash
# Build
docker build -t rag-chatbot:latest -f docker/Dockerfile .

# Run
docker run -p 8000:8000 --env-file .env rag-chatbot:latest
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace rag-chatbot

# Create secret with API key
kubectl create secret generic openai-secret \
  --from-literal=api-key=sk-... \
  -n rag-chatbot

# Apply manifests
kubectl apply -f k8s/ -n rag-chatbot

# Scale
kubectl scale deployment rag-chatbot --replicas=3 -n rag-chatbot

# Check status
kubectl get all -n rag-chatbot
```

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Configure proper `LOG_LEVEL` (INFO or WARNING)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure rate limiting
- [ ] Set up alerts
- [ ] Enable HTTPS/TLS
- [ ] Configure auto-scaling (HPA)
- [ ] Set up backup for vector store
- [ ] Configure secrets management
- [ ] Set up CI/CD pipeline

## 💰 Cost Optimization

### OpenAI Costs

**Estimated costs** (using gpt-4-turbo-preview):
- Input: $10 per 1M tokens
- Output: $30 per 1M tokens
- Embeddings: $0.13 per 1M tokens

**Example**: 1,000 chats/day
- Average input: 500 tokens (context + query)
- Average output: 200 tokens (response)
- Daily cost: ~$10
- Monthly cost: ~$300

### Cost Reduction Strategies

1. **Use Smaller Models**: Switch to GPT-3.5-Turbo (10x cheaper)
2. **Optimize Context**: Reduce `TOP_K` to retrieve fewer chunks
3. **Cache Results**: Cache common queries
4. **Batch Processing**: Process multiple queries together
5. **Monitor Usage**: Set up alerts for unusual usage

## 🎓 Learning Objectives

### What You'll Learn

1. **RAG Systems**
   - How to build production RAG pipelines
   - Document chunking strategies
   - Vector search optimization
   - Context management

2. **FastAPI Best Practices**
   - Async endpoints
   - Dependency injection
   - Background tasks
   - Streaming responses

3. **Vector Databases**
   - ChromaDB integration
   - Embedding generation
   - Similarity search
   - Data persistence

4. **Production Engineering**
   - Error handling
   - Monitoring
   - Testing
   - Deployment

5. **LLM Integration**
   - OpenAI API usage
   - Streaming
   - Token management
   - Cost optimization

## 🐛 Troubleshooting

### Common Issues

**1. OpenAI API Error: Rate Limit**
```
Solution: Implement exponential backoff (already included)
```

**2. ChromaDB Connection Error**
```
Solution: Ensure CHROMA_PERSIST_DIRECTORY is writable
```

**3. Out of Memory**
```
Solution: Reduce CHUNK_SIZE or process fewer documents at once
```

**4. Slow Response Times**
```
Solution: Reduce TOP_K, use smaller model, optimize chunking
```

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

## 📚 Additional Resources

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Documentation](docs/API.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

This is a learning resource! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share your adaptations

## 📝 License

MIT License - Use freely for learning and production!

## 🎉 Next Steps

1. **Run Locally**: Get it working on your machine
2. **Add Your Data**: Upload your documents
3. **Customize**: Adapt for your use case
4. **Deploy**: Push to production
5. **Monitor**: Track performance and costs
6. **Iterate**: Improve based on real usage

**You now have a production-ready RAG chatbot!** 🚀

---

**Questions?** Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) or review the code comments for detailed explanations.
