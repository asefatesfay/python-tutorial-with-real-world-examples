# Module 10: Deployment & Production ML/AI

**Goal**: Deploy ML/AI models to production with APIs, monitoring, and scaling.

**Reality Check**: Building models is 20%, production deployment is 80% of the work.

## 📚 What You'll Learn

### API Development
- FastAPI fundamentals
- REST API design for ML
- Request/response validation
- Async endpoints
- WebSocket for streaming
- Authentication & authorization
- Rate limiting

### Model Serving
- Model serialization (pickle, ONNX, TorchScript)
- Model versioning
- A/B testing
- Shadow deployment
- Model registry (MLflow)
- Batch vs real-time inference

### Docker & Containerization
- Dockerfile for ML apps
- Multi-stage builds
- GPU support in Docker
- Docker Compose for services
- Container optimization

### Cloud Deployment
- Deploy to AWS (EC2, Lambda, ECS)
- Deploy to GCP (Cloud Run, Vertex AI)
- Deploy to Azure (App Service, ML)
- Serverless ML (Lambda, Cloud Functions)
- Managed ML platforms (SageMaker, Vertex AI)

### Monitoring & Observability
- Logging (structured logs)
- Metrics (Prometheus, Grafana)
- Alerting (PagerDuty, Slack)
- Model performance monitoring
- Data drift detection
- Cost tracking

### MLOps
- CI/CD for ML (GitHub Actions)
- Automated testing (unit, integration)
- Model retraining pipelines
- Feature stores
- Experiment tracking (MLflow, W&B)
- Model governance

### Optimization
- Model compression (quantization, pruning)
- Inference optimization (ONNX, TensorRT)
- Caching strategies
- Load balancing
- Auto-scaling

## 🎯 Real-World Scenarios

- **Real-time Predictions**: Credit scoring API
- **Batch Processing**: Daily recommendation updates
- **Streaming**: Live fraud detection
- **Edge Deployment**: Mobile ML models
- **Multi-model Serving**: Ensemble predictions
- **LLM Applications**: RAG API with streaming

## 📂 Module Structure

```
10-deployment-production/
├── README.md (you are here)
├── api_development/
│   ├── 01_fastapi_basics.py         # First API
│   ├── 02_ml_api.py                 # Serve ML model
│   ├── 03_validation.py             # Input validation
│   ├── 04_async_endpoints.py        # Async operations
│   ├── 05_streaming.py              # WebSocket streaming
│   └── 06_authentication.py         # Secure API
├── containerization/
│   ├── 01_dockerfile/               # Basic Dockerfile
│   ├── 02_ml_dockerfile/            # ML-optimized
│   ├── 03_docker_compose/           # Multi-service
│   └── 04_gpu_docker/               # GPU support
├── cloud_deployment/
│   ├── aws/
│   │   ├── 01_ec2_deployment.sh     # Deploy to EC2
│   │   ├── 02_lambda_serverless.py  # Serverless
│   │   └── 03_sagemaker.py          # Managed ML
│   ├── gcp/
│   │   ├── 01_cloud_run.sh          # Container deployment
│   │   └── 02_vertex_ai.py          # Managed ML
│   └── azure/
│       ├── 01_app_service.sh        # Web app
│       └── 02_azure_ml.py           # Managed ML
├── monitoring/
│   ├── 01_logging.py                # Structured logging
│   ├── 02_metrics.py                # Prometheus metrics
│   ├── 03_tracing.py                # Distributed tracing
│   ├── 04_model_monitoring.py       # Performance tracking
│   └── 05_drift_detection.py        # Data drift
├── mlops/
│   ├── 01_experiment_tracking.py    # MLflow
│   ├── 02_model_registry.py         # Version models
│   ├── 03_ci_cd_pipeline.yml        # GitHub Actions
│   ├── 04_automated_testing.py      # Test ML code
│   └── 05_retraining_pipeline.py    # Auto retrain
├── optimization/
│   ├── 01_model_quantization.py     # Reduce model size
│   ├── 02_onnx_conversion.py        # Convert to ONNX
│   ├── 03_caching.py                # Response caching
│   └── 04_load_balancing.py         # Scale horizontally
└── projects/
    ├── ml_api_complete/             # Full ML API
    ├── rag_api/                     # RAG with streaming
    └── production_ready/            # Enterprise-grade
```

## 💡 Deployment Architecture

### Simple Architecture
```
User → Load Balancer → API Server → ML Model
                           ↓
                      Database
```

### Production Architecture
```
User → CDN → Load Balancer → API Servers (auto-scaled)
                                  ↓
                          Model Serving Layer
                                  ↓
                     Vector DB | Cache | Database
                                  ↓
                          Monitoring & Logging
```

## 🔧 FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]
    confidence = model.predict_proba([request.features]).max()
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )
```

## 📊 Deployment Strategies

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| Blue-Green | Production apps | Zero downtime | 2x resources |
| Canary | Gradual rollout | Low risk | Complex |
| Rolling | Continuous deploy | Resource efficient | Temporary mixed versions |
| Shadow | Test in production | Safe testing | Higher cost |
| A/B Testing | Compare models | Data-driven | Need traffic splitting |

## 🎓 Production Checklist

**Before Deployment**:
- [ ] Model performance acceptable
- [ ] Input validation implemented
- [ ] Error handling comprehensive
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Load testing completed
- [ ] Security review done
- [ ] Documentation written

**After Deployment**:
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Alerts configured
- [ ] Backup/rollback plan ready
- [ ] On-call rotation established
- [ ] Performance within SLA
- [ ] Cost tracking enabled

## 🚀 Quick Start

```bash
# Install dependencies
poetry add fastapi uvicorn pydantic

# Create simple API
poetry run python 10-deployment-production/api_development/01_fastapi_basics.py

# Run locally
uvicorn main:app --reload

# Build Docker image
docker build -t ml-api .

# Run container
docker run -p 8000:8000 ml-api

# Deploy to cloud (example: AWS)
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/ml-api:latest
```

## 💰 Cost Optimization

**1. Right-size instances**
- Don't over-provision
- Use spot instances for batch jobs
- Auto-scale based on demand

**2. Cache aggressively**
- Cache predictions for common inputs
- Use CDN for static assets
- Cache embeddings

**3. Batch when possible**
- Process in batches vs one-by-one
- Schedule batch jobs during off-peak

**4. Optimize models**
- Quantization (8-bit, 4-bit)
- Pruning (remove unnecessary weights)
- Distillation (smaller student model)

**5. Monitor costs**
- Set up billing alerts
- Track cost per prediction
- Optimize expensive operations

## ⚠️ Common Production Issues

**1. Model Performance Degradation**
- Data drift (input distribution changes)
- Concept drift (relationships change)
- Solution: Monitor + retrain

**2. High Latency**
- Model too large
- No caching
- Solution: Optimize, cache, scale

**3. Out of Memory**
- Batch size too large
- Model too big for instance
- Solution: Smaller batches, bigger instance

**4. Cold Start (Serverless)**
- Model loading takes time
- Solution: Keep warm, use smaller models

**5. Security Vulnerabilities**
- Exposed API keys
- No rate limiting
- Solution: Authentication, rate limits, secrets management

## 📈 Monitoring Metrics

**System Metrics**:
- CPU/GPU utilization
- Memory usage
- Request latency (p50, p95, p99)
- Request rate (RPS)
- Error rate

**Model Metrics**:
- Prediction accuracy
- Confidence scores
- Input distribution
- Feature importance
- Model latency

**Business Metrics**:
- Cost per prediction
- User engagement
- Conversion rate
- Revenue impact

## 🎯 Expected Outcomes

After this module:
- ✅ Build production APIs with FastAPI
- ✅ Containerize ML applications
- ✅ Deploy to cloud platforms
- ✅ Implement monitoring and logging
- ✅ Set up CI/CD pipelines
- ✅ Handle production incidents
- ✅ Optimize for cost and performance
- ✅ Ship ML/AI to real users!

---

## 🎓 Congratulations!

You've completed the ML/AI Mastery curriculum! You now have:

- ✅ Strong Python foundations (Modules 0-2)
- ✅ Mathematical intuition (Module 3)
- ✅ Data processing skills (Modules 4-5)
- ✅ ML fundamentals (Module 6)
- ✅ Deep learning expertise (Module 7)
- ✅ LLM/RAG capabilities (Modules 8-9)
- ✅ Production deployment skills (Module 10)

**You're ready to build and deploy real AI products!** 🚀

---

**Next Steps**:
1. Build portfolio projects
2. Contribute to open source
3. Apply to ML/AI positions
4. Keep learning (field evolves fast!)
5. Join ML communities (Twitter, Discord, Reddit)
