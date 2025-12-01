"""
FastAPI ML Model Deployment

Learn how to deploy ML models as production REST APIs.
Focus: Building scalable, maintainable ML services.

Install: poetry add fastapi uvicorn pydantic
Run: poetry run python 10-deployment-production/api_development/01_fastapi_ml_serving.py
Then visit: http://localhost:8000/docs
"""

import math
import random
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json


# ============================================================================
# 1. Why FastAPI for ML?
# ============================================================================

def demo_why_fastapi():
    """
    Why FastAPI is perfect for ML deployment.
    
    INTUITION - The Restaurant Analogy:
    
    Your ML Model = Chef (knows how to cook)
    Need: Restaurant (serving customers)
    
    Bad restaurant (Flask/raw Python):
    - Manual order taking (no automatic validation)
    - Slow service (no async support)
    - Messy menu (no auto-documentation)
    - Errors everywhere (weak type checking)
    
    Good restaurant (FastAPI):
    - Digital ordering (automatic validation)
    - Fast service (async, concurrent requests)
    - Beautiful menu (auto-generated docs)
    - Quality control (type checking)
    
    WHY FASTAPI WINS FOR ML:
    
    1. Speed:
       - Async/await support (handle 1000s of requests)
       - One of fastest Python frameworks
       - Critical for production ML (users don't wait!)
    
    2. Type Safety:
       - Pydantic models (catch errors early)
       - Automatic validation (bad input rejected)
       - Less debugging in production
    
    3. Auto Documentation:
       - Swagger UI (interactive API docs)
       - ReDoc (beautiful documentation)
       - No manual doc writing!
    
    4. Modern Python:
       - Type hints (clear code)
       - Async/await (efficient I/O)
       - Dependency injection (clean architecture)
    
    5. Easy Testing:
       - Built-in test client
       - Simple unit tests
       - Fast CI/CD
    
    Real Impact:
    
    Flask API:
    - 100 req/sec max
    - Manual validation (bugs!)
    - No docs (teammates confused)
    - Hard to maintain
    
    FastAPI:
    - 1000+ req/sec
    - Automatic validation
    - Beautiful docs
    - Easy to maintain ✅
    
    TYPICAL ML API NEEDS:
    
    Endpoint 1: /predict
    - Input: Features (validated!)
    - Output: Prediction + confidence
    - Latency: <100ms
    
    Endpoint 2: /batch_predict
    - Input: List of samples
    - Output: List of predictions
    - Latency: <1sec for 100 samples
    
    Endpoint 3: /health
    - Check if model loaded
    - Check if dependencies work
    - Used by load balancers
    
    Endpoint 4: /metrics
    - Request count
    - Latency stats
    - Error rates
    - For monitoring systems
    """
    print("=" * 70)
    print("1. Why FastAPI for ML Deployment?")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Restaurant Analogy")
    print()
    print("   Your ML Model = Chef (trained, ready to predict)")
    print("   Need: Restaurant system to serve predictions")
    print()
    print("   ❌ Bad Restaurant (Flask):")
    print("      • Manual order validation → Errors slip through")
    print("      • Slow service → Can't handle rush hour")
    print("      • No menu → Customers confused")
    print("      • No quality control → Wrong orders")
    print()
    print("   ✅ Good Restaurant (FastAPI):")
    print("      • Digital ordering → Auto-validate inputs")
    print("      • Fast service → Handle 1000s of requests")
    print("      • Beautiful menu → Auto-generated docs")
    print("      • Quality control → Type checking catches errors")
    print()
    
    print("🚀 KEY FASTAPI FEATURES FOR ML:")
    print()
    print("   1️⃣  Automatic Validation:")
    print("      User sends: {'age': 'twenty-five'}  ← String instead of int")
    print("      FastAPI: Rejects with clear error message")
    print("      Your code: Never sees bad data! ✅")
    print()
    print("   2️⃣  Async/Await (Speed):")
    print("      Traditional:")
    print("      Request 1 → Process → Response (blocks other requests)")
    print("      Request 2 → Waits... → Process → Response")
    print("      100 req/sec max")
    print()
    print("      FastAPI with async:")
    print("      Request 1 → Start processing...")
    print("      Request 2 → Start processing... (concurrent!)")
    print("      Request 3 → Start processing...")
    print("      All finish quickly → 1000+ req/sec ⚡")
    print()
    print("   3️⃣  Auto-Generated Docs:")
    print("      Write: @app.post('/predict')")
    print("      Get: Interactive Swagger UI at /docs")
    print("      Frontend devs can test immediately!")
    print()
    print("   4️⃣  Type Safety:")
    print("      class PredictionRequest(BaseModel):")
    print("          age: int  # Must be integer")
    print("          income: float  # Must be float")
    print()
    print("      Catches bugs before production! ✅")
    print()
    
    print("📊 PERFORMANCE COMPARISON:")
    print()
    print("   Scenario: ML prediction API (100ms model latency)")
    print()
    print("   Flask (synchronous):")
    print("   • Requests/sec: 100")
    print("   • With 10 workers: 1,000 req/sec")
    print("   • Memory: 500MB per worker = 5GB total")
    print()
    print("   FastAPI (async):")
    print("   • Requests/sec: 1,000+ (single worker!)")
    print("   • With 4 workers: 4,000+ req/sec")
    print("   • Memory: 500MB per worker = 2GB total")
    print()
    print("   💡 FastAPI: 4x throughput, 60% less memory!")
    print()
    
    print("🏗️  TYPICAL ML API STRUCTURE:")
    print()
    print("   Endpoint: POST /predict")
    print("   Input:")
    print("   {")
    print('     "features": {')
    print('       "age": 35,')
    print('       "income": 75000,')
    print('       "credit_score": 720')
    print("     }")
    print("   }")
    print()
    print("   Output:")
    print("   {")
    print('     "prediction": "approved",')
    print('     "confidence": 0.87,')
    print('     "model_version": "v2.1.0",')
    print('     "latency_ms": 45')
    print("   }")
    print()
    
    print("💡 FASTAPI vs ALTERNATIVES:")
    print()
    print("   Flask:")
    print("   ✅ Simple, widely known")
    print("   ❌ No async, no auto-validation, no auto-docs")
    print()
    print("   Django:")
    print("   ✅ Full-featured (DB, auth, admin)")
    print("   ❌ Overkill for APIs, slower, more complex")
    print()
    print("   FastAPI:")
    print("   ✅ Fast, async, auto-validation, auto-docs")
    print("   ✅ Perfect for ML APIs!")
    print("   ❌ Newer (less examples, but growing fast)")


# ============================================================================
# 2. Building a Simple ML API
# ============================================================================

class PredictionRequest:
    """
    Request model for predictions.
    
    In real FastAPI, use Pydantic BaseModel:
    from pydantic import BaseModel
    
    class PredictionRequest(BaseModel):
        age: int
        income: float
        credit_score: int
    """
    def __init__(self, age: int, income: float, credit_score: int):
        self.age = age
        self.income = income
        self.credit_score = credit_score


class PredictionResponse:
    """
    Response model for predictions.
    """
    def __init__(self, prediction: str, confidence: float, 
                 model_version: str, latency_ms: float):
        self.prediction = prediction
        self.confidence = confidence
        self.model_version = model_version
        self.latency_ms = latency_ms
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms
        }


class SimpleLoanModel:
    """
    Simple ML model for loan approval prediction.
    
    In reality, this would be:
    - Trained scikit-learn model (pickle)
    - PyTorch model (.pt file)
    - TensorFlow model (.h5 or SavedModel)
    - ONNX model (.onnx)
    """
    
    def __init__(self, version: str = "v1.0.0"):
        """Initialize model."""
        self.version = version
        self.model_loaded = True
        print(f"✅ Model {version} loaded successfully")
    
    def predict(self, age: int, income: float, credit_score: int) -> Tuple[str, float]:
        """
        Make loan approval prediction.
        
        Simplified logic (real model would use trained weights):
        - High credit score (>700) + decent income → Approve
        - Everything else → Deny
        
        Returns:
            (prediction, confidence)
        """
        # Simulate model inference time
        import time
        time.sleep(0.001)  # 1ms inference
        
        # Simple decision logic (replace with real model.predict())
        score = 0.0
        
        # Credit score is most important
        if credit_score >= 750:
            score += 0.5
        elif credit_score >= 700:
            score += 0.3
        elif credit_score >= 650:
            score += 0.1
        
        # Income matters
        if income >= 80000:
            score += 0.3
        elif income >= 50000:
            score += 0.2
        elif income >= 30000:
            score += 0.1
        
        # Age factor
        if 25 <= age <= 60:
            score += 0.2
        elif 18 <= age <= 70:
            score += 0.1
        
        # Make prediction
        if score >= 0.6:
            return "approved", score
        else:
            return "denied", 1.0 - score


class FastAPISimulator:
    """
    Simulates FastAPI application.
    
    In real code, use:
    from fastapi import FastAPI
    app = FastAPI()
    """
    
    def __init__(self):
        """Initialize application."""
        self.model = SimpleLoanModel(version="v1.0.0")
        self.request_count = 0
        self.routes = {}
    
    def post(self, path: str):
        """Decorator to register POST endpoint."""
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator
    
    def get(self, path: str):
        """Decorator to register GET endpoint."""
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator
    
    def call_endpoint(self, path: str, data: Optional[Dict] = None) -> Dict:
        """Simulate calling an endpoint."""
        if path not in self.routes:
            return {"error": "Endpoint not found"}
        
        self.request_count += 1
        return self.routes[path](data)


def demo_simple_api():
    """
    Build a simple ML prediction API.
    """
    print("\n" + "=" * 70)
    print("2. Building a Simple ML API")
    print("=" * 70)
    print()
    
    print("🏗️  STEP 1: Initialize FastAPI App")
    print()
    print("   Real code:")
    print("   ```python")
    print("   from fastapi import FastAPI")
    print("   ")
    print("   app = FastAPI(")
    print('       title="Loan Approval API",')
    print('       description="ML model for loan predictions",')
    print('       version="1.0.0"')
    print("   )")
    print("   ```")
    print()
    
    app = FastAPISimulator()
    print("   ✅ FastAPI app created")
    print()
    
    print("📦 STEP 2: Load ML Model")
    print()
    print("   In reality:")
    print("   ```python")
    print("   import joblib")
    print("   model = joblib.load('loan_model.pkl')")
    print("   ```")
    print()
    print("   Or for PyTorch:")
    print("   ```python")
    print("   import torch")
    print("   model = torch.load('model.pt')")
    print("   model.eval()")
    print("   ```")
    print()
    
    print("🔌 STEP 3: Create Prediction Endpoint")
    print()
    print("   Real code:")
    print("   ```python")
    print("   @app.post('/predict')")
    print("   async def predict(request: PredictionRequest):")
    print("       # Validate input (automatic!)")
    print("       # Make prediction")
    print("       # Return response")
    print("       ...")
    print("   ```")
    print()
    
    # Define prediction endpoint
    @app.post("/predict")
    def predict(data: Dict) -> Dict:
        """Make loan approval prediction."""
        start_time = datetime.now()
        
        # In real FastAPI, data is auto-validated by Pydantic
        try:
            request = PredictionRequest(
                age=data["age"],
                income=data["income"],
                credit_score=data["credit_score"]
            )
        except KeyError as e:
            return {"error": f"Missing field: {e}"}
        
        # Make prediction
        prediction, confidence = app.model.predict(
            request.age,
            request.income,
            request.credit_score
        )
        
        # Calculate latency
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        # Return response
        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=app.model.version,
            latency_ms=latency
        )
        
        return response.dict()
    
    print("   ✅ POST /predict endpoint created")
    print()
    
    print("🏥 STEP 4: Add Health Check Endpoint")
    print()
    print("   Why: Load balancers check if service is healthy")
    print()
    
    @app.get("/health")
    def health(data: Optional[Dict] = None) -> Dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": app.model.model_loaded,
            "model_version": app.model.version,
            "uptime_seconds": 3600  # Simulated
        }
    
    print("   ✅ GET /health endpoint created")
    print()
    
    print("📊 STEP 5: Add Metrics Endpoint")
    print()
    
    @app.get("/metrics")
    def metrics(data: Optional[Dict] = None) -> Dict:
        """Metrics endpoint for monitoring."""
        return {
            "total_requests": app.request_count,
            "avg_latency_ms": 45.2,  # Simulated
            "error_rate": 0.001,  # 0.1%
            "model_version": app.model.version
        }
    
    print("   ✅ GET /metrics endpoint created")
    print()
    
    print("🧪 STEP 6: Test the API")
    print()
    
    # Test case 1: Approved loan
    print("   Test 1: High credit score, good income")
    test1 = {
        "age": 35,
        "income": 80000,
        "credit_score": 750
    }
    result1 = app.call_endpoint("/predict", test1)
    print(f"   Input: {test1}")
    print(f"   Output: {json.dumps(result1, indent=2)}")
    print()
    
    # Test case 2: Denied loan
    print("   Test 2: Low credit score")
    test2 = {
        "age": 25,
        "income": 30000,
        "credit_score": 600
    }
    result2 = app.call_endpoint("/predict", test2)
    print(f"   Input: {test2}")
    print(f"   Output: {json.dumps(result2, indent=2)}")
    print()
    
    # Test case 3: Health check
    print("   Test 3: Health check")
    health_result = app.call_endpoint("/health")
    print(f"   Output: {json.dumps(health_result, indent=2)}")
    print()
    
    # Test case 4: Metrics
    print("   Test 4: Metrics")
    metrics_result = app.call_endpoint("/metrics")
    print(f"   Output: {json.dumps(metrics_result, indent=2)}")
    print()
    
    print("✅ API WORKING!")
    print()
    print("   In production, run with:")
    print("   ```bash")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000")
    print("   ```")
    print()
    print("   Then visit:")
    print("   • http://localhost:8000/docs (Swagger UI)")
    print("   • http://localhost:8000/redoc (ReDoc)")
    print("   • Interactive testing right in the browser!")


# ============================================================================
# 3. Production Best Practices
# ============================================================================

def demo_best_practices():
    """
    Production best practices for ML APIs.
    """
    print("\n" + "=" * 70)
    print("3. Production Best Practices")
    print("=" * 70)
    print()
    
    print("🔒 BEST PRACTICE 1: Input Validation")
    print()
    print("   Problem: Users send garbage data")
    print()
    print("   Bad:")
    print("   ```python")
    print("   age = request.json['age']  # Might be string, missing, etc.")
    print("   ```")
    print()
    print("   Good (FastAPI + Pydantic):")
    print("   ```python")
    print("   class PredictionRequest(BaseModel):")
    print("       age: int = Field(ge=18, le=100)  # 18-100")
    print("       income: float = Field(ge=0)  # Non-negative")
    print("       credit_score: int = Field(ge=300, le=850)")
    print("   ")
    print("   @app.post('/predict')")
    print("   async def predict(req: PredictionRequest):")
    print("       # req is guaranteed valid!")
    print("   ```")
    print()
    print("   Benefits:")
    print("   ✅ Automatic validation (bad input rejected)")
    print("   ✅ Clear error messages to users")
    print("   ✅ Your code never sees invalid data")
    print()
    
    print("⚡ BEST PRACTICE 2: Async/Await for I/O")
    print()
    print("   Use async when:")
    print("   • Loading model from cloud storage")
    print("   • Calling external APIs (feature stores)")
    print("   • Database queries")
    print("   • Anything that waits (network, disk)")
    print()
    print("   ```python")
    print("   @app.post('/predict')")
    print("   async def predict(req: PredictionRequest):")
    print("       # Fetch user features from DB (async)")
    print("       user_features = await db.get_user(req.user_id)")
    print("       ")
    print("       # Call external API (async)")
    print("       enriched_data = await external_api.enrich(user_features)")
    print("       ")
    print("       # CPU-bound prediction (sync)")
    print("       prediction = model.predict(enriched_data)")
    print("       ")
    print("       return prediction")
    print("   ```")
    print()
    print("   Result: 10x more concurrent requests!")
    print()
    
    print("🔄 BEST PRACTICE 3: Model Versioning")
    print()
    print("   Problem: Deploy new model, but keep old one for rollback")
    print()
    print("   Strategy 1: Include version in response")
    print("   ```python")
    print("   return {")
    print('       "prediction": "approved",')
    print('       "model_version": "v2.1.0",  # Track which model')
    print('       "deployed_at": "2024-01-15"')
    print("   }")
    print("   ```")
    print()
    print("   Strategy 2: A/B testing")
    print("   ```python")
    print("   # Route 10% of traffic to new model")
    print("   if random.random() < 0.1:")
    print("       prediction = model_v2.predict(features)")
    print("   else:")
    print("       prediction = model_v1.predict(features)")
    print("   ```")
    print()
    print("   Strategy 3: Blue-Green Deployment")
    print("   • Blue: Current model (v1.0)")
    print("   • Green: New model (v2.0)")
    print("   • Switch traffic from Blue → Green")
    print("   • If issues, instant rollback to Blue")
    print()
    
    print("📊 BEST PRACTICE 4: Monitoring & Logging")
    print()
    print("   Track everything:")
    print()
    print("   ```python")
    print("   import logging")
    print("   from prometheus_client import Counter, Histogram")
    print("   ")
    print("   # Metrics")
    print("   prediction_count = Counter('predictions_total', 'Total predictions')")
    print("   latency = Histogram('prediction_latency', 'Prediction latency')")
    print("   ")
    print("   @app.post('/predict')")
    print("   async def predict(req: PredictionRequest):")
    print("       start = time.time()")
    print("       ")
    print("       # Log request")
    print("       logging.info(f'Prediction request: {req.dict()}')")
    print("       ")
    print("       # Make prediction")
    print("       result = model.predict(req)")
    print("       ")
    print("       # Record metrics")
    print("       prediction_count.inc()")
    print("       latency.observe(time.time() - start)")
    print("       ")
    print("       # Log result")
    print("       logging.info(f'Prediction: {result}')")
    print("       ")
    print("       return result")
    print("   ```")
    print()
    print("   Monitor:")
    print("   • Request rate (requests/sec)")
    print("   • Latency (p50, p95, p99)")
    print("   • Error rate (5xx errors)")
    print("   • Prediction distribution (detect drift)")
    print()
    
    print("🔒 BEST PRACTICE 5: Error Handling")
    print()
    print("   ```python")
    print("   from fastapi import HTTPException")
    print("   ")
    print("   @app.post('/predict')")
    print("   async def predict(req: PredictionRequest):")
    print("       try:")
    print("           prediction = model.predict(req)")
    print("           return prediction")
    print("       except ValueError as e:")
    print("           # Bad input that passed validation")
    print("           raise HTTPException(status_code=400, detail=str(e))")
    print("       except Exception as e:")
    print("           # Internal error")
    print("           logging.error(f'Prediction failed: {e}')")
    print("           raise HTTPException(status_code=500, detail='Internal error')")
    print("   ```")
    print()
    print("   Return proper HTTP status codes:")
    print("   • 200: Success")
    print("   • 400: Bad request (user's fault)")
    print("   • 422: Validation error (Pydantic)")
    print("   • 500: Internal error (your fault)")
    print("   • 503: Service unavailable (overloaded)")
    print()
    
    print("🚀 BEST PRACTICE 6: Batch Predictions")
    print()
    print("   For high throughput, support batching:")
    print()
    print("   ```python")
    print("   class BatchRequest(BaseModel):")
    print("       samples: List[PredictionRequest]")
    print("   ")
    print("   @app.post('/batch_predict')")
    print("   async def batch_predict(req: BatchRequest):")
    print("       # Convert to numpy array")
    print("       X = np.array([s.dict() for s in req.samples])")
    print("       ")
    print("       # Batch prediction (much faster!)")
    print("       predictions = model.predict(X)")
    print("       ")
    print("       return {'predictions': predictions.tolist()}")
    print("   ```")
    print()
    print("   Benefits:")
    print("   • 10x faster than individual predictions")
    print("   • Better GPU utilization")
    print("   • Reduce API overhead")
    print()
    
    print("🐳 BEST PRACTICE 7: Containerization (Docker)")
    print()
    print("   Dockerfile:")
    print("   ```dockerfile")
    print("   FROM python:3.11-slim")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Install dependencies")
    print("   COPY requirements.txt .")
    print("   RUN pip install -r requirements.txt")
    print("   ")
    print("   # Copy code")
    print("   COPY . .")
    print("   ")
    print("   # Download model")
    print("   RUN python download_model.py")
    print("   ")
    print("   # Run app")
    print("   CMD ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000']")
    print("   ```")
    print()
    print("   Benefits:")
    print("   ✅ Reproducible (same environment everywhere)")
    print("   ✅ Easy deployment (just docker run)")
    print("   ✅ Scalable (Kubernetes, ECS, etc.)")
    print()
    
    print("☁️  BEST PRACTICE 8: Cloud Deployment")
    print()
    print("   Options:")
    print()
    print("   1️⃣  AWS:")
    print("      • ECS/Fargate: Managed containers")
    print("      • Lambda: Serverless (if model is small)")
    print("      • SageMaker: Fully managed ML")
    print()
    print("   2️⃣  GCP:")
    print("      • Cloud Run: Serverless containers")
    print("      • GKE: Kubernetes")
    print("      • Vertex AI: Managed ML")
    print()
    print("   3️⃣  Azure:")
    print("      • Container Instances: Simple deployment")
    print("      • AKS: Kubernetes")
    print("      • Azure ML: Managed ML")
    print()
    print("   Recommendation for starting:")
    print("   → AWS Fargate or GCP Cloud Run")
    print("      (Easy, auto-scaling, pay-per-use)")


# ============================================================================
# 4. Complete Example Structure
# ============================================================================

def demo_complete_structure():
    """
    Complete project structure for production ML API.
    """
    print("\n" + "=" * 70)
    print("4. Production Project Structure")
    print("=" * 70)
    print()
    
    print("📁 RECOMMENDED PROJECT STRUCTURE:")
    print()
    print("   ml-api/")
    print("   ├── app/")
    print("   │   ├── __init__.py")
    print("   │   ├── main.py              # FastAPI app")
    print("   │   ├── models.py            # Pydantic models")
    print("   │   ├── ml_model.py          # ML model loading/prediction")
    print("   │   ├── config.py            # Configuration")
    print("   │   └── utils.py             # Helper functions")
    print("   ├── models/")
    print("   │   └── model_v1.pkl         # Trained model file")
    print("   ├── tests/")
    print("   │   ├── test_api.py          # API tests")
    print("   │   └── test_model.py        # Model tests")
    print("   ├── docker/")
    print("   │   └── Dockerfile")
    print("   ├── .env                     # Environment variables")
    print("   ├── requirements.txt         # Python dependencies")
    print("   └── README.md")
    print()
    
    print("📄 KEY FILES:")
    print()
    print("   1️⃣  main.py (FastAPI app):")
    print("   ```python")
    print("   from fastapi import FastAPI, HTTPException")
    print("   from app.models import PredictionRequest, PredictionResponse")
    print("   from app.ml_model import ModelService")
    print("   from app.config import settings")
    print("   ")
    print("   app = FastAPI(title='ML API')")
    print("   model_service = ModelService()")
    print("   ")
    print("   @app.on_event('startup')")
    print("   async def startup():")
    print("       await model_service.load_model()")
    print("   ")
    print("   @app.post('/predict', response_model=PredictionResponse)")
    print("   async def predict(request: PredictionRequest):")
    print("       return await model_service.predict(request)")
    print("   ```")
    print()
    
    print("   2️⃣  models.py (Pydantic models):")
    print("   ```python")
    print("   from pydantic import BaseModel, Field")
    print("   ")
    print("   class PredictionRequest(BaseModel):")
    print("       age: int = Field(ge=18, le=100)")
    print("       income: float = Field(ge=0)")
    print("       credit_score: int = Field(ge=300, le=850)")
    print("   ")
    print("   class PredictionResponse(BaseModel):")
    print("       prediction: str")
    print("       confidence: float")
    print("       model_version: str")
    print("   ```")
    print()
    
    print("   3️⃣  ml_model.py (ML service):")
    print("   ```python")
    print("   import joblib")
    print("   from pathlib import Path")
    print("   ")
    print("   class ModelService:")
    print("       def __init__(self):")
    print("           self.model = None")
    print("           self.version = 'v1.0.0'")
    print("       ")
    print("       async def load_model(self):")
    print("           model_path = Path('models/model_v1.pkl')")
    print("           self.model = joblib.load(model_path)")
    print("           ")
    print("       async def predict(self, request):")
    print("           features = [[request.age, request.income, request.credit_score]]")
    print("           prediction = self.model.predict(features)[0]")
    print("           confidence = self.model.predict_proba(features).max()")
    print("           return {")
    print("               'prediction': prediction,")
    print("               'confidence': confidence,")
    print("               'model_version': self.version")
    print("           }")
    print("   ```")
    print()
    
    print("   4️⃣  config.py (Configuration):")
    print("   ```python")
    print("   from pydantic_settings import BaseSettings")
    print("   ")
    print("   class Settings(BaseSettings):")
    print("       app_name: str = 'ML API'")
    print("       model_path: str = 'models/model_v1.pkl'")
    print("       log_level: str = 'INFO'")
    print("       ")
    print("       class Config:")
    print("           env_file = '.env'")
    print("   ")
    print("   settings = Settings()")
    print("   ```")
    print()
    
    print("   5️⃣  Dockerfile:")
    print("   ```dockerfile")
    print("   FROM python:3.11-slim")
    print("   ")
    print("   WORKDIR /app")
    print("   ")
    print("   # Install dependencies")
    print("   COPY requirements.txt .")
    print("   RUN pip install --no-cache-dir -r requirements.txt")
    print("   ")
    print("   # Copy application")
    print("   COPY app/ ./app/")
    print("   COPY models/ ./models/")
    print("   ")
    print("   # Expose port")
    print("   EXPOSE 8000")
    print("   ")
    print("   # Run application")
    print("   CMD ['uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000']")
    print("   ```")
    print()
    
    print("🚀 DEPLOYMENT COMMANDS:")
    print()
    print("   Local Development:")
    print("   ```bash")
    print("   # Install dependencies")
    print("   pip install -r requirements.txt")
    print("   ")
    print("   # Run server (with auto-reload)")
    print("   uvicorn app.main:app --reload")
    print("   ")
    print("   # Visit: http://localhost:8000/docs")
    print("   ```")
    print()
    print("   Docker:")
    print("   ```bash")
    print("   # Build image")
    print("   docker build -t ml-api:v1 .")
    print("   ")
    print("   # Run container")
    print("   docker run -p 8000:8000 ml-api:v1")
    print("   ```")
    print()
    print("   Kubernetes:")
    print("   ```bash")
    print("   # Deploy")
    print("   kubectl apply -f k8s/deployment.yaml")
    print("   ")
    print("   # Scale")
    print("   kubectl scale deployment ml-api --replicas=5")
    print("   ```")
    print()
    
    print("📊 TESTING:")
    print()
    print("   ```python")
    print("   from fastapi.testclient import TestClient")
    print("   from app.main import app")
    print("   ")
    print("   client = TestClient(app)")
    print("   ")
    print("   def test_predict():")
    print("       response = client.post('/predict', json={")
    print("           'age': 35,")
    print("           'income': 80000,")
    print("           'credit_score': 750")
    print("       })")
    print("       assert response.status_code == 200")
    print("       assert 'prediction' in response.json()")
    print("   ```")
    print()
    print("   Run tests:")
    print("   ```bash")
    print("   pytest tests/")
    print("   ```")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🚀 FastAPI ML Model Deployment\n")
    print("Learn how to deploy ML models as production APIs!")
    print()
    
    demo_why_fastapi()
    demo_simple_api()
    demo_best_practices()
    demo_complete_structure()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why FastAPI?
   - Fast: Async support, 1000+ req/sec
   - Type-safe: Automatic validation
   - Auto-docs: Swagger UI at /docs
   - Modern: Type hints, async/await

2. Basic Structure:
   - Load model on startup
   - Validate input (Pydantic)
   - Make prediction
   - Return structured response

3. Production Best Practices:
   - Input validation (Field constraints)
   - Async/await for I/O
   - Model versioning (A/B testing)
   - Monitoring & logging
   - Error handling
   - Batch predictions
   - Containerization (Docker)

4. Deployment Options:
   - AWS: Fargate, Lambda, SageMaker
   - GCP: Cloud Run, GKE, Vertex AI
   - Azure: Container Instances, AKS, Azure ML

Minimal FastAPI Example:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class Request(BaseModel):
    feature1: float
    feature2: float

@app.post('/predict')
async def predict(req: Request):
    X = [[req.feature1, req.feature2]]
    prediction = model.predict(X)[0]
    return {'prediction': prediction}
```

Run with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Production Checklist:
✅ Input validation (Pydantic)
✅ Health check endpoint
✅ Metrics endpoint (Prometheus)
✅ Logging (structured JSON)
✅ Error handling (proper HTTP codes)
✅ Docker container
✅ CI/CD pipeline
✅ Load testing (locust, k6)
✅ Monitoring (Grafana, Datadog)
✅ Auto-scaling (Kubernetes HPA)

Resources:
- FastAPI docs: fastapi.tiangolo.com
- Example repo: github.com/tiangolo/full-stack-fastapi-template
- Deployment guide: fastapi.tiangolo.com/deployment/

Congratulations! You now know how to:
• Build production ML APIs
• Deploy to cloud platforms
• Monitor and scale services
• Handle errors gracefully
• Version and A/B test models
""")


if __name__ == "__main__":
    main()
