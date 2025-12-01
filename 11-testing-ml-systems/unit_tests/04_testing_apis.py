"""
Testing ML APIs with pytest

Learn how to test FastAPI endpoints for ML services.
Focus: API endpoints, request/response, error handling, performance.

Install: poetry add --group dev pytest pytest-asyncio fastapi httpx
Run: pytest unit_tests/04_testing_apis.py -v
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import List, Optional
import time


# ============================================================================
# 1. Why Test ML APIs?
# ============================================================================

def demo_why_test_apis():
    """
    Why testing ML APIs is critical.
    """
    print("=" * 70)
    print("1. Why Test ML APIs?")
    print("=" * 70)
    print()
    
    print("💥 REAL-WORLD API HORROR STORIES:")
    print()
    print("   Story 1: The 500 Error")
    print("   • Endpoint: /predict")
    print("   • Bug: Missing validation on input")
    print("   • User sends: {'age': 'twenty'}")
    print("   • Response: 500 Internal Server Error")
    print("   • Impact: Users can't use API 💀")
    print("   • Cost: Lost revenue + support tickets")
    print()
    print("   Story 2: The Timeout")
    print("   • Endpoint: /predict")
    print("   • Bug: Model loading takes 30 seconds")
    print("   • Result: Every request times out")
    print("   • Impact: 100% failure rate 💀")
    print("   • Cost: Complete service outage")
    print()
    print("   Story 3: The Memory Leak")
    print("   • Endpoint: /batch_predict")
    print("   • Bug: Not releasing memory after prediction")
    print("   • Result: Server crashes after 1000 requests")
    print("   • Impact: Need frequent restarts 💀")
    print("   • Cost: Poor reliability")
    print()
    
    print("🎯 WHY API TESTS MATTER:")
    print()
    print("   APIs are user-facing:")
    print("   • Bad API = Bad user experience")
    print("   • Errors = Support tickets")
    print("   • Slow = Users leave")
    print("   • Crashes = Lost revenue")
    print()
    print("   Tests prevent:")
    print("   • 500 errors (validation)")
    print("   • Timeouts (performance)")
    print("   • Memory leaks (load testing)")
    print("   • Security issues (input sanitization)")
    print()
    
    print("✅ WHAT TO TEST:")
    print()
    print("   1. Happy Path:")
    print("      • Valid input → 200 OK")
    print("      • Correct response format")
    print("   ")
    print("   2. Error Cases:")
    print("      • Invalid input → 422 Validation Error")
    print("      • Missing fields → 422")
    print("      • Server error → 500")
    print("   ")
    print("   3. Performance:")
    print("      • Response time < 1s")
    print("      • Handle 100 requests/sec")
    print("   ")
    print("   4. Security:")
    print("      • SQL injection prevented")
    print("      • XSS prevented")
    print("      • Rate limiting works")
    print()


# ============================================================================
# 2. Simple ML API
# ============================================================================

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[float] = Field(..., min_items=1, max_items=100)
    model_version: Optional[str] = Field(default="v1")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, 0.6, 0.7],
                "model_version": "v1"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: float
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


# Simple ML model (mock)
class SimpleModel:
    """Simple model for testing."""
    
    def __init__(self):
        self.is_loaded = True
        self.version = "v1"
    
    def predict(self, features: List[float]) -> float:
        """Make prediction."""
        if not features:
            raise ValueError("Features cannot be empty")
        
        # Simple rule: average of features
        return sum(features) / len(features)
    
    def predict_with_confidence(self, features: List[float]) -> tuple[float, float]:
        """Make prediction with confidence."""
        prediction = self.predict(features)
        
        # Simple confidence: inverse of variance
        variance = sum((f - prediction) ** 2 for f in features) / len(features)
        confidence = 1.0 / (1.0 + variance)
        
        return prediction, confidence


# FastAPI app
app = FastAPI(title="ML API", version="1.0.0")
model = SimpleModel()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
        version=model.version
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prediction endpoint."""
    start_time = time.time()
    
    try:
        prediction, confidence = model.predict_with_confidence(request.features)
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=request.model_version or "v1",
            latency_ms=latency_ms
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# 3. Testing API Endpoints
# ============================================================================

@pytest.fixture
def client():
    """Fixture providing test client."""
    return TestClient(app)


def test_health_check_returns_200(client):
    """Test that health check returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_response_format(client):
    """Test health check response format."""
    response = client.get("/health")
    data = response.json()
    
    # Check fields exist
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    
    # Check values
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert isinstance(data["version"], str)


def test_predict_valid_input(client):
    """Test prediction with valid input."""
    response = client.post("/predict", json={
        "features": [0.5, 0.6, 0.7],
        "model_version": "v1"
    })
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_response_types(client):
    """Test prediction response types."""
    response = client.post("/predict", json={
        "features": [0.5, 0.6, 0.7]
    })
    
    data = response.json()
    
    # Check types
    assert isinstance(data["prediction"], (int, float))
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["model_version"], str)
    assert isinstance(data["latency_ms"], (int, float))


def test_predict_confidence_range(client):
    """Test that confidence is in valid range."""
    response = client.post("/predict", json={
        "features": [0.5, 0.6, 0.7]
    })
    
    data = response.json()
    
    # Confidence between 0 and 1
    assert 0 <= data["confidence"] <= 1


def demo_testing_endpoints():
    """Demo testing API endpoints."""
    print("\n" + "=" * 70)
    print("2. Testing API Endpoints")
    print("=" * 70)
    print()
    
    print("✅ ENDPOINT TESTS:")
    print()
    print("   1. Status Code:")
    print("      assert response.status_code == 200")
    print("   ")
    print("   2. Response Format:")
    print("      assert 'prediction' in response.json()")
    print("   ")
    print("   3. Response Types:")
    print("      assert isinstance(data['prediction'], float)")
    print("   ")
    print("   4. Response Range:")
    print("      assert 0 <= data['confidence'] <= 1")
    print()
    
    print("🔧 TEST CLIENT:")
    print()
    print("   from fastapi.testclient import TestClient")
    print("   ")
    print("   client = TestClient(app)")
    print("   response = client.get('/health')")
    print("   assert response.status_code == 200")
    print()


# ============================================================================
# 4. Testing Error Handling
# ============================================================================

def test_predict_missing_features(client):
    """Test prediction with missing features."""
    response = client.post("/predict", json={
        "model_version": "v1"
    })
    
    # Should return 422 Validation Error
    assert response.status_code == 422


def test_predict_empty_features(client):
    """Test prediction with empty features."""
    response = client.post("/predict", json={
        "features": []
    })
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_predict_invalid_feature_type(client):
    """Test prediction with invalid feature type."""
    response = client.post("/predict", json={
        "features": ["not", "a", "number"]
    })
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_predict_too_many_features(client):
    """Test prediction with too many features."""
    response = client.post("/predict", json={
        "features": [0.5] * 101  # Max is 100
    })
    
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_invalid_endpoint(client):
    """Test invalid endpoint returns 404."""
    response = client.get("/invalid")
    assert response.status_code == 404


def demo_testing_errors():
    """Demo testing error handling."""
    print("\n" + "=" * 70)
    print("3. Testing Error Handling")
    print("=" * 70)
    print()
    
    print("🚨 ERROR CASES TO TEST:")
    print()
    print("   1. Missing Required Fields:")
    print("      POST /predict without 'features'")
    print("      → 422 Validation Error ✅")
    print("   ")
    print("   2. Invalid Types:")
    print("      features: ['a', 'b', 'c']")
    print("      → 422 Validation Error ✅")
    print("   ")
    print("   3. Out of Range:")
    print("      features: [0.5] * 101  (max 100)")
    print("      → 422 Validation Error ✅")
    print("   ")
    print("   4. Invalid Endpoint:")
    print("      GET /invalid")
    print("      → 404 Not Found ✅")
    print()
    
    print("💡 WHY ERROR TESTS MATTER:")
    print()
    print("   Without validation:")
    print("   • Server crashes (500 error)")
    print("   • Poor user experience")
    print("   • Security vulnerabilities")
    print()
    print("   With validation:")
    print("   • Clear error messages (422)")
    print("   • Graceful failures")
    print("   • Better UX ✅")
    print()


# ============================================================================
# 5. Testing Request Validation
# ============================================================================

@pytest.mark.parametrize("features,expected_status", [
    ([0.5, 0.6], 200),  # Valid
    ([0.1], 200),  # Valid (single feature)
    ([], 422),  # Empty
    ([0.5] * 100, 200),  # Max valid
    ([0.5] * 101, 422),  # Too many
])
def test_predict_feature_count_validation(client, features, expected_status):
    """Test feature count validation."""
    response = client.post("/predict", json={"features": features})
    assert response.status_code == expected_status


def test_predict_optional_model_version(client):
    """Test that model_version is optional."""
    # Without model_version
    response = client.post("/predict", json={
        "features": [0.5, 0.6]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data


def demo_testing_validation():
    """Demo testing request validation."""
    print("\n" + "=" * 70)
    print("4. Testing Request Validation")
    print("=" * 70)
    print()
    
    print("✅ VALIDATION TESTS:")
    print()
    print("   Pydantic validation:")
    print("   class PredictionRequest(BaseModel):")
    print("       features: List[float] = Field(..., min_items=1, max_items=100)")
    print("       model_version: Optional[str] = Field(default='v1')")
    print()
    print("   Tests:")
    print("   • min_items=1: Empty list rejected")
    print("   • max_items=100: 101 items rejected")
    print("   • List[float]: Strings rejected")
    print("   • Optional[str]: Can omit model_version")
    print()


# ============================================================================
# 6. Testing API Performance
# ============================================================================

def test_predict_response_time(client):
    """Test that prediction is fast enough."""
    response = client.post("/predict", json={
        "features": [0.5, 0.6, 0.7]
    })
    
    data = response.json()
    latency_ms = data["latency_ms"]
    
    # Should be less than 100ms
    assert latency_ms < 100, f"Prediction too slow: {latency_ms}ms"


def test_predict_batch_performance(client):
    """Test performance with multiple requests."""
    start_time = time.time()
    
    # Send 100 requests
    for _ in range(100):
        response = client.post("/predict", json={
            "features": [0.5, 0.6, 0.7]
        })
        assert response.status_code == 200
    
    elapsed = time.time() - start_time
    
    # 100 requests in less than 2 seconds
    assert elapsed < 2.0, f"Batch too slow: {elapsed:.2f}s"


def test_health_check_response_time(client):
    """Test that health check is fast."""
    start_time = time.time()
    response = client.get("/health")
    elapsed = (time.time() - start_time) * 1000
    
    # Health check should be < 10ms
    assert elapsed < 10, f"Health check too slow: {elapsed:.2f}ms"


def demo_testing_performance():
    """Demo testing API performance."""
    print("\n" + "=" * 70)
    print("5. Testing API Performance")
    print("=" * 70)
    print()
    
    print("⚡ PERFORMANCE TESTS:")
    print()
    print("   1. Response Time:")
    print("      • Single prediction < 100ms")
    print("      • Health check < 10ms")
    print("   ")
    print("   2. Throughput:")
    print("      • 100 requests < 2s")
    print("      • = 50 requests/second")
    print("   ")
    print("   3. Latency Percentiles:")
    print("      • P50 < 50ms")
    print("      • P95 < 200ms")
    print("      • P99 < 500ms")
    print()
    
    print("💡 WHY PERFORMANCE MATTERS:")
    print()
    print("   User expectations:")
    print("   • < 100ms: Instant")
    print("   • < 1s: Good")
    print("   • < 3s: Acceptable")
    print("   • > 3s: Frustrating")
    print()
    print("   Slow API:")
    print("   • Users leave")
    print("   • High bounce rate")
    print("   • Poor reviews")
    print()
    print("   Fast API:")
    print("   • Happy users ✅")
    print("   • High engagement ✅")
    print("   • Good reviews ✅")
    print()


# ============================================================================
# 7. Testing with Fixtures
# ============================================================================

@pytest.fixture
def valid_request():
    """Fixture providing valid request data."""
    return {
        "features": [0.5, 0.6, 0.7],
        "model_version": "v1"
    }


@pytest.fixture
def invalid_request():
    """Fixture providing invalid request data."""
    return {
        "features": [],  # Empty
        "model_version": "v1"
    }


def test_with_valid_fixture(client, valid_request):
    """Test with valid request fixture."""
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 200


def test_with_invalid_fixture(client, invalid_request):
    """Test with invalid request fixture."""
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Testing ML APIs\n")
    
    demo_why_test_apis()
    demo_testing_endpoints()
    demo_testing_errors()
    demo_testing_validation()
    demo_testing_performance()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Test APIs:
   - User-facing (bad API = bad UX)
   - Prevent 500 errors
   - Ensure fast responses
   - Catch validation errors

2. Endpoint Tests:
   ✅ Status codes (200, 422, 404, 500)
   ✅ Response format (JSON schema)
   ✅ Response types (float, str, etc.)
   ✅ Response ranges (0 ≤ confidence ≤ 1)

3. Error Handling:
   ✅ Missing fields → 422
   ✅ Invalid types → 422
   ✅ Out of range → 422
   ✅ Server errors → 500
   ✅ Invalid endpoint → 404

4. Validation Tests:
   - Pydantic models enforce types
   - Field constraints (min/max)
   - Optional fields
   - Default values

5. Performance Tests:
   ✅ Response time < 100ms
   ✅ Health check < 10ms
   ✅ Throughput: 50+ req/sec
   ✅ Handle load testing

API Testing Checklist:
```
Endpoint Tests:
□ Happy path (200 OK)
□ Response format correct
□ Response types correct
□ Response ranges valid

Error Tests:
□ Missing fields (422)
□ Invalid types (422)
□ Empty input (422)
□ Out of range (422)
□ Server errors (500)
□ Invalid endpoints (404)

Validation Tests:
□ Required fields enforced
□ Optional fields work
□ Type validation works
□ Range validation works

Performance Tests:
□ Response time < 100ms
□ Health check < 10ms
□ Handle 100 requests
□ No memory leaks
```

FastAPI TestClient:
```python
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get("/health")
assert response.status_code == 200

response = client.post("/predict", json={...})
data = response.json()
assert "prediction" in data
```

Next Steps:
→ 05_data_validation.py (Validate input data)
→ integration_tests/ (End-to-end tests)
""")


if __name__ == "__main__":
    main()
