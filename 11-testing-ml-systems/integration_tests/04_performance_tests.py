"""
Performance and Load Testing

Learn how to test ML systems under load and measure performance.
Focus: Response time, throughput, resource usage, stress testing.

Install: poetry add --group dev pytest pytest-benchmark
Run: pytest integration_tests/04_performance_tests.py -v
"""

import pytest
import time
import numpy as np
import pandas as pd
from typing import List
import concurrent.futures
import psutil
import os


# ============================================================================
# 1. Why Performance Testing?
# ============================================================================

def demo_why_performance_testing():
    """
    Why performance testing is critical for ML systems.
    """
    print("=" * 70)
    print("1. Why Performance Testing?")
    print("=" * 70)
    print()
    
    print("💥 REAL-WORLD PERFORMANCE DISASTERS:")
    print()
    print("   Story 1: The Slow Endpoint")
    print("   • Development: Responses in 100ms ✅")
    print("   • Production: Responses in 30 seconds 💀")
    print("   • Cause: Model loaded on every request")
    print("   • Impact: Timeouts, angry users")
    print("   • Solution: Load model once, cache")
    print()
    print("   Story 2: The Memory Leak")
    print("   • Development: 500MB memory usage")
    print("   • Production: 32GB → Crashes every 6 hours 💀")
    print("   • Cause: Not releasing tensors after prediction")
    print("   • Impact: Frequent restarts, downtime")
    print("   • Solution: Proper resource cleanup")
    print()
    print("   Story 3: The Bottleneck")
    print("   • Load test: 10 users → Works fine")
    print("   • Production: 1000 users → 95% failures 💀")
    print("   • Cause: Single-threaded processing")
    print("   • Impact: Black Friday disaster")
    print("   • Solution: Async processing, load balancing")
    print()
    
    print("🎯 WHY PERFORMANCE MATTERS:")
    print()
    print("   User expectations:")
    print("   • < 100ms: Instant (feels immediate)")
    print("   • < 1s: Fast (acceptable for search)")
    print("   • < 3s: Slow (user gets impatient)")
    print("   • > 3s: Very slow (user leaves)")
    print()
    print("   Business impact:")
    print("   • Amazon: 100ms delay → 1% sales loss")
    print("   • Google: 500ms delay → 20% traffic loss")
    print("   • Your ML API: Slow → Users leave")
    print()
    
    print("💰 ROI OF PERFORMANCE TESTING:")
    print()
    print("   Without performance tests:")
    print("   • Production outage: 4 hours")
    print("   • Lost revenue: $50,000")
    print("   • Emergency scaling: $10,000/month")
    print()
    print("   With performance tests:")
    print("   • Catch before deploy")
    print("   • Optimize proactively")
    print("   • Right-size infrastructure")
    print("   • Lost revenue: $0 ✅")
    print()


# ============================================================================
# 2. Simple ML Service to Test
# ============================================================================

class SimpleMLService:
    """Simple ML service for performance testing."""
    
    def __init__(self):
        self.model_weights = np.random.randn(1000, 100)  # Simulate model
        self.cache = {}
    
    def predict(self, features: List[float]) -> float:
        """Make prediction."""
        # Simulate computation
        features_array = np.array(features)
        result = np.dot(self.model_weights[0], features_array)
        return float(result)
    
    def predict_batch(self, features_list: List[List[float]]) -> List[float]:
        """Batch prediction."""
        return [self.predict(f) for f in features_list]
    
    def predict_with_cache(self, features: List[float]) -> float:
        """Prediction with caching."""
        cache_key = tuple(features)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.predict(features)
        self.cache[cache_key] = result
        return result


# ============================================================================
# 3. Response Time Testing
# ============================================================================

@pytest.fixture
def ml_service():
    """Fixture providing ML service."""
    return SimpleMLService()


def test_single_prediction_response_time(ml_service):
    """Test single prediction response time."""
    features = [0.5] * 100
    
    start_time = time.time()
    ml_service.predict(features)
    elapsed = time.time() - start_time
    
    # Should be less than 10ms
    assert elapsed < 0.01, f"Too slow: {elapsed*1000:.2f}ms"


def test_batch_prediction_response_time(ml_service):
    """Test batch prediction response time."""
    batch_features = [[0.5] * 100 for _ in range(100)]
    
    start_time = time.time()
    ml_service.predict_batch(batch_features)
    elapsed = time.time() - start_time
    
    # 100 predictions should be less than 1 second
    assert elapsed < 1.0, f"Batch too slow: {elapsed:.2f}s"


def test_cached_prediction_faster(ml_service):
    """Test that cached predictions are faster."""
    features = [0.5] * 100
    
    # First call (not cached)
    start_time = time.time()
    ml_service.predict_with_cache(features)
    first_call_time = time.time() - start_time
    
    # Second call (cached)
    start_time = time.time()
    ml_service.predict_with_cache(features)
    second_call_time = time.time() - start_time
    
    # Cached should be at least 2x faster
    assert second_call_time < first_call_time / 2


@pytest.mark.parametrize("n_features", [10, 50, 100, 500])
def test_prediction_scales_with_features(ml_service, n_features):
    """Test that prediction time scales reasonably with feature count."""
    features = [0.5] * n_features
    
    start_time = time.time()
    ml_service.predict(features)
    elapsed = time.time() - start_time
    
    # Should still be fast even with many features
    assert elapsed < 0.1, f"Too slow with {n_features} features: {elapsed*1000:.2f}ms"


def demo_response_time_testing():
    """Demo response time testing."""
    print("\n" + "=" * 70)
    print("2. Response Time Testing")
    print("=" * 70)
    print()
    
    print("⏱️  RESPONSE TIME TARGETS:")
    print()
    print("   Operation              Target      Max")
    print("   ─────────────────────────────────────")
    print("   Single prediction      < 10ms      50ms")
    print("   Batch (100)            < 100ms     1s")
    print("   Cached prediction      < 1ms       5ms")
    print("   Model loading          < 1s        5s")
    print()
    
    print("📊 MEASURING RESPONSE TIME:")
    print()
    print("   start_time = time.time()")
    print("   result = ml_service.predict(features)")
    print("   elapsed = time.time() - start_time")
    print("   ")
    print("   assert elapsed < 0.01  # 10ms")
    print()


# ============================================================================
# 4. Throughput Testing
# ============================================================================

def test_throughput_sequential(ml_service):
    """Test throughput with sequential requests."""
    features = [0.5] * 100
    n_requests = 1000
    
    start_time = time.time()
    for _ in range(n_requests):
        ml_service.predict(features)
    elapsed = time.time() - start_time
    
    throughput = n_requests / elapsed
    
    # Should handle at least 100 requests per second
    assert throughput >= 100, f"Low throughput: {throughput:.0f} req/s"
    
    print(f"Throughput: {throughput:.0f} requests/second")


def test_throughput_concurrent(ml_service):
    """Test throughput with concurrent requests."""
    features = [0.5] * 100
    n_requests = 1000
    n_workers = 10
    
    def make_request():
        return ml_service.predict(features)
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(make_request) for _ in range(n_requests)]
        results = [f.result() for f in futures]
    elapsed = time.time() - start_time
    
    throughput = n_requests / elapsed
    
    # Concurrent should be faster than sequential
    print(f"Concurrent throughput: {throughput:.0f} requests/second")
    
    # Should handle at least 500 requests per second with concurrency
    assert throughput >= 500, f"Low concurrent throughput: {throughput:.0f} req/s"


def demo_throughput_testing():
    """Demo throughput testing."""
    print("\n" + "=" * 70)
    print("3. Throughput Testing")
    print("=" * 70)
    print()
    
    print("📈 THROUGHPUT METRICS:")
    print()
    print("   Sequential:")
    print("   • Single thread")
    print("   • Baseline performance")
    print("   • Target: > 100 req/s")
    print()
    print("   Concurrent:")
    print("   • Multiple threads/processes")
    print("   • Real-world load")
    print("   • Target: > 500 req/s")
    print()
    
    print("💡 THROUGHPUT FORMULA:")
    print()
    print("   throughput = n_requests / elapsed_time")
    print("   ")
    print("   Example:")
    print("   • 1000 requests in 2 seconds")
    print("   • throughput = 1000 / 2 = 500 req/s")
    print()


# ============================================================================
# 5. Memory Usage Testing
# ============================================================================

def test_memory_usage_single_prediction(ml_service):
    """Test memory usage for single prediction."""
    process = psutil.Process(os.getpid())
    
    # Measure before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Make predictions
    features = [0.5] * 100
    for _ in range(1000):
        ml_service.predict(features)
    
    # Measure after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    mem_increase = mem_after - mem_before
    
    # Memory shouldn't increase by more than 50MB
    assert mem_increase < 50, f"Memory leak: +{mem_increase:.1f}MB"
    
    print(f"Memory increase: {mem_increase:.1f}MB")


def test_memory_usage_batch_prediction(ml_service):
    """Test memory usage for batch prediction."""
    process = psutil.Process(os.getpid())
    
    # Measure before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Make batch predictions
    batch_features = [[0.5] * 100 for _ in range(1000)]
    ml_service.predict_batch(batch_features)
    
    # Measure after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    mem_increase = mem_after - mem_before
    
    # Memory shouldn't increase by more than 100MB
    assert mem_increase < 100, f"Memory leak: +{mem_increase:.1f}MB"
    
    print(f"Batch memory increase: {mem_increase:.1f}MB")


def demo_memory_testing():
    """Demo memory usage testing."""
    print("\n" + "=" * 70)
    print("4. Memory Usage Testing")
    print("=" * 70)
    print()
    
    print("💾 MEMORY TESTING:")
    print()
    print("   import psutil")
    print("   process = psutil.Process(os.getpid())")
    print("   ")
    print("   mem_before = process.memory_info().rss")
    print("   # run predictions...")
    print("   mem_after = process.memory_info().rss")
    print("   ")
    print("   mem_increase = mem_after - mem_before")
    print("   assert mem_increase < threshold")
    print()
    
    print("🚨 MEMORY LEAK SIGNS:")
    print()
    print("   • Memory keeps growing")
    print("   • Never gets released")
    print("   • Eventually crashes")
    print()
    print("   Common causes:")
    print("   • Not releasing tensors")
    print("   • Unbounded caches")
    print("   • Circular references")
    print("   • Not closing connections")
    print()


# ============================================================================
# 6. Stress Testing
# ============================================================================

@pytest.mark.slow
def test_stress_many_requests(ml_service):
    """Stress test with many requests."""
    features = [0.5] * 100
    n_requests = 10000
    
    start_time = time.time()
    
    # Make many requests
    for i in range(n_requests):
        try:
            ml_service.predict(features)
        except Exception as e:
            pytest.fail(f"Failed at request {i}: {e}")
    
    elapsed = time.time() - start_time
    throughput = n_requests / elapsed
    
    print(f"Stress test: {n_requests} requests in {elapsed:.1f}s ({throughput:.0f} req/s)")
    
    # Should complete without errors
    assert True


@pytest.mark.slow
def test_stress_large_batch(ml_service):
    """Stress test with large batch."""
    # Very large batch
    large_batch = [[0.5] * 100 for _ in range(10000)]
    
    start_time = time.time()
    results = ml_service.predict_batch(large_batch)
    elapsed = time.time() - start_time
    
    print(f"Large batch: 10000 predictions in {elapsed:.1f}s")
    
    # Should handle large batch
    assert len(results) == 10000
    assert elapsed < 30  # Should complete in reasonable time


def demo_stress_testing():
    """Demo stress testing."""
    print("\n" + "=" * 70)
    print("5. Stress Testing")
    print("=" * 70)
    print()
    
    print("🔥 STRESS TEST GOALS:")
    print()
    print("   • Find breaking point")
    print("   • Test under extreme load")
    print("   • Verify graceful degradation")
    print("   • Identify resource limits")
    print()
    
    print("📊 STRESS TEST SCENARIOS:")
    print()
    print("   1. Many Requests:")
    print("      • 10,000+ sequential requests")
    print("      • Should not crash")
    print("   ")
    print("   2. Large Batches:")
    print("      • 10,000+ items in one batch")
    print("      • Should handle gracefully")
    print("   ")
    print("   3. Concurrent Load:")
    print("      • 100+ concurrent users")
    print("      • Should maintain performance")
    print("   ")
    print("   4. Extended Duration:")
    print("      • Run for hours/days")
    print("      • Check for memory leaks")
    print()


# ============================================================================
# 7. Latency Percentiles
# ============================================================================

def test_latency_percentiles(ml_service):
    """Test latency percentiles (P50, P95, P99)."""
    features = [0.5] * 100
    n_requests = 1000
    latencies = []
    
    # Measure latency for each request
    for _ in range(n_requests):
        start_time = time.time()
        ml_service.predict(features)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        latencies.append(elapsed)
    
    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Latency percentiles:")
    print(f"  P50: {p50:.2f}ms")
    print(f"  P95: {p95:.2f}ms")
    print(f"  P99: {p99:.2f}ms")
    
    # Targets
    assert p50 < 10, f"P50 too high: {p50:.2f}ms"
    assert p95 < 50, f"P95 too high: {p95:.2f}ms"
    assert p99 < 100, f"P99 too high: {p99:.2f}ms"


def demo_latency_percentiles():
    """Demo latency percentiles."""
    print("\n" + "=" * 70)
    print("6. Latency Percentiles")
    print("=" * 70)
    print()
    
    print("📊 UNDERSTANDING PERCENTILES:")
    print()
    print("   P50 (median):")
    print("   • 50% of requests faster")
    print("   • Typical user experience")
    print()
    print("   P95:")
    print("   • 95% of requests faster")
    print("   • Most users' worst case")
    print()
    print("   P99:")
    print("   • 99% of requests faster")
    print("   • Worst case scenario")
    print()
    
    print("🎯 LATENCY TARGETS:")
    print()
    print("   ML API:")
    print("   • P50: < 10ms")
    print("   • P95: < 50ms")
    print("   • P99: < 100ms")
    print()
    print("   Why P99 matters:")
    print("   • 1000 requests/sec")
    print("   • 1% = 10 requests/sec with bad latency")
    print("   • = 36,000 bad experiences per hour!")
    print()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Performance and Load Testing\n")
    
    demo_why_performance_testing()
    demo_response_time_testing()
    demo_throughput_testing()
    demo_memory_testing()
    demo_stress_testing()
    demo_latency_percentiles()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Performance Testing:
   - Prevent production disasters
   - Meet user expectations (< 1s)
   - Optimize costs (right-size infrastructure)
   - ROI: Prevent $50k+ outages

2. Response Time:
   ✅ Single prediction < 10ms
   ✅ Batch (100) < 100ms
   ✅ Cached < 1ms
   ✅ Model loading < 1s

3. Throughput:
   ✅ Sequential: > 100 req/s
   ✅ Concurrent: > 500 req/s
   ✅ Test with ThreadPoolExecutor

4. Memory:
   ✅ Monitor with psutil
   ✅ Check for memory leaks
   ✅ Memory increase < 50MB for 1000 predictions

5. Stress Testing:
   ✅ 10,000+ requests
   ✅ Large batches (10,000 items)
   ✅ Concurrent users (100+)
   ✅ Extended duration (hours)

6. Latency Percentiles:
   ✅ P50 < 10ms
   ✅ P95 < 50ms
   ✅ P99 < 100ms

Performance Testing Checklist:
```
Response Time:
□ Single prediction < 10ms
□ Batch prediction < 100ms
□ Cached predictions faster
□ Scales with input size

Throughput:
□ Sequential > 100 req/s
□ Concurrent > 500 req/s
□ Measure actual throughput

Memory:
□ No memory leaks
□ Memory increase reasonable
□ Resources released

Stress Tests:
□ Handle 10,000+ requests
□ Handle large batches
□ Handle concurrent load
□ No crashes under stress

Latency:
□ P50 < 10ms
□ P95 < 50ms
□ P99 < 100ms
```

Performance Testing Tools:
```python
# Response time
start = time.time()
result = service.predict(features)
elapsed = time.time() - start
assert elapsed < 0.01  # 10ms

# Throughput
throughput = n_requests / elapsed

# Memory
import psutil
process = psutil.Process(os.getpid())
mem = process.memory_info().rss / 1024 / 1024  # MB

# Latency percentiles
import numpy as np
p95 = np.percentile(latencies, 95)

# Concurrent testing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(predict) for _ in range(1000)]
```

Performance Targets:
```
User Experience:
< 100ms: Instant
< 1s: Fast  
< 3s: Acceptable
> 3s: Slow (users leave)

Business Impact:
Amazon: 100ms delay = 1% sales loss
Google: 500ms delay = 20% traffic loss
Your API: Slow = Users leave
```

Next Steps:
✅ Module 11 Complete!
→ Module 12: Complete Production Projects
""")


if __name__ == "__main__":
    main()
