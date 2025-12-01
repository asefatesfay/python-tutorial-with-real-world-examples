"""
Production ML Monitoring

Learn how to monitor ML models in production.
Focus: Track performance, detect issues, ensure reliability.

This file covers monitoring strategies without external dependencies.
Real monitoring uses: Prometheus, Grafana, Datadog, etc.
"""

import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import math


# ============================================================================
# 1. Why Monitor ML Models?
# ============================================================================

def demo_why_monitoring():
    """
    Why monitoring is critical for production ML.
    
    INTUITION - The Self-Driving Car Analogy:
    
    Training Model = Teaching someone to drive
    - Practice in parking lot (training data)
    - Pass driving test (validation)
    - Get license (deploy model)
    
    But roads change:
    - New construction (data distribution shift)
    - Bad weather (edge cases)
    - New traffic patterns (concept drift)
    - Car issues (model degradation)
    
    Without monitoring:
    - Don't notice problems
    - Crash happens! 💥
    
    With monitoring:
    - Dashboard shows issues
    - Alert before crash
    - Safe driving! ✅
    
    WHAT CAN GO WRONG IN PRODUCTION:
    
    1. Data Drift:
       Training: Users aged 25-45
       Production: Users aged 18-65
       
       Result: Model performs poorly on 18-25 and 45-65
    
    2. Concept Drift:
       Training: Pre-COVID shopping patterns
       Production: Post-COVID online shopping surge
       
       Result: Model predictions are wrong
    
    3. Model Degradation:
       Month 1: 95% accuracy
       Month 3: 90% accuracy
       Month 6: 80% accuracy
       
       Result: Gradually getting worse (nobody noticed!)
    
    4. Infrastructure Issues:
       - Slow predictions (>1 sec)
       - Out of memory
       - API timeouts
       - Service crashes
    
    5. Adversarial Inputs:
       - Users gaming the system
       - Spam / fraud attempts
       - Edge cases
    
    WHAT TO MONITOR:
    
    1. Model Performance:
       - Accuracy, precision, recall (if you have labels)
       - Prediction distribution (detect drift)
       - Confidence scores (low confidence = problem)
    
    2. System Performance:
       - Latency (p50, p95, p99)
       - Throughput (requests/sec)
       - Error rate (5xx errors)
       - Resource usage (CPU, memory, GPU)
    
    3. Input Data:
       - Feature distributions (compare to training)
       - Missing values (increased?)
       - Outliers (unusual values)
       - Data quality scores
    
    4. Business Metrics:
       - User engagement (clicks, conversions)
       - Revenue impact
       - User satisfaction (support tickets)
    
    MONITORING LEVELS:
    
    Level 1: Basic (Minimum viable monitoring)
    - Request count, error rate, latency
    - Simple alerts (5xx > 1%)
    
    Level 2: Standard (Production ready)
    - + Prediction distribution
    - + Feature statistics
    - + Confidence tracking
    - + Data quality checks
    
    Level 3: Advanced (Large scale)
    - + Model drift detection
    - + A/B testing metrics
    - + Explainability tracking
    - + Automated retraining triggers
    
    Level 4: Enterprise (Mission critical)
    - + Real-time anomaly detection
    - + Multi-model comparison
    - + Causal analysis
    - + Regulatory compliance
    
    MONITORING TOOLS:
    
    Open Source:
    - Prometheus: Metrics collection
    - Grafana: Dashboards
    - ELK Stack: Log aggregation
    - MLflow: ML experiment tracking
    
    Commercial:
    - Datadog: All-in-one monitoring
    - New Relic: APM + ML monitoring
    - Arize AI: ML observability
    - Evidently AI: ML monitoring
    
    REAL-WORLD IMPACT:
    
    Without Monitoring:
    - Model degraded for 3 months (unnoticed)
    - Lost revenue: $500K
    - Customer complaints: 1000+
    - Team's reaction: "We had no idea!"
    
    With Monitoring:
    - Alert: "Accuracy dropped 5%"
    - Team investigates within 1 hour
    - Fix deployed next day
    - Lost revenue: $5K (100x less)
    - Customers: Happy ✅
    """
    print("=" * 70)
    print("1. Why Monitor ML Models in Production?")
    print("=" * 70)
    print()
    print("💭 INTUITION: The Self-Driving Car Analogy")
    print()
    print("   Training Model = Learning to drive:")
    print("   🚗 Practice in parking lot (training data)")
    print("   📝 Pass driving test (validation)")
    print("   ✅ Get license (deploy to production)")
    print()
    print("   But roads change constantly:")
    print("   🚧 New construction (data distribution shift)")
    print("   🌧️  Bad weather (edge cases)")
    print("   🚦 New traffic patterns (concept drift)")
    print("   🔧 Car maintenance needed (model degradation)")
    print()
    print("   ❌ Without Dashboard (monitoring):")
    print("      • Don't notice engine issues")
    print("      • Don't see road changes")
    print("      • Crash! 💥")
    print()
    print("   ✅ With Dashboard (monitoring):")
    print("      • Warning lights (alerts)")
    print("      • GPS shows construction (data drift)")
    print("      • Speed, fuel, engine status visible")
    print("      • Safe arrival! 🎯")
    print()
    
    print("🚨 WHAT CAN GO WRONG IN PRODUCTION:")
    print()
    print("   1️⃣  Data Drift (Input distribution changed)")
    print()
    print("      Training data:")
    print("      Age: 25-45 (average 35)")
    print("      Income: $50K-$100K")
    print()
    print("      Production data (6 months later):")
    print("      Age: 18-65 (average 40)")
    print("      Income: $30K-$150K")
    print()
    print("      Result: Model performs poorly on new ranges!")
    print("             (Trained on different data)")
    print()
    print("   2️⃣  Concept Drift (Relationships changed)")
    print()
    print("      Pre-COVID (training):")
    print("      Feature: 'Has car' → High loan approval")
    print()
    print("      Post-COVID (production):")
    print("      Feature: 'Has car' → Less relevant")
    print("      (More remote work, less commuting)")
    print()
    print("      Result: Model's learned patterns outdated!")
    print()
    print("   3️⃣  Model Degradation (Performance decays)")
    print()
    print("      Month 1: 95% accuracy ✅")
    print("      Month 3: 90% accuracy ⚠️")
    print("      Month 6: 80% accuracy ❌")
    print("      Month 9: 70% accuracy 💥")
    print()
    print("      Cause: World changed, model stayed same")
    print("      Solution: Retrain regularly!")
    print()
    print("   4️⃣  Infrastructure Issues")
    print()
    print("      • Slow predictions (>1 sec → users leave)")
    print("      • Memory leaks (eventually crashes)")
    print("      • API timeouts (5xx errors)")
    print("      • Service crashes (no predictions!)")
    print()
    print("   5️⃣  Adversarial Inputs (Gaming the system)")
    print()
    print("      Fraud detection model:")
    print("      Fraudsters figure out: 'Add legitimate-looking item'")
    print("      → Bypass detection")
    print()
    print("      Spam filter:")
    print("      Spammers add random words")
    print("      → Looks legitimate")
    print()
    
    print("📊 WHAT TO MONITOR:")
    print()
    print("   1. Model Performance Metrics:")
    print()
    print("      With labels (ground truth):")
    print("      • Accuracy: How often correct?")
    print("      • Precision: Of predicted positives, % actually positive")
    print("      • Recall: Of actual positives, % caught")
    print("      • F1-score: Harmonic mean of precision & recall")
    print()
    print("      Without labels:")
    print("      • Prediction distribution (approved vs denied)")
    print("      • Average confidence scores")
    print("      • Confidence score distribution")
    print()
    print("   2. System Performance:")
    print()
    print("      • Latency:")
    print("        - p50 (median): 50ms")
    print("        - p95: 150ms (95% of requests faster)")
    print("        - p99: 300ms (99% faster)")
    print()
    print("      • Throughput: 1000 requests/sec")
    print("      • Error rate: 0.1% (1 error per 1000 requests)")
    print("      • Resource usage: 60% CPU, 2GB memory")
    print()
    print("   3. Input Data Quality:")
    print()
    print("      • Feature distributions:")
    print("        Training: Age mean=35, std=10")
    print("        Production: Age mean=40, std=15")
    print("        → Data drift detected! ⚠️")
    print()
    print("      • Missing values:")
    print("        Week 1: 2% missing")
    print("        Week 5: 15% missing")
    print("        → Data pipeline issue! 🚨")
    print()
    print("      • Outliers:")
    print("        Income: $1M (max in training: $200K)")
    print("        → Unexpected value! ⚠️")
    print()
    print("   4. Business Metrics:")
    print()
    print("      • User engagement: Click-through rate")
    print("      • Revenue impact: Sales influenced by model")
    print("      • User satisfaction: Support tickets")
    print("      • Cost: API costs, infrastructure")
    print()
    
    print("🎯 MONITORING MATURITY LEVELS:")
    print()
    print("   Level 1: Basic (MVP)")
    print("   ────────────────────")
    print("   • Request count")
    print("   • Error rate (5xx)")
    print("   • Latency (p99)")
    print("   • Simple alerts (error rate > 1%)")
    print()
    print("   Time to implement: 1 day")
    print("   Good for: Starting out")
    print()
    print("   Level 2: Standard (Production Ready)")
    print("   ─────────────────────────────────────")
    print("   • Level 1 +")
    print("   • Prediction distribution")
    print("   • Confidence scores")
    print("   • Feature statistics")
    print("   • Data quality checks")
    print()
    print("   Time to implement: 1 week")
    print("   Good for: Most production systems ✅")
    print()
    print("   Level 3: Advanced (Large Scale)")
    print("   ────────────────────────────────")
    print("   • Level 2 +")
    print("   • Model drift detection (automated)")
    print("   • A/B testing metrics")
    print("   • Explainability tracking")
    print("   • Automated retraining triggers")
    print()
    print("   Time to implement: 1 month")
    print("   Good for: High-value models")
    print()
    print("   Level 4: Enterprise (Mission Critical)")
    print("   ───────────────────────────────────────")
    print("   • Level 3 +")
    print("   • Real-time anomaly detection")
    print("   • Multi-model comparison")
    print("   • Causal analysis")
    print("   • Regulatory compliance tracking")
    print()
    print("   Time to implement: 3+ months")
    print("   Good for: Financial, healthcare")
    print()
    
    print("🛠️  MONITORING TOOL LANDSCAPE:")
    print()
    print("   Open Source:")
    print("   • Prometheus: Metrics collection")
    print("   • Grafana: Beautiful dashboards")
    print("   • ELK Stack: Log aggregation & search")
    print("   • MLflow: ML experiment tracking")
    print()
    print("   Commercial (All-in-one):")
    print("   • Datadog: $15-31/host/month")
    print("   • New Relic: $25-99/user/month")
    print("   • Arize AI: ML-specific observability")
    print("   • Evidently AI: ML monitoring & drift")
    print()
    print("   Recommendation:")
    print("   Starting → Prometheus + Grafana (free)")
    print("   Growing → Datadog (easy, powerful)")
    print("   Enterprise → Custom solution + Arize")
    print()
    
    print("💰 REAL-WORLD IMPACT:")
    print()
    print("   Scenario: Loan approval model")
    print()
    print("   ❌ Without Monitoring:")
    print("      • Model degraded over 3 months")
    print("      • Approval rate dropped 15%")
    print("      • Lost revenue: $500,000")
    print("      • Customer complaints: 1,000+")
    print("      • Team: 'We had no idea this was happening!'")
    print("      • Time to fix: 2 weeks (once discovered)")
    print()
    print("   ✅ With Monitoring:")
    print("      • Alert: 'Accuracy dropped 5%' (Week 1)")
    print("      • Team notified: 10 minutes")
    print("      • Investigation: 2 hours")
    print("      • Fix deployed: Next day")
    print("      • Lost revenue: $5,000 (100x less!)")
    print("      • Customers: Happy")
    print()
    print("   ROI of Monitoring:")
    print("   Cost: $500/month (Datadog)")
    print("   Saved: $495,000 (first incident)")
    print("   ROI: 990x 🚀")


# ============================================================================
# 2. Building a Monitoring System
# ============================================================================

class MetricsCollector:
    """
    Collects and tracks metrics for ML models.
    
    In production, use Prometheus or Datadog.
    This is a simplified version for learning.
    """
    
    def __init__(self, name: str = "ml_model"):
        """Initialize metrics collector."""
        self.name = name
        
        # Counters (always increase)
        self.total_requests = 0
        self.total_errors = 0
        self.total_predictions = defaultdict(int)  # prediction -> count
        
        # Histograms (track distributions)
        self.latencies = deque(maxlen=1000)  # Last 1000 latencies
        self.confidence_scores = deque(maxlen=1000)
        
        # Gauges (current values)
        self.current_memory_mb = 0
        self.current_cpu_percent = 0
        
        # Feature statistics
        self.feature_stats = defaultdict(lambda: {
            'values': deque(maxlen=1000),
            'mean': 0,
            'std': 0,
            'min': float('inf'),
            'max': float('-inf')
        })
    
    def record_request(self, prediction: str, confidence: float, 
                      latency_ms: float, features: Dict[str, float]):
        """Record a prediction request."""
        # Update counters
        self.total_requests += 1
        self.total_predictions[prediction] += 1
        
        # Update histograms
        self.latencies.append(latency_ms)
        self.confidence_scores.append(confidence)
        
        # Update feature statistics
        for feature_name, value in features.items():
            stats = self.feature_stats[feature_name]
            stats['values'].append(value)
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            
            # Calculate running mean and std
            values_list = list(stats['values'])
            stats['mean'] = sum(values_list) / len(values_list)
            if len(values_list) > 1:
                variance = sum((x - stats['mean']) ** 2 for x in values_list) / len(values_list)
                stats['std'] = math.sqrt(variance)
    
    def record_error(self):
        """Record an error."""
        self.total_errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        # Calculate percentiles for latency
        sorted_latencies = sorted(self.latencies) if self.latencies else [0]
        n = len(sorted_latencies)
        
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        # Calculate error rate
        error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
        
        # Prediction distribution
        pred_dist = dict(self.total_predictions)
        
        # Feature statistics summary
        feature_summary = {}
        for name, stats in self.feature_stats.items():
            if stats['values']:
                feature_summary[name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                }
        
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate_percent': error_rate,
            'latency_ms': {
                'p50': sorted_latencies[p50_idx],
                'p95': sorted_latencies[p95_idx],
                'p99': sorted_latencies[p99_idx]
            },
            'predictions': pred_dist,
            'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            'features': feature_summary
        }
    
    def check_alerts(self) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        metrics = self.get_metrics()
        
        # Alert: High error rate
        if metrics['error_rate_percent'] > 1.0:
            alerts.append(f"⚠️  HIGH ERROR RATE: {metrics['error_rate_percent']:.2f}% (threshold: 1%)")
        
        # Alert: High latency
        if metrics['latency_ms']['p99'] > 500:
            alerts.append(f"⚠️  HIGH LATENCY: p99={metrics['latency_ms']['p99']:.0f}ms (threshold: 500ms)")
        
        # Alert: Low confidence
        if metrics['avg_confidence'] < 0.7:
            alerts.append(f"⚠️  LOW CONFIDENCE: {metrics['avg_confidence']:.2f} (threshold: 0.7)")
        
        # Alert: Prediction distribution skew
        if metrics['predictions']:
            total_preds = sum(metrics['predictions'].values())
            for pred, count in metrics['predictions'].items():
                ratio = count / total_preds
                if ratio > 0.95:  # >95% of one class
                    alerts.append(f"⚠️  PREDICTION SKEW: {pred} = {ratio*100:.0f}% of predictions")
        
        return alerts


def demo_monitoring_system():
    """
    Demonstrate building a monitoring system.
    """
    print("\n" + "=" * 70)
    print("2. Building a Monitoring System")
    print("=" * 70)
    print()
    
    print("🏗️  SYSTEM COMPONENTS:")
    print()
    print("   1. Metrics Collector")
    print("   ────────────────────")
    print("   • Collects metrics from predictions")
    print("   • Tracks: latency, errors, predictions, features")
    print("   • In production: Prometheus, StatsD")
    print()
    print("   2. Metrics Storage")
    print("   ──────────────────")
    print("   • Time-series database")
    print("   • Stores historical data")
    print("   • In production: Prometheus, InfluxDB, TimescaleDB")
    print()
    print("   3. Dashboards")
    print("   ─────────────")
    print("   • Visualize metrics")
    print("   • Real-time monitoring")
    print("   • In production: Grafana, Datadog")
    print()
    print("   4. Alerting")
    print("   ───────────")
    print("   • Check thresholds")
    print("   • Notify team (email, Slack, PagerDuty)")
    print("   • In production: Prometheus Alertmanager, PagerDuty")
    print()
    
    print("💻 CODE EXAMPLE:")
    print()
    
    # Initialize metrics collector
    metrics = MetricsCollector(name="loan_model")
    
    print("   # Initialize metrics collector")
    print("   metrics = MetricsCollector(name='loan_model')")
    print()
    
    # Simulate predictions
    print("   # Simulate predictions over time...")
    print()
    
    predictions_data = [
        {"prediction": "approved", "confidence": 0.85, "latency_ms": 45, 
         "features": {"age": 35, "income": 75000, "credit_score": 750}},
        {"prediction": "approved", "confidence": 0.92, "latency_ms": 38,
         "features": {"age": 40, "income": 90000, "credit_score": 780}},
        {"prediction": "denied", "confidence": 0.78, "latency_ms": 52,
         "features": {"age": 25, "income": 35000, "credit_score": 620}},
        {"prediction": "approved", "confidence": 0.88, "latency_ms": 41,
         "features": {"age": 45, "income": 100000, "credit_score": 800}},
    ]
    
    for i, pred_data in enumerate(predictions_data, 1):
        metrics.record_request(
            prediction=pred_data["prediction"],
            confidence=pred_data["confidence"],
            latency_ms=pred_data["latency_ms"],
            features=pred_data["features"]
        )
        print(f"   Request {i}: {pred_data['prediction']} (confidence={pred_data['confidence']}, latency={pred_data['latency_ms']}ms)")
    
    print()
    
    # Get metrics
    print("   # Get current metrics")
    current_metrics = metrics.get_metrics()
    
    print(f"   metrics.get_metrics() →")
    print()
    print(f"   Total Requests: {current_metrics['total_requests']}")
    print(f"   Error Rate: {current_metrics['error_rate_percent']:.2f}%")
    print(f"   Latency (p50/p95/p99): {current_metrics['latency_ms']['p50']:.0f}/{current_metrics['latency_ms']['p95']:.0f}/{current_metrics['latency_ms']['p99']:.0f}ms")
    print(f"   Avg Confidence: {current_metrics['avg_confidence']:.2f}")
    print(f"   Predictions: {current_metrics['predictions']}")
    print()
    
    print("   # Feature statistics")
    for feature_name, stats in current_metrics['features'].items():
        print(f"   {feature_name}:")
        print(f"     Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        print(f"     Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
    print()
    
    # Check alerts
    print("   # Check for alerts")
    alerts = metrics.check_alerts()
    if alerts:
        print("   🚨 ALERTS:")
        for alert in alerts:
            print(f"     {alert}")
    else:
        print("   ✅ No alerts (all healthy)")
    print()
    
    print("📊 REAL PROMETHEUS CODE:")
    print()
    print("   ```python")
    print("   from prometheus_client import Counter, Histogram, Gauge")
    print("   ")
    print("   # Define metrics")
    print("   request_count = Counter(")
    print("       'ml_predictions_total',")
    print("       'Total ML predictions',")
    print("       ['model_version', 'prediction']")
    print("   )")
    print("   ")
    print("   latency = Histogram(")
    print("       'ml_prediction_latency_seconds',")
    print("       'Prediction latency',")
    print("       buckets=[0.01, 0.05, 0.1, 0.5, 1.0]")
    print("   )")
    print("   ")
    print("   confidence = Histogram(")
    print("       'ml_prediction_confidence',")
    print("       'Prediction confidence scores',")
    print("       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]")
    print("   )")
    print("   ")
    print("   # Record metrics")
    print("   @app.post('/predict')")
    print("   async def predict(request):")
    print("       start = time.time()")
    print("       ")
    print("       # Make prediction")
    print("       prediction, conf = model.predict(request)")
    print("       ")
    print("       # Record metrics")
    print("       latency.observe(time.time() - start)")
    print("       confidence.observe(conf)")
    print("       request_count.labels(")
    print("           model_version='v1',")
    print("           prediction=prediction")
    print("       ).inc()")
    print("       ")
    print("       return {'prediction': prediction, 'confidence': conf}")
    print("   ```")


# ============================================================================
# 3. Detecting Data Drift
# ============================================================================

def demo_data_drift():
    """
    Detect data drift in production.
    """
    print("\n" + "=" * 70)
    print("3. Detecting Data Drift")
    print("=" * 70)
    print()
    
    print("📊 WHAT IS DATA DRIFT?")
    print()
    print("   Training Data:")
    print("   Age: 25-45 (mean=35, std=5)")
    print("   📊 ▁▃▇█▇▃▁ (bell curve)")
    print()
    print("   Production Data (6 months later):")
    print("   Age: 30-60 (mean=45, std=8)")
    print("   📊 ▁▁▃▇█▇▃▁▁ (shifted right)")
    print()
    print("   Drift detected! ⚠️")
    print("   • Mean changed: 35 → 45 (+10)")
    print("   • Std changed: 5 → 8 (+3)")
    print("   • Distribution shape changed")
    print()
    print("   Impact:")
    print("   Model trained on younger users (25-45)")
    print("   Now seeing older users (30-60)")
    print("   → Predictions less accurate for 50-60 age group")
    print()
    
    print("🔍 DRIFT DETECTION METHODS:")
    print()
    print("   1. Statistical Tests")
    print("   ────────────────────")
    print("   Kolmogorov-Smirnov (KS) Test:")
    print("   • Compares two distributions")
    print("   • Returns p-value (0-1)")
    print("   • p < 0.05 → Distributions different")
    print()
    print("   Example:")
    print("   Training: [30, 35, 40, 35, 38, ...]")
    print("   Production: [45, 50, 55, 48, 52, ...]")
    print("   KS test: p=0.001 → Drift detected! ⚠️")
    print()
    print("   2. Population Stability Index (PSI)")
    print("   ───────────────────────────────────")
    print("   • Measures distribution change")
    print("   • PSI < 0.1: No drift ✅")
    print("   • PSI 0.1-0.2: Small drift ⚠️")
    print("   • PSI > 0.2: Large drift 🚨")
    print()
    print("   Formula:")
    print("   PSI = Σ (actual% - expected%) × ln(actual% / expected%)")
    print()
    print("   3. Comparing Statistics")
    print("   ───────────────────────")
    print("   Simple but effective:")
    print()
    print("   Training mean: 35")
    print("   Production mean: 45")
    print("   Change: +28% → Investigate!")
    print()
    print("   Training std: 5")
    print("   Production std: 8")
    print("   Change: +60% → More variability!")
    print()
    
    print("💻 DRIFT DETECTION CODE:")
    print()
    print("   ```python")
    print("   class DriftDetector:")
    print("       def __init__(self, training_stats):")
    print("           self.training_stats = training_stats")
    print("       ")
    print("       def check_drift(self, production_stats, threshold=0.2):")
    print("           '''Check if drift exceeds threshold'''")
    print("           alerts = []")
    print("           ")
    print("           for feature in self.training_stats:")
    print("               train_mean = self.training_stats[feature]['mean']")
    print("               prod_mean = production_stats[feature]['mean']")
    print("               ")
    print("               # Calculate relative change")
    print("               change = abs(prod_mean - train_mean) / train_mean")
    print("               ")
    print("               if change > threshold:")
    print("                   alerts.append(f'{feature}: {change*100:.0f}% drift')")
    print("           ")
    print("           return alerts")
    print("   ```")
    print()
    
    print("📈 EXAMPLE:")
    print()
    
    # Training statistics
    training_stats = {
        'age': {'mean': 35, 'std': 5},
        'income': {'mean': 60000, 'std': 15000},
        'credit_score': {'mean': 700, 'std': 50}
    }
    
    # Production statistics (after 6 months)
    production_stats = {
        'age': {'mean': 45, 'std': 8},  # Drifted!
        'income': {'mean': 62000, 'std': 16000},  # Slight drift
        'credit_score': {'mean': 705, 'std': 52}  # Stable
    }
    
    print("   Training Stats:")
    for feature, stats in training_stats.items():
        print(f"     {feature}: mean={stats['mean']}, std={stats['std']}")
    print()
    
    print("   Production Stats (6 months later):")
    for feature, stats in production_stats.items():
        print(f"     {feature}: mean={stats['mean']}, std={stats['std']}")
    print()
    
    print("   Drift Analysis:")
    threshold = 0.15  # 15% change triggers alert
    
    for feature in training_stats:
        train_mean = training_stats[feature]['mean']
        prod_mean = production_stats[feature]['mean']
        change = abs(prod_mean - train_mean) / train_mean
        
        status = "🚨 DRIFT" if change > threshold else "✅ OK"
        print(f"     {feature}: {change*100:.1f}% change - {status}")
    
    print()
    
    print("🎯 WHAT TO DO WHEN DRIFT DETECTED:")
    print()
    print("   1. Investigate:")
    print("      • Why did distribution change?")
    print("      • Is it seasonal (temporary)?")
    print("      • Is it permanent (market shift)?")
    print()
    print("   2. Evaluate Impact:")
    print("      • Check model performance on new data")
    print("      • If performance good → Monitor closely")
    print("      • If performance bad → Retrain!")
    print()
    print("   3. Retrain Model:")
    print("      • Include new data")
    print("      • Validate on recent data")
    print("      • A/B test new model vs old")
    print()
    print("   4. Update Baselines:")
    print("      • Update training statistics")
    print("      • Reset drift detection")
    print()
    
    print("⏰ MONITORING FREQUENCY:")
    print()
    print("   Real-time (expensive):")
    print("   • Every prediction")
    print("   • Use sampling (10% of requests)")
    print()
    print("   Batch (recommended):")
    print("   • Daily: Check feature distributions")
    print("   • Weekly: Full drift analysis")
    print("   • Monthly: Retrain if needed")
    print()
    print("   Critical models:")
    print("   • Hourly drift checks")
    print("   • Automated retraining pipelines")


# ============================================================================
# Run All Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n📊 Production ML Monitoring\n")
    print("Learn how to monitor ML models in production!")
    print()
    
    demo_why_monitoring()
    demo_monitoring_system()
    demo_data_drift()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Monitor?
   - Data drift: Input distribution changes
   - Concept drift: Relationships change
   - Model degradation: Performance decays
   - Infrastructure issues: Latency, errors
   - Adversarial inputs: Gaming the system

2. What to Monitor:
   - Model: Accuracy, predictions, confidence
   - System: Latency (p50/p95/p99), errors, throughput
   - Data: Feature distributions, missing values
   - Business: Revenue, user satisfaction

3. Monitoring Levels:
   Level 1 (Basic): Errors, latency, request count
   Level 2 (Standard): + Predictions, confidence, features
   Level 3 (Advanced): + Drift detection, A/B testing
   Level 4 (Enterprise): + Real-time anomaly, compliance

4. Tools:
   Open Source: Prometheus + Grafana (free)
   Commercial: Datadog ($15-31/host/month)
   ML-Specific: Arize AI, Evidently AI

5. Data Drift Detection:
   - Statistical tests (KS test)
   - PSI (Population Stability Index)
   - Compare mean/std (threshold: 15-20%)

Minimal Monitoring Code:
```python
from prometheus_client import Counter, Histogram

# Define metrics
predictions = Counter('predictions_total', 'Total predictions')
latency = Histogram('latency_seconds', 'Prediction latency')

@app.post('/predict')
async def predict(request):
    start = time.time()
    
    # Make prediction
    result = model.predict(request)
    
    # Record metrics
    predictions.inc()
    latency.observe(time.time() - start)
    
    return result
```

Essential Alerts:
✅ Error rate > 1%
✅ Latency p99 > 500ms
✅ Confidence < 0.7
✅ Feature drift > 20%
✅ Prediction skew > 95% one class

Monitoring Checklist:
✅ Prometheus metrics endpoint (/metrics)
✅ Grafana dashboard (latency, errors, predictions)
✅ Alerts (Slack/email/PagerDuty)
✅ Log aggregation (ELK stack)
✅ Feature distribution tracking
✅ Drift detection (weekly)
✅ Model performance tracking (if labels available)
✅ Business metrics (revenue impact)

Resources:
- Prometheus: prometheus.io
- Grafana: grafana.com
- Evidently AI: evidentlyai.com (ML drift detection)
- ML Monitoring Guide: ml-ops.org/content/ml-monitoring

Congratulations! You now know how to:
• Monitor ML models in production
• Detect data drift
• Set up alerts
• Build dashboards
• Ensure model reliability
""")


if __name__ == "__main__":
    main()
