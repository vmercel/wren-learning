# Lesson 5: Production ML Systems — Deployment, Monitoring, and Continuous Learning Pipelines

> This capstone lesson bridges the gap between trained models and production-grade ML systems. We cover model serving architectures, feature stores, drift detection, A/B testing frameworks, and continuous retraining pipelines — the engineering that determines whether your algorithm actually delivers value at scale.

*Lesson 5 of 5*

## From Model to System: The Production ML Gap

Lessons 1–4 covered algorithm internals, ensemble methods, neural architectures, and hyperparameter optimization. But a trained model is only ~10% of a production ML system. The remaining 90% involves:

- **Feature engineering & storage** — Feature stores ensuring training-serving consistency
- **Model serving** — Latency-constrained inference at scale (batch vs. real-time)
- **Monitoring** — Data drift, concept drift, and performance degradation detection
- **Continuous learning** — Automated retraining triggers and safe rollout
- **Governance** — Reproducibility, audit trails, and rollback capabilities

### Why Models Fail in Production

| Failure Mode | Root Cause | Lesson Connection |
|---|---|---|
| Training-serving skew | Feature computation differs between training and inference | Lesson 2 (feature importance) |
| Silent model degradation | Input distribution shifts post-deployment | Lesson 3 (generalization) |
| Latency violations | Model too complex for serving SLA | Lesson 4 (model complexity tradeoffs) |
| Cascading failures | Upstream data pipeline breaks | All lessons |

The key insight: **production ML is a systems problem**, not just an algorithms problem.

## Model Serving Architectures and Latency Optimization

### Serving Patterns

1. **Batch prediction** — Pre-compute predictions on a schedule (e.g., nightly recommendation scores). Simple but stale.
2. **Real-time serving** — Synchronous inference per request. Requires strict latency budgets (p99 < 50ms typical).
3. **Near-real-time (streaming)** — Predictions triggered by events via Kafka/Flink. Balances freshness and throughput.
4. **Embedded/edge** — Model compiled into device firmware (ONNX Runtime, TFLite). No network round-trip.

### Latency Optimization Techniques

- **Model distillation** — Train a smaller student model to mimic a large teacher (from Lesson 3's neural architectures)
- **Quantization** — FP32 → INT8 reduces memory bandwidth 4x with typically <1% accuracy loss
- **Pruning** — Remove near-zero weights; structured pruning removes entire channels
- **Operator fusion** — Combine sequential ops (Conv → BN → ReLU) into single kernels
- **Caching** — Cache predictions for repeated or similar inputs (approximate nearest neighbor lookup)
- **Feature pre-computation** — Materialize expensive features in a feature store rather than computing at serving time

### Critical Edge Case: Cold Start

When a new user/item has no history, your model receives missing or default features. Strategies:
- Fallback to a popularity-based or rule-based model
- Use a separate cold-start model trained on demographic/metadata features
- Implement a multi-armed bandit for exploration during cold start

## Production Model Serving with FastAPI, Feature Store Simulation, and Shadow Mode

```python
import numpy as np
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from collections import deque
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
import joblib
import json

# =============================================================
# 1. FEATURE STORE — Ensures training-serving consistency
# =============================================================
class FeatureStore:
    """Simulates an online feature store (e.g., Feast, Tecton).
    Key invariant: features served online MUST match training features."""
    
    def __init__(self):
        self._online_store: Dict[str, Dict[str, float]] = {}
        self._feature_schema: Dict[str, type] = {}
    
    def register_features(self, schema: Dict[str, type]):
        self._feature_schema = schema
    
    def materialize(self, entity_id: str, features: Dict[str, float]):
        # Validate schema match — catches training-serving skew early
        for key, val in features.items():
            if key not in self._feature_schema:
                raise ValueError(f"Unknown feature '{key}' — not in registered schema")
            if not isinstance(val, (int, float)):
                raise TypeError(f"Feature '{key}' must be numeric, got {type(val)}")
        self._online_store[entity_id] = features
    
    def get_features(self, entity_id: str) -> Optional[Dict[str, float]]:
        return self._online_store.get(entity_id)


# =============================================================
# 2. DRIFT DETECTOR — Population Stability Index (PSI)
# =============================================================
class DriftDetector:
    """Detects data drift using Population Stability Index.
    PSI < 0.1: no drift, 0.1-0.25: moderate, > 0.25: significant."""
    
    def __init__(self, reference_data: np.ndarray, n_bins: int = 10):
        self.n_bins = n_bins
        # Compute reference distribution
        self.bin_edges = np.percentile(
            reference_data, np.linspace(0, 100, n_bins + 1)
        )
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        ref_counts = np.histogram(reference_data, bins=self.bin_edges)[0]
        self.ref_proportions = ref_counts / ref_counts.sum()
        self.ref_proportions = np.clip(self.ref_proportions, 1e-6, None)  # avoid log(0)
    
    def compute_psi(self, current_data: np.ndarray) -> float:
        cur_counts = np.histogram(current_data, bins=self.bin_edges)[0]
        cur_proportions = cur_counts / cur_counts.sum()
        cur_proportions = np.clip(cur_proportions, 1e-6, None)
        
        psi = np.sum(
            (cur_proportions - self.ref_proportions) *
            np.log(cur_proportions / self.ref_proportions)
        )
        return float(psi)
    
    def check(self, current_data: np.ndarray) -> Dict:
        psi = self.compute_psi(current_data)
        if psi > 0.25:
            level = "CRITICAL"
        elif psi > 0.1:
            level = "WARNING"
        else:
            level = "OK"
        return {"psi": round(psi, 4), "level": level}


# =============================================================
# 3. MODEL REGISTRY — Versioning + A/B + Shadow Mode
# =============================================================
@dataclass
class ModelVersion:
    version: str
    model: object
    metrics: Dict[str, float]
    created_at: float = field(default_factory=time.time)

class ModelRegistry:
    """Manages model versions with canary/shadow deployment support."""
    
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.production_version: Optional[str] = None
        self.shadow_version: Optional[str] = None
        self.canary_percentage: float = 0.0  # 0-1, fraction routed to shadow
    
    def register(self, version: str, model, metrics: Dict[str, float]):
        self.versions[version] = ModelVersion(version, model, metrics)
        print(f"Registered model v{version} | metrics: {metrics}")
    
    def promote_to_production(self, version: str):
        if version not in self.versions:
            raise KeyError(f"Version {version} not found")
        self.production_version = version
        print(f"Promoted v{version} to production")
    
    def set_shadow(self, version: str, canary_pct: float = 0.0):
        """Shadow mode: run new model in parallel, log predictions, don't serve them.
        Canary mode: route canary_pct of traffic to the new model."""
        self.shadow_version = version
        self.canary_percentage = canary_pct
    
    def get_serving_model(self, request_id: str) -> tuple:
        """Returns (model, version, is_canary). Uses request_id for deterministic routing."""
        if self.canary_percentage > 0 and self.shadow_version:
            # Deterministic routing based on request hash (reproducible)
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            if (hash_val % 1000) / 1000 < self.canary_percentage:
                v = self.shadow_version
                return self.versions[v].model, v, True
        
        v = self.production_version
        return self.versions[v].model, v, False


# =============================================================
# 4. PREDICTION SERVICE — Ties it all together
# =============================================================
class PredictionService:
    def __init__(self, feature_store: FeatureStore, registry: ModelRegistry,
                 drift_detector: DriftDetector):
        self.feature_store = feature_store
        self.registry = registry
        self.drift_detector = drift_detector
        self.prediction_log: List[Dict] = []
        self.recent_features = deque(maxlen=500)  # rolling window for drift
    
    def predict(self, entity_id: str, request_id: str) -> Dict:
        start = time.perf_counter()
        
        # 1. Fetch features from feature store
        features = self.feature_store.get_features(entity_id)
        if features is None:
            return {"error": "entity_not_found", "fallback": "cold_start_model"}
        
        feature_array = np.array(list(features.values())).reshape(1, -1)
        self.recent_features.append(feature_array.flatten())
        
        # 2. Get model (handles canary routing)
        model, version, is_canary = self.registry.get_serving_model(request_id)
        
        # 3. Predict
        prediction = model.predict_proba(feature_array)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])
        
        # 4. Shadow mode: also run shadow model, log but don't serve
        shadow_pred = None
        if self.registry.shadow_version and not is_canary:
            shadow_model = self.registry.versions[self.registry.shadow_version].model
            shadow_prediction = shadow_model.predict_proba(feature_array)[0]
            shadow_pred = {
                "class": int(np.argmax(shadow_prediction)),
                "confidence": float(shadow_prediction[np.argmax(shadow_prediction)])
            }
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # 5. Log everything (critical for monitoring)
        log_entry = {
            "request_id": request_id,
            "entity_id": entity_id,
            "model_version": version,
            "is_canary": is_canary,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "shadow_prediction": shadow_pred,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time()
        }
        self.prediction_log.append(log_entry)
        
        return {
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "model_version": version,
            "latency_ms": round(latency_ms, 2)
        }
    
    def run_drift_check(self) -> Dict:
        if len(self.recent_features) < 50:
            return {"status": "insufficient_data"}
        # Check drift on each feature dimension
        recent = np.array(list(self.recent_features))
        results = {}
        for i in range(recent.shape[1]):
            results[f"feature_{i}"] = self.drift_detector.compute_psi(recent[:, i])
        max_psi = max(results.values())
        return {
            "per_feature_psi": {k: round(v, 4) for k, v in results.items()},
            "max_psi": round(max_psi, 4),
            "alert": max_psi > 0.25
        }


# =============================================================
# 5. END-TO-END DEMO
# =============================================================
if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=5,
        n_redundant=1, random_state=42
    )
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]
    
    # Train two model versions
    model_v1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model_v1.fit(X_train, y_train)
    acc_v1 = model_v1.score(X_test, y_test)
    
    model_v2 = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model_v2.fit(X_train, y_train)
    acc_v2 = model_v2.score(X_test, y_test)
    
    # Setup feature store
    fs = FeatureStore()
    fs.register_features({f"f{i}": float for i in range(8)})
    for idx in range(len(X_test)):
        features = {f"f{i}": float(X_test[idx, i]) for i in range(8)}
        fs.materialize(f"user_{idx}", features)
    
    # Setup drift detector (reference = training distribution)
    drift = DriftDetector(X_train[:, 0])  # monitoring feature 0
    
    # Setup model registry with shadow deployment
    registry = ModelRegistry()
    registry.register("1.0", model_v1, {"accuracy": round(acc_v1, 4)})
    registry.register("2.0", model_v2, {"accuracy": round(acc_v2, 4)})
    registry.promote_to_production("1.0")
    registry.set_shadow("2.0", canary_pct=0.1)  # 10% canary traffic
    
    # Run prediction service
    service = PredictionService(fs, registry, drift)
    
    print("\n--- Serving Predictions ---")
    canary_count = 0
    for i in range(100):
        result = service.predict(f"user_{i}", f"req_{i}")
        if i < 3:
            print(f"  Request {i}: {result}")
        if service.prediction_log[-1]["is_canary"]:
            canary_count += 1
    
    print(f"\nCanary traffic: {canary_count}/100 requests ({canary_count}%)")
    
    # Shadow comparison
    shadow_matches = sum(
        1 for log in service.prediction_log
        if log["shadow_prediction"] and
           log["shadow_prediction"]["class"] == log["predicted_class"]
    )
    shadow_total = sum(1 for log in service.prediction_log if log["shadow_prediction"])
    print(f"Shadow agreement: {shadow_matches}/{shadow_total} ({shadow_matches/max(shadow_total,1)*100:.1f}%)")
    
    # Drift check
    drift_result = service.run_drift_check()
    print(f"\nDrift check: max_psi={drift_result['max_psi']}, alert={drift_result['alert']}")
    
    # Simulate drift: shift feature distributions
    print("\n--- Simulating Data Drift ---")
    for i in range(200):
        drifted_features = {f"f{j}": float(X_test[i % len(X_test), j] + 3.0) for j in range(8)}
        fs.materialize(f"drifted_user_{i}", drifted_features)
        service.predict(f"drifted_user_{i}", f"drift_req_{i}")
    
    drift_result = service.run_drift_check()
    print(f"Drift check after shift: max_psi={drift_result['max_psi']}, alert={drift_result['alert']}")
    
    # Latency summary
    latencies = [log["latency_ms"] for log in service.prediction_log]
    print(f"\nLatency — p50: {np.percentile(latencies, 50):.2f}ms, "
          f"p95: {np.percentile(latencies, 95):.2f}ms, "
          f"p99: {np.percentile(latencies, 99):.2f}ms")
```

## Continuous Learning Pipelines and Automated Retraining

### When to Retrain

Retraining isn't free — it consumes compute, risks regressions, and requires validation. Use these signals:

1. **Data drift detected** — PSI > 0.25 on key features (automated trigger)
2. **Performance degradation** — Online metrics (click-through rate, conversion) drop below threshold
3. **Scheduled cadence** — Weekly/monthly retraining for slowly drifting domains
4. **Label availability** — When ground truth labels arrive (often delayed by hours/days)

### Safe Retraining Pipeline

```
[New Data] → [Feature Engineering] → [Train Candidate]
     ↓
[Offline Evaluation] — Accuracy, calibration, fairness metrics
     ↓  (passes gates?)
[Shadow Deployment] — Run in parallel with production model
     ↓  (shadow metrics acceptable?)
[Canary Rollout] — 5% → 25% → 50% → 100% traffic
     ↓  (online metrics stable?)
[Full Production] — Update model registry, archive old version
```

### Critical Safeguards

- **Champion-challenger framework**: New model must beat current production model on held-out data AND online metrics before full promotion
- **Automatic rollback**: If p99 latency exceeds SLA or error rate spikes, instantly revert to previous version
- **Data validation**: Check for schema changes, null rates, cardinality shifts before training begins
- **Reproducibility**: Pin all random seeds, library versions, and data snapshots. Store the full training config alongside the model artifact

### Feature Store Anti-Patterns

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Point-in-time leakage | Training uses future features | Use `as_of` timestamp joins |
| Online/offline skew | Different feature code paths | Single feature definition, dual materialization |
| Unbounded feature freshness | Stale cache served as current | TTL on feature values, freshness monitoring |
| Missing value inconsistency | Training imputes, serving doesn't | Centralize imputation in feature transform |

## Production Observability: Beyond Accuracy

### The Four Pillars of ML Monitoring

**1. Data Quality Monitoring**
- Schema validation (new categories, type changes)
- Statistical bounds (min, max, mean, variance per feature)
- Null rates and cardinality tracking
- Upstream data freshness (is the ETL pipeline still running?)

**2. Model Performance Monitoring**
- **Proxy metrics** (when ground truth is delayed): prediction distribution stability, confidence calibration
- **Delayed evaluation**: Join predictions with eventual outcomes (e.g., did the user convert 7 days later?)
- **Segment-level analysis**: Overall accuracy may be stable while a critical segment degrades

**3. System Performance Monitoring**
- Latency percentiles (p50, p95, p99) — not just mean!
- Throughput (requests/second)
- Error rates (model errors vs. infrastructure errors)
- Memory and CPU utilization of model servers
- Feature store read latency

**4. Business Impact Monitoring**
- Revenue/conversion lift from model vs. baseline
- A/B test statistical significance
- Cost per prediction (GPU/CPU hours)
- Alert fatigue metrics (are monitoring alerts actionable?)

### Alerting Strategy

```
SEVERITY 1 (page on-call): Model serving errors > 1%, latency p99 > 500ms
SEVERITY 2 (Slack alert):  PSI > 0.25 on any feature, prediction distribution shift
SEVERITY 3 (daily digest):  PSI 0.1-0.25, slight accuracy degradation in segments
SEVERITY 4 (weekly review): Model staleness > configured threshold
```

The goal: **catch silent failures before they impact users**. A model that returns wrong predictions with high confidence is worse than one that throws an error.

## Key Takeaways

- A trained model is only ~10% of a production ML system. Feature stores, serving infrastructure, monitoring, and retraining pipelines are equally critical and far more likely to cause production incidents.
- Always deploy new models through a shadow → canary → full production pipeline. Shadow mode catches prediction discrepancies without user impact; canary mode validates online metrics on a small traffic slice before full rollout.
- Monitor data drift (PSI/KL divergence on inputs), concept drift (performance on delayed labels), and system health (latency percentiles, error rates) independently — each can fail without the others signaling.
- Training-serving skew is the most common and insidious production ML bug. Use a feature store with a single feature definition that materializes to both offline (training) and online (serving) stores.
- Build automated rollback into your deployment pipeline. If any monitored metric breaches its threshold within the canary window, the system should revert to the previous model version without human intervention.

## Exercises

### Build a Complete Retraining Trigger System *(hard)*

Extend the DriftDetector and PredictionService classes to implement an automated retraining pipeline. Requirements: (1) Track PSI across all features over a sliding window of the last 1000 predictions. (2) When any feature's PSI exceeds 0.25, trigger a simulated retraining that fits a new model on recent data plus original training data. (3) Implement a champion-challenger evaluation: the new model must achieve accuracy >= production model accuracy - 0.01 on a holdout set before being promoted. (4) Log all retraining decisions (triggered/skipped, old vs new accuracy, promoted/rejected) with timestamps. Test by gradually introducing covariate shift to your synthetic data.

### Latency Optimization Benchmark *(hard)*

Take the GradientBoostingClassifier from the example and benchmark the following optimization path: (1) Measure baseline inference latency (p50, p95, p99 over 10,000 predictions). (2) Train a simpler model (LogisticRegression or small RandomForest with max_depth=3) as a distilled model using the GBM's predicted probabilities as soft labels. (3) Measure the distilled model's latency and accuracy vs. the original. (4) Implement a prediction cache using an LRU dict — for features within L2 distance < 0.1 of a cached input, return the cached prediction. Measure cache hit rate and latency improvement. (5) Report a table: model variant, accuracy, p50/p95/p99 latency, and throughput (predictions/second).

---

**Next up:** You have completed the full ML Algorithms curriculum — from foundational supervised/unsupervised internals through ensemble methods, deep learning architectures, hyperparameter optimization, and now production deployment. Review all five lessons and apply the end-to-end pipeline to a real dataset of your choice.