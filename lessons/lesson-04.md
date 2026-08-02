# Lesson 4: Ensemble Methods — Bagging, Boosting, and Stacking in Production

> This lesson dives deep into ensemble methods that combine multiple learners to achieve superior predictive performance. We cover the theory and internals of bagging (Random Forests), boosting (XGBoost, LightGBM, CatBoost), and stacking, with emphasis on hyperparameter tuning, overfitting diagnostics, and production deployment patterns.

*Lesson 4 of 5*

## Why Ensembles Work: Bias-Variance Decomposition Revisited

In Lessons 1–3, we covered individual algorithms—linear models, SVMs, tree-based methods, and neural networks. Each has characteristic failure modes: high-bias models underfit, high-variance models overfit. Ensemble methods exploit a fundamental statistical insight: **combining diverse, imperfect models reduces generalization error**.

### The Math Behind It
For regression with \( M \) models, if each model has error variance \( \sigma^2 \) and pairwise correlation \( \rho \):

`Var(ensemble) = ρσ² + ((1 - ρ) / M) * σ²`

As \( M \to \infty \), variance approaches \( \rho\sigma^2 \). **The key lever is reducing correlation \( \rho \) between base learners**, not just adding more models.

### Three Paradigms
| Strategy | Reduces | Mechanism | Examples |
|----------|---------|-----------|----------|
| **Bagging** | Variance | Bootstrap aggregation + feature subsampling | Random Forest, Extra Trees |
| **Boosting** | Bias (primarily) | Sequential residual fitting | XGBoost, LightGBM, CatBoost |
| **Stacking** | Both | Meta-learner over diverse base models | Blending, Super Learner |

Each paradigm has distinct failure modes, tuning surfaces, and production trade-offs we'll examine in depth.

## Bagging Deep Dive: Random Forests Beyond the Basics

### Internals You Need to Know

**Bootstrap sampling** creates ~63.2% unique samples per tree (due to \( 1 - 1/e \)), leaving ~36.8% as out-of-bag (OOB) samples. OOB error is an unbiased estimate of generalization error—**you get free cross-validation**.

**Feature subsampling** (`max_features`) is the primary decorrelation mechanism:
- Classification default: `sqrt(n_features)`
- Regression default: `n_features / 3`
- Lower values → more decorrelation → lower variance, but potentially higher bias

### Edge Cases and Pitfalls

1. **Extrapolation failure**: RF predictions are bounded by training target range. If your target can grow beyond training values (e.g., stock prices, population), RF will plateau. Consider quantile regression forests or hybrid approaches.

2. **Feature importance bias**: Default impurity-based importance (`feature_importances_`) is biased toward high-cardinality features. Always use **permutation importance** in production.

3. **Correlated features**: If features A and B are correlated, importance gets split between them, underestimating each. Use `feature_importance` with grouped permutation or SHAP.

4. **Memory at scale**: Each tree stores split thresholds and leaf values. A 1000-tree forest on 1M rows with depth 20 can consume 5–15 GB. Use `max_depth`, `min_samples_leaf`, and `max_leaf_nodes` to control size.

## Production Random Forest with OOB Diagnostics

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import joblib
import time

# Load data
X, y = fetch_california_housing(return_X_y=True)
feature_names = fetch_california_housing().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Production-grade RF configuration
rf = RandomForestRegressor(
    n_estimators=500,          # Diminishing returns after ~300-500
    max_depth=20,              # Prevent memory explosion
    min_samples_leaf=5,        # Regularization + smaller model
    max_features=0.33,         # Default for regression
    max_samples=0.8,           # Subsample for speed (not full bootstrap)
    oob_score=True,            # Free validation estimate
    n_jobs=-1,                 # Parallel training
    random_state=42,
    warm_start=False           # Set True to incrementally add trees
)

start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start

print(f"Train time: {train_time:.2f}s")
print(f"OOB R²: {rf.oob_score_:.4f}")
print(f"Test R²: {rf.score(X_test, y_test):.4f}")
print(f"Model size: {len(joblib.dump(rf, '/tmp/rf.joblib', compress=3))} files")

# Permutation importance (preferred over .feature_importances_)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
for name, imp, std in sorted(
    zip(feature_names, perm_imp.importances_mean, perm_imp.importances_std),
    key=lambda x: -x[1]
):
    print(f"  {name:>12}: {imp:.4f} ± {std:.4f}")

# Diagnose: are we using enough trees?
# Plot OOB error vs n_estimators using warm_start
from sklearn.ensemble import RandomForestRegressor as RF
oob_errors = []
rf_ws = RF(n_estimators=1, oob_score=True, max_depth=20,
           min_samples_leaf=5, warm_start=True, n_jobs=-1, random_state=42)
for n in range(1, 301):
    rf_ws.n_estimators = n
    rf_ws.fit(X_train, y_train)
    oob_errors.append(1 - rf_ws.oob_score_)

print(f"\nOOB error at 50 trees:  {oob_errors[49]:.4f}")
print(f"OOB error at 150 trees: {oob_errors[149]:.4f}")
print(f"OOB error at 300 trees: {oob_errors[299]:.4f}")
```

## Gradient Boosting Internals: XGBoost, LightGBM, CatBoost

### How Boosting Differs Fundamentally

Boosting fits models **sequentially**, where each new tree corrects the residuals (pseudo-residuals for general losses) of the ensemble so far. The ensemble prediction is:

`F_M(x) = Σ_{m=1}^{M} η · h_m(x)` where `η` is the learning rate.

### The Three Frameworks Compared

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Tree growth | Level-wise | **Leaf-wise** (faster, risk of overfit) | Oblivious trees (symmetric) |
| Categorical handling | Must encode | Native (optimal split) | **Best native** (ordered target stats) |
| Missing values | Native (learns direction) | Native | Native |
| GPU support | Yes | Yes | **Best GPU** |
| Speed (large data) | Moderate | **Fastest** | Slowest (but often best accuracy) |
| Overfitting resistance | Moderate | Requires careful tuning | **Best** (ordered boosting) |

### Critical Hyperparameters and Their Effects

**Learning rate × n_estimators tradeoff**: Lower learning rate + more trees = better generalization but slower training. Production rule of thumb: start with `lr=0.05, n_estimators=2000, early_stopping_rounds=50`.

**Tree complexity controls**:
- `max_depth`: 4–8 for most problems. CatBoost defaults to 6 (oblivious trees).
- `min_child_weight` (XGBoost) / `min_child_samples` (LightGBM): Higher = more regularization. Critical for imbalanced classes.
- `subsample` + `colsample_bytree`: Stochastic gradient boosting. 0.7–0.9 typical.

**Regularization parameters**:
- `reg_alpha` (L1) + `reg_lambda` (L2): Penalize leaf weights. Start with `lambda=1, alpha=0`.
- `gamma` (XGBoost): Minimum loss reduction for split. Acts as pruning.

### Edge Cases

1. **Ordered boosting (CatBoost)**: Standard boosting has a subtle target leakage—residuals are computed on the same data used to fit trees. CatBoost uses permutation-based ordered boosting to eliminate this, giving ~1-3% accuracy improvement on small datasets.

2. **Leaf-wise growth pitfall**: LightGBM's leaf-wise strategy can create extremely deep, unbalanced trees. Always set `max_depth` or `num_leaves` (recommended: `num_leaves = 2^max_depth * 0.7`).

3. **Monotone constraints**: In production, you often need monotonic relationships (e.g., higher credit score → lower risk). All three frameworks support `monotone_constraints` — use them for model interpretability and regulatory compliance.

## XGBoost and LightGBM: Tuning Pipeline with Optuna

```python
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data preparation
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# --- XGBoost with early stopping ---
def objective_xgb(trial):
    params = {
        'n_estimators': 2000,  # High ceiling; early stopping will cut
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'tree_method': 'hist',  # Fast histogram-based
        'random_state': 42,
        'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

# Run Optuna (50 trials for demonstration)
study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xgb, n_trials=50)

print(f"Best XGBoost RMSE: {study_xgb.best_value:.4f}")
print(f"Best params: {study_xgb.best_params}")

# --- LightGBM with native categorical + callbacks ---
def objective_lgb(trial):
    params = {
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50)

print(f"\nBest LightGBM RMSE: {study_lgb.best_value:.4f}")
print(f"Best params: {study_lgb.best_params}")

# --- Final model: retrain on full train set with best params ---
best_params = study_lgb.best_params
final_model = lgb.LGBMRegressor(
    n_estimators=2000, verbose=-1, n_jobs=-1, random_state=42,
    learning_rate=best_params['lr'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    min_child_samples=best_params['min_child_samples'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)
print(f"\nFinal test RMSE: {mean_squared_error(y_test, final_model.predict(X_test), squared=False):.4f}")
print(f"Trees used: {final_model.best_iteration_}")
```

## Stacking: Meta-Learning for Maximum Performance

### When to Stack

Stacking yields the largest gains when base learners are **diverse and individually strong**. Combining 5 weak models rarely beats 1 tuned XGBoost, but combining XGBoost + LightGBM + CatBoost + Ridge + MLP with a meta-learner often beats any individual model by 1-5%.

### Implementation Details That Matter

**Cross-validated stacking** (mandatory): The meta-learner trains on out-of-fold predictions. If you train on in-fold predictions, the meta-learner simply learns which model overfits best.

```
For each fold k:
  Train base models on folds != k
  Generate predictions on fold k → these become meta-features
Meta-learner trains on all meta-features
```

**Meta-learner selection**:
- **Ridge/Logistic Regression**: Safest default. Learns optimal linear combination.
- **XGBoost as meta-learner**: Can capture non-linear interactions between base model predictions, but overfitting risk increases.
- **Non-negative least squares**: When you want interpretable positive weights.

### Production Considerations

1. **Latency**: Stacking multiplies inference time by the number of base models. For real-time serving, consider pre-computing base model scores or using model distillation.

2. **Maintenance burden**: Each base model is a separate artifact to version, monitor, and retrain. Use MLflow or similar to manage the full stack.

3. **Diminishing returns**: In Kaggle competitions, top teams stack 20+ models. In production, 2-3 diverse models with a linear meta-learner captures 80% of the stacking benefit with manageable complexity.

4. **Feature augmentation stacking**: Instead of only using base model predictions as meta-features, concatenate original features with predictions. This helps the meta-learner learn *when* to trust each base model.

## Production Stacking with scikit-learn and Custom Meta-Features

```python
from sklearn.ensemble import (
    StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# Base learners: diversity is key
estimators = [
    ('rf', RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )),
    ('lgbm', lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        max_depth=8, verbose=-1, n_jobs=-1, random_state=42
    )),
    ('gbr', GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=10, random_state=42
    )),
    ('mlp', Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, random_state=42
        ))
    ])),
]

# Stacking with cross-validated out-of-fold predictions
# passthrough=True adds original features alongside meta-predictions
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(alphas=np.logspace(-4, 4, 20)),
    cv=5,
    passthrough=False,  # Set True for feature augmentation stacking
    n_jobs=-1
)

# Compare individual models vs stack
print("Cross-validated RMSE (5-fold):")
for name, est in estimators:
    scores = cross_val_score(est, X_train, y_train,
                             cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    print(f"  {name:>6}: {-scores.mean():.4f} ± {scores.std():.4f}")

scores_stack = cross_val_score(stacking_reg, X_train, y_train,
                               cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"  {'stack':>6}: {-scores_stack.mean():.4f} ± {scores_stack.std():.4f}")

# Fit and evaluate on test set
stacking_reg.fit(X_train, y_train)
preds = stacking_reg.predict(X_test)
print(f"\nStacking test RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")

# Inspect meta-learner weights (how much each base model contributes)
meta_learner = stacking_reg.final_estimator_
print(f"\nMeta-learner weights (Ridge):")
for (name, _), coef in zip(estimators, meta_learner.coef_):
    print(f"  {name:>6}: {coef:.4f}")
print(f"  Ridge alpha selected: {meta_learner.alpha_:.4f}")
```

## Key Takeaways

- Bagging reduces variance by decorrelating base learners (Random Forest), while boosting reduces bias by sequentially correcting residuals (XGBoost, LightGBM, CatBoost). Understanding which error component dominates guides your algorithm choice.
- Hyperparameter tuning for gradient boosting should always use early stopping with a learning rate × n_estimators tradeoff. Use Optuna or similar Bayesian optimization rather than grid search—the parameter space is too large for exhaustive search.
- LightGBM's leaf-wise growth is fastest but requires careful num_leaves/max_depth constraints. CatBoost's ordered boosting provides the best out-of-the-box accuracy especially with categorical features. XGBoost is the most mature and widely deployed.
- Stacking works best with diverse, individually strong base models and a simple meta-learner (Ridge/Logistic Regression). Always use cross-validated out-of-fold predictions to prevent the meta-learner from learning overfitting patterns.
- In production, prefer 2-3 model stacks over massive ensembles. Account for the latency multiplication and maintenance burden of serving multiple models simultaneously.

## Exercises

### Boosting Ablation Study *(hard)*

Using a dataset of your choice (recommended: Kaggle's House Prices or a tabular dataset from OpenML), train XGBoost, LightGBM, and CatBoost with the same hyperparameter budget (100 Optuna trials each). Compare: (1) best RMSE achieved, (2) wall-clock training time, (3) model file size on disk, (4) inference latency (time to predict 10,000 samples). Create a summary table and identify which framework you'd choose for a latency-sensitive production system vs. a batch prediction system.

### Build a Custom Stacking Pipeline *(hard)*

Implement a 2-level stacking ensemble from scratch (without using sklearn's StackingRegressor): (1) Generate out-of-fold predictions for 3 diverse base models using 5-fold CV. (2) Train a Ridge meta-learner on the stacked predictions. (3) Compare with passthrough=True (concatenating original features). (4) Measure the improvement over the best single model. Bonus: Add a 2nd stacking level where a LightGBM meta-learner sits on top of the Ridge meta-learner's predictions plus original features.

---

**Next up:** Lesson 5 brings everything together with a capstone on model deployment and monitoring: serialization formats (ONNX, PMML), feature stores, A/B testing ML models, detecting data drift and concept drift, and building a production ML pipeline that retrains and validates automatically.