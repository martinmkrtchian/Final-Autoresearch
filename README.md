# AutoResearch: Energy Stock Intraday Return Classification

An automated machine learning research framework that iteratively builds and improves classifiers to predict whether each day's intraday return for energy sector stocks will be **positive (1) or negative (0)**. Each experiment is time-budgeted, automatically logged, and kept or reverted based on validation ROC-AUC.

---

## Tickers

`XOM` · `CVX` · `BP` · `CAT` · `SHEL` · `COP` · `CSUAY` · `PBR` · `ENB` · `MITSY` · `ITOCY`

---

## Repository Structure

| File | Purpose |
|------|---------|
| `training_data.csv` | Dataset with 12 features and binary `Target` variable |
| `test_data.csv` | Holdout test set (same structure as training data) |
| `model.py` | Model definition — `build_model()` is edited each iteration |
| `run.py` | Trains model, evaluates ROC-AUC, enforces 5-min budget, logs to `experiments.json` |
| `prepare.py` | Reads `experiments.json` and generates `performance.png` |
| `evaluate_test.py` | Trains best model on full training data, scores on test set, saves `test_predictions.csv` |
| `experiments.json` | Auto-generated log of all experiments (train/val/test AUC, runtime, status) |
| `performance.png` | Auto-generated chart: ROC-AUC progress + experiment summary table |

---

## Features

The model uses 12 input features:

| Feature | Description |
|---------|-------------|
| `Return_prev_5d` | Stock return over previous 5 days |
| `Return_prev_10d` | Stock return over previous 10 days |
| `Close_yesterday` | Previous day's closing price |
| `Volatility` | Rolling volatility |
| `Oil_Open` | Oil opening price |
| `Oil_volume` | Oil trading volume |
| `Return_prev_5d_oil` | Oil return over previous 5 days |
| `Return_prev_10d_oil` | Oil return over previous 10 days |
| `Open` | Stock opening price |
| `Volume` | Stock trading volume |
| `DayOfWeek` | Day of the week (0–4) |
| `Month` | Month of the year (1–12) |

**Target:** `1` if intraday return (`Close/Open - 1`) > 0, else `0`

---

## How to Run

### 1. Establish baseline
```bash
python run.py "baseline"
```

### 2. AutoResearch loop — edit model.py, then run
```bash
python run.py "description of what you changed"
```
If the new `val_roc_auc` is better than the previous best, the change is **kept**. Otherwise, revert `model.py` and try a different idea.

### 3. Generate performance chart
```bash
python prepare.py
```

### 4. Evaluate best model on test set
```bash
python evaluate_test.py
```
Saves predictions to `test_predictions.csv`.

---

## AutoResearch Loop

```
┌─────────────────────────────────────────────────────┐
│  Propose one change to model.py                     │
│  (estimator, features, hyperparams, ensemble, ...)  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          python run.py "description"
                     │
          ┌──────────▼──────────┐
          │  val_roc_auc        │
          │  > previous best?   │
          └──────────┬──────────┘
              Yes    │    No
               │     │     │
               ▼     │     ▼
             KEEP    │   REVERT model.py
               │     │     │
               └─────┴─────┘
                     │
                     ▼
           Repeat ≥ 15 iterations
```

---

## Experiment Results

| # | Description | Val AUC | Test AUC | Status |
|---|-------------|:-------:|:--------:|--------|
| 1 | Baseline XGB n=600 depth=4 lr=0.02 | 0.5957 | 0.6133 | KEPT |
| 2 | GBC n=200 depth=4 lr=0.05 | 0.6002 | 0.5962 | KEPT |
| 3 | XGB scale_pos_weight class imbalance | 0.5957 | 0.6111 | DISCARDED |
| 4 | GBC + SelectFromModel threshold=median | 0.6136 | 0.6346 | KEPT |
| 5 | GBC + SelectFromModel + interaction features | 0.6147 | 0.6118 | KEPT |
| 6 | XGB tuned n=400 depth=5 lr=0.05 + interactions | 0.6106 | 0.6329 | DISCARDED |
| 7 | RandomForest n=300 depth=6 + interactions | 0.5484 | 0.5435 | DISCARDED |
| 8 | RF tuned n=500 min_leaf=5 max_features=sqrt | 0.6250 | 0.6467 | KEPT |
| 9 | CatBoost iter=400 depth=6 lr=0.05 | 0.6114 | 0.6245 | DISCARDED |
| 10 | CatBoost tuned iter=800 depth=4 lr=0.02 l2=5 | 0.5865 | 0.5848 | DISCARDED |
| 11 | XGB n=300 depth=3 lr=0.1 + interactions | 0.5904 | 0.5968 | DISCARDED |
| 12 | Soft voting GBC+XGB+RF + interactions | 0.6056 | 0.6307 | DISCARDED |
| 13 | RF n=500 + PolynomialFeatures degree=2 | 0.6339 | 0.6520 | KEPT |
| 14 | ExtraTrees n=500 + PolynomialFeatures degree=2 | 0.6350 | 0.6534 | KEPT |
| 15 | Stacking ET+RF+GBC→LR + PolynomialFeatures | 0.6332 | 0.6634 | DISCARDED |
| **16** | **ExtraTrees n=1000 min_leaf=3 + PolynomialFeatures** | **0.6388** | **0.6551** | **KEPT** |

---

## Best Model

**ExtraTrees (n=1000, min_samples_leaf=3, max_features=sqrt) + PolynomialFeatures(degree=2, interaction_only=True)**

```python
Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("clf",  ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=3,
                                   max_features="sqrt", random_state=42, n_jobs=-1))
])
```

| Metric | Score |
|--------|-------|
| Validation ROC-AUC | 0.6388 |
| Test ROC-AUC | 0.6551 |

---

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib
```
