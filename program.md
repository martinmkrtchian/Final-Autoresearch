# AutoResearch: Energy Stock Intraday Return Classification

## Objective
Classify whether each day's **intraday return** for energy stocks will be **positive (1) or negative (0)**.
- Target: `Intraday_return = Close/Open - 1` → binary label (`1` if > 0, else `0`)
- Optimization metric: **Validation ROC-AUC**

## Tickers
XOM, CVX, BP, CAT, SHEL, COP, CSUAY, PBR, ENB, MITSY, ITOCY

## Data Source
- csv file called "training_data.csv" that is in this repository 



## Runtime Budget
Each experiment has a **5-minute wall-clock budget**. If exceeded:
- The run is logged as `timeout: true`
- The change is automatically discarded
- You must revert `model.py` to the previous version

## Files
| File | Purpose |
|------|---------|
| `training_data.csv` | 80% of the data that the model will use for training and validation |
| `test_data.csv` | 20% of the data used to evaluate the performance of the model |
| `model.py` | Data loading, feature engineering, model definition — **edited each iteration** |
| `run.py` | Trains model, evaluates ROC-AUC, enforces 5-min budget, logs to `experiments.json` |
| `prepare.py` | Reads `experiments.json`, generates `performance.png` |



---

## AutoResearch Loop Instructions

Read `program.md` for your instructions, then read `model.py`.

Run `python run.py "baseline"` to establish the baseline ROC-AUC.

Then enter the AutoResearch loop:

1. Propose one modification to `model.py` (e.g., different estimator,
   feature engineering, hyperparameter change, class balancing, new features).
2. Edit `model.py` with your change.
3. Run: `python run.py "<short description of what you changed>"`
4. Compare the new `val_roc_auc` to the current best.
   - If improved: **KEEP** the change, note the new best.
   - If worse or TIMEOUT: **REVERT** `model.py` to the previous version.
5. Calculate the ROC-AUC on the test_data.csv dataset
6. Repeat from step 1. Try **at least 15 different ideas**.

After all iterations:
1. Run `python prepare.py` → generates `performance.png`
2. Print a summary table of all experiments (kept / discarded / timeout)

---
## An important note

Make sure that the model does not overfit the training dataset. As soon as some overfitting appears, regularize the model. 

## Suggested Ideas to Try
1)Switch estimator to GradientBoostingClassifier (after leakage fix)

2)Try class_weight="balanced" to handle class imbalance

3)Feature selection: Drop low-importance features (use permutation importance)

4)Hyperparameter tuning:

  n_estimators: [100, 200, 400]
  max_depth: [3, 5, 7]
  learning_rate: [0.01, 0.05, 0.1]

5) Try multiple linear or polynomial regression and based on it understand which features are statistically significant predictors.

6) Try CatBoost
7) Find proper hyperparameters for Cat Boost
8) Try XGBoost with some grid and tune hyperparameters
9) Try Random Forest with different hyperparameters
10) Try ensembles with some of the models used above that perform well.
11) Based on the results think about what could be tuned. If certain model works better, stick to it and tune the hyperparameters.
12) Avoid overfitting the model to the training dataset. The training and validation scores should be similar. The difference should be no larger than 0.15. 
