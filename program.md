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

### Phase 1: Can a linear model find signal in the data? (Experiments 1–2)
1. Start with a logistic regression as the simplest possible baseline. Does the relationship between features and intraday returns have any linear structure at all?
2. The target classes may not be balanced. Does applying class weighting change anything for the linear model?

### Phase 2: Do nonlinear models capture more structure? (Experiments 3–5)
3. Try a Random Forest. Tree-based models can capture interactions between features — does that matter here?
4. Revisit class balancing, this time on the Random Forest. Is the improvement consistent across model families?
5. Random Forests can overfit with deep trees. Try constraining depth and requiring a minimum number of samples per leaf — does regularizing the forest improve generalization?

### Phase 3: Does boosting outperform bagging on this task? (Experiments 6–8)
6. Try a Gradient Boosting classifier. Boosting and bagging are both ensemble strategies, but they work very differently — which is better suited to this problem?
7. Gradient boosting can also overfit. Try adding regularization (subsampling, minimum leaf size) — does it help close the train-validation gap?
8. Learning rate and number of estimators are closely linked. Try a lower learning rate with more trees — is there a generalization benefit?

### Phase 4: Can XGBoost push performance further? (Experiments 9–12)
9. XGBoost is a high-performance boosting framework with built-in regularization. Does it outperform the sklearn GradientBoosting?
10. XGBoost provides explicit L1/L2 regularization parameters. Does tuning these reduce overfitting?
11. Try XGBoost with class balancing (scale_pos_weight). Does addressing class imbalance help in the same way it did for earlier models?
12. Combine the regularization and class balancing insights from experiments 10 and 11. Do they reinforce each other?

### Phase 5: Is CatBoost a better fit for this dataset? (Experiments 13–16)
13. CatBoost is another gradient boosting framework with different internal mechanics. How does it compare to XGBoost out of the box?
14. Try adding L2 regularization to CatBoost. Does it have the same effect as in XGBoost?
15. Apply CatBoost's built-in class weighting. Is the effect consistent with what we saw in other models?
16. Combine the best regularization and class balancing settings for CatBoost. What is the ceiling for a single CatBoost model?

### Phase 6: Can combining models beat any single model? (Experiments 17–20)
17. Ensemble the best XGBoost and CatBoost configurations by averaging their predicted probabilities. Do two diverse boosting models generalize better together?
18. Add the GradientBoosting model to the ensemble. Does a third, more different learner add useful diversity?
19. Add Random Forest to the ensemble as a fourth model. Does a bagging model contribute anything beyond what the boosting models already capture?
20. Based on what the ensemble experiments reveal, settle on the best combination. What is the final performance ceiling?
