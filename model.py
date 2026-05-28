import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

training_data = pd.read_csv('training_data.csv')

feature_cols = [
        "Return_prev_5d", "Return_prev_10d", "Close_yesterday", 'Volatility',
        "Oil_Open", "Oil_volume", 'Return_prev_5d_oil', 'Return_prev_10d_oil',
        "Open", "Volume",
        "DayOfWeek", "Month"
    ]

X = training_data[feature_cols]
y = training_data[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class EnsembleAvg:
    """Average soft probabilities from multiple sklearn-compatible classifiers."""
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = np.mean([est.predict_proba(X) for est in self.estimators], axis=0)
        return probs

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_model():
    """
    Returns the classifier to be used.
    Modify this function during AutoResearch iterations.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=42,
            max_iter=1000,
        )),
    ])
    return model
