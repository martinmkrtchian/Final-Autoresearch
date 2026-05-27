import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model import build_model

test_data = pd.read_csv('test_data.csv')

feature_cols = [
        "Return_prev_5d", "Return_prev_10d", "Close_yesterday", 'Volatility',
        "Oil_Open", "Oil_volume", 'Return_prev_5d_oil', 'Return_prev_10d_oil',
        "Open", "Volume",
        "DayOfWeek", "Month"
    ]

model = build_model()
model.fit(test_data[feature_cols], test_data['Target'])

eval_proba   = model.predict_proba(test_data[feature_cols])[:, 1]
thr = 0.5
eval_auc   = roc_auc_score(test_data['Target'],   eval_proba>thr)
print(eval_auc)
