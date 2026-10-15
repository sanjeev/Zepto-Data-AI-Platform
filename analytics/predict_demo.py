"""Reload the saved end-to-end pipeline and predict on raw, unpreprocessed passengers.

Run from this folder:  python predict_demo.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cleaning import structural_clean

pipeline = joblib.load("titanic_best_pipeline.joblib")
print("Loaded:", [name for name, _ in pipeline.steps], "->", type(pipeline.named_steps["clf"]).__name__)

# Raw rows exactly as a new passenger record would arrive (missing values included).
new_passengers = pd.DataFrame([
    {"pclass": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 211.34, "sex": "female", "embarked": "S"},
    {"pclass": 3, "age": 24, "sibsp": 0, "parch": 0, "fare": 7.90, "sex": "male", "embarked": "S"},
    {"pclass": 2, "age": np.nan, "sibsp": 1, "parch": 2, "fare": 26.00, "sex": "female", "embarked": None},
])
print(new_passengers.assign(pred_survived=pipeline.predict(new_passengers),
                            p_survived=pipeline.predict_proba(new_passengers)[:, 1].round(3)).to_string())

# Re-create the notebook's held-out test split from the committed CSV and confirm the score.
df = structural_clean(pd.read_csv("titanic.csv"))
X = df[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]]
_, X_test, _, y_test = train_test_split(X, df["survived"], test_size=0.2, stratify=df["survived"], random_state=42)
print("test accuracy of reloaded pipeline:", round(accuracy_score(y_test, pipeline.predict(X_test)), 4))
