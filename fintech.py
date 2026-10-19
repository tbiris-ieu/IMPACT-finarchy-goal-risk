"""
Finarchy: build the dataset and train the goal-abandonment model.

Run it directly to (re)create fintech.xlsx and fintech.joblib:

    python fintech.py

flask_app.py also imports train_and_save() as a fallback, for the case where
the saved fintech.joblib cannot be used with the installed scikit-learn version.
Everything is seeded, so every run produces the same data and the same model.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Output files live next to this script, whatever the current working directory is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fintech.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "fintech.joblib")


def build_dataset(n=1000, seed=42):
    """Population: active savings goals on Finarchy (one row per goal)."""
    # Set the seed so all variables always get the same values
    np.random.seed(seed)

    months_since_goal_created = np.random.randint(1, 37, n)                      # discrete
    avg_monthly_contribution = np.round(np.random.uniform(10, 500, n), 2)         # continuous (EUR)
    missed_contributions_last_6_months = np.random.randint(0, 7, n)               # discrete

    # Target: goal_abandoned (1 = abandoned before goal date, 0 = on track)
    # Generated randomly, with probability driven by the three variables
    logit = (-1.0
             - 0.03 * months_since_goal_created
             - 0.006 * avg_monthly_contribution
             + 0.9 * missed_contributions_last_6_months)
    prob = 1 / (1 + np.exp(-logit))
    goal_abandoned = np.random.binomial(1, prob)

    return pd.DataFrame({
        "months_since_goal_created": months_since_goal_created,
        "avg_monthly_contribution": avg_monthly_contribution,
        "missed_contributions_last_6_months": missed_contributions_last_6_months,
        "goal_abandoned": goal_abandoned,
    })


def train_and_save(write_data=True, verbose=True):
    """Build the data, train the logistic regression, save the files, return the model."""
    df = build_dataset()

    # Save the data to Excel without index
    if write_data:
        df.to_excel(DATA_PATH, index=False)

    # Split into train and test
    X = df.drop(columns="goal_abandoned")
    y = df["goal_abandoned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Validate on train and test
    if verbose:
        print("Classification report - TRAIN")
        print(classification_report(y_train, model.predict(X_train)))
        print("Classification report - TEST")
        print(classification_report(y_test, model.predict(X_test)))

    # Dump the model
    joblib.dump(model, MODEL_PATH)
    if verbose:
        print("Saved fintech.xlsx and fintech.joblib")

    return model


if __name__ == "__main__":
    train_and_save()
