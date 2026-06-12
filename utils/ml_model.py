import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

MODELS_DIR = "models"


def prepare_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)


def train_random_forest(X_train, y_train):
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 6, None],
        "classifier__min_samples_split": [2, 5],
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_xgboost(X_train, y_train):
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        )),
    ])
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(filename):
    with open(os.path.join(MODELS_DIR, filename), "rb") as f:
        return pickle.load(f)


def predict_diabetes(model, input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "label": "Diabetic" if prediction == 1 else "Non-Diabetic",
        "probability": round(float(probability) * 100, 1),
    }


def train_and_save_all(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    rf_model, rf_params = train_random_forest(X_train, y_train)
    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    save_model(rf_model, "random_forest.pkl")
    save_model(xgb_model, "xgboost.pkl")
    return {
        "rf": {"model": rf_model, "metrics": rf_metrics, "params": rf_params},
        "xgb": {"model": xgb_model, "metrics": xgb_metrics, "params": xgb_params},
    }
