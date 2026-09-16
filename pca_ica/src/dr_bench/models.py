"""Classifier registry.

The linear baseline (logistic_regression), two SVMs, KNN, two tree ensembles,
and a neural net (MLP). Sensible fixed hyperparameters — tune later with Optuna
if you want; the point of the scaffold is the model × reducer × dataset sweep,
not squeezing the last point of AUC out of any one model.

To add a model: add a branch here and its name to MODELS.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS = [
    "logistic_regression",
    "svm_rbf",
    "svm_linear",
    "knn",
    "random_forest",
    "xgboost",
    "mlp",
]


def make_model(name: str, random_state: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=random_state)
    if name == "svm_linear":
        return SVC(kernel="linear", probability=True, random_state=random_state)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=random_state)
    raise ValueError(f"Unknown model: {name}. Options: {MODELS}")
