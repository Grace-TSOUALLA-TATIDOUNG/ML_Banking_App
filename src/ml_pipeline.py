from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data_preprocessing import encode_target, preprocess_features

from xgboost import XGBClassifier
from typing import Any
from pathlib import Path

import pandas as pd
import joblib
import json


def data_split(df: pd.DataFrame) :

    try :
        X = df.drop(columns=["y"])
        y = encode_target(df["y"])
    except KeyError :
        raise ValueError("Column 'y' not found")
    
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )


def cross_validation(number_splits: int) :

    cv = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=42)

    return cv


def create_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with several sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def create_preprocessor() -> ColumnTransformer:
    """Create sklearn preprocessing after the business preprocessing step."""
    return ColumnTransformer(
        transformers=[
            ("cat", create_one_hot_encoder(), selector(dtype_include=["object", "category"])),
            ("num", "passthrough", selector(dtype_include=["number"])),
        ],
        remainder="passthrough",
    )


def create_pipeline(model) -> Pipeline:
    """Create a full production-ready pipeline: business preprocessing + encoding + model."""
    return Pipeline(
        steps=[
            ("business_preprocessing", FunctionTransformer(preprocess_features, validate=False)),
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )



def define_models(scale_pos_weight: float) -> dict[str, dict[str, Any]]:
    """Define candidate models and hyperparameters.

    Parameters are prefixed with model__ because the estimator is inside a Pipeline.
    """
    models_and_params: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "model": LogisticRegression(solver="liblinear", random_state=42),
            "params": {
                "model__max_iter": [3000, 3500],
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [250, 500, 700, 1000],
                "model__max_depth": [3, 5, 7, 10, 12],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [250, 500, 700, 1000],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                "model__max_depth": [3, 5, 7, 10, 12],
            },
        },
    }

    if XGBClassifier is not None:
        models_and_params["xgboost"] = {
            "model": XGBClassifier(
                tree_method="hist",
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
            ),
            "params": {
                "model__n_estimators": [250, 500, 700, 1000],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                "model__max_depth": [3, 5, 7, 10, 12],
            },
        }

    return models_and_params

"""
    ce qui est stocké pour chaque model, c'est le model entrainé ainsi que les paramètres qui ont obtenu la meilleur score moyen F1
"""

def model_selection_and_training(
    models_dict: dict[str, dict[str, Any]],
    train_data: pd.DataFrame,
    train_targets: pd.Series,
    cv_model,
    n_iter: int = 50,
):
    """Compare full pipelines and return the best fitted pipeline."""
    results = []
    best_estimators = {}

    print("Exécution de la comparaison des pipelines...")

    for model_name, config in models_dict.items():
        pipeline = create_pipeline(config["model"])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            cv=cv_model,
            scoring="f1",
            n_iter=n_iter,
            n_jobs=-1,
            verbose=2,
            refit=True,
            random_state=42,
        )

        search.fit(train_data, train_targets)

        results.append(
            {
                "model": model_name,
                "best_cv_score": search.best_score_,
                "best_params": search.best_params_,
            }
        )
        best_estimators[model_name] = search.best_estimator_

    results_df = pd.DataFrame(results).sort_values(by="best_cv_score", ascending=False)

    best_model_name = results_df.iloc[0]["model"]
    best_params = results_df.iloc[0]["best_params"]
    best_pipeline = best_estimators[best_model_name]

    return best_model_name, best_params, best_pipeline, results_df


def model_evaluation(model: Pipeline, test_data: pd.DataFrame, test_targets: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(test_data)
    y_proba = model.predict_proba(test_data)[:, 1]

    report_dict = classification_report(test_targets, y_pred, output_dict=True)
    matrix = confusion_matrix(test_targets, y_pred)

    print("\nClassification report :")
    print(classification_report(test_targets, y_pred))
    print("\nMatrice de confusion :")
    print(matrix)

    return {
        "f1_positive_class": float(f1_score(test_targets, y_pred)),
        "roc_auc": float(roc_auc_score(test_targets, y_proba)),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }


def save_pipeline(model: Pipeline, path: str = "app/model/pipeline.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_training_outputs(results_df: pd.DataFrame, metrics: dict[str, Any], output_dir: str = "app/model") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / "model_comparison.csv", index=False)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


# Backward-compatible alias for your previous function name.
def model_pickle(model: Pipeline) -> None:
    save_pipeline(model)