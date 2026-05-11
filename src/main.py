"""Train and save the best complete ML pipeline.

Run from the project root:
    python -m src.main

For a quick local smoke test:
    python -m src.main --n-iter 2 --cv 2
"""

from __future__ import annotations

import argparse

import src.data_preprocessing as dp
from src.ml_pipeline import (
    cross_validation,
    data_split,
    define_models,
    model_evaluation,
    model_selection_and_training,
    save_pipeline,
    save_training_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/bank-additional-full.csv")
    parser.add_argument("--model-path", default="app/model/pipeline.pkl")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bank_df = dp.read_data(args.data_path)
    X_train, X_test, y_train, y_test = data_split(bank_df)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    cv_model = cross_validation(number_splits=args.cv)
    models_params = define_models(scale_pos_weight)

    best_model_name, best_params, best_pipeline, results_df = model_selection_and_training(
        models_params,
        X_train,
        y_train,
        cv_model,
        n_iter=args.n_iter,
    )

    print(f"best_model_name : {best_model_name}\n")
    print(f"params : {best_params}\n")
    print(f"best_pipeline : {best_pipeline}\n")

    metrics = model_evaluation(best_pipeline, X_test, y_test)
    metrics["best_model_name"] = best_model_name
    metrics["best_params"] = best_params

    save_pipeline(best_pipeline, args.model_path)
    save_training_outputs(results_df, metrics)

    print(f"\nPipeline sauvegardée dans : {args.model_path}")
