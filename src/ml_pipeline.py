from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier

import pandas as pd


def data_split(df) :

    try :
        bank_data = df.drop(columns=["y"])
        y = df["y"]
    except KeyError :
        raise ValueError("Column 'y' not found")
    
    X_train, X_test, y_train, y_test = train_test_split(bank_data, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def cross_validation(number_splits) :

    cv = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=42)

    return cv


def define_models(scale_pos_weight) :


    models_and_params = {

        "logistic_regression": {
            "model": LogisticRegression(),
            "params": {
                "max_iter" : [3000, 3500],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "class_weight": [None, "balanced"]
            }
        },

        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [250, 500, 700, 1000],
                "max_depth": [3, 5, 7, 10, 12],
                "class_weight": ["balanced", "balanced_subsample"]
            }
        },

        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [250, 500, 700, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                "max_depth": [3, 5, 7, 10, 12]
            }
        },

        "xgboost": {
            "model": XGBClassifier(tree_method="hist", random_state=42, scale_pos_weight=scale_pos_weight),
            "params": {
                "n_estimators": [250, 500, 700, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                "max_depth": [3, 5, 7, 10, 12]
            }
        }
    }

    return models_and_params

"""
    ce qui est stocké pour chaque model, c'est le model entrainé ainsi que les paramètres qui ont obtenu la meilleur score moyen F1
"""
def model_selection_and_training(models_dict, train_data, train_targets, cv_model) :

    results = []
    best_estimators = {}

    print("exécution...")
    
    for model_name, config in models_dict.items() :

        grid = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            cv=cv_model,
            scoring="f1",
            n_iter=50,      
            n_jobs=-1,
            verbose=2,
            refit=True
        )

        grid.fit(train_data, train_targets)

        results.append({
            "model": model_name,
            "best_cv_score": grid.best_score_,
            "best_params": grid.best_params_
        })

        best_estimators[model_name] = grid.best_estimator_

    results_df = pd.DataFrame(results).sort_values(by="best_cv_score", ascending=False)

    best_model_name = results_df.iloc[0]["model"]
    best_params = results_df.iloc[0]["best_params"]
    best_model = best_estimators[best_model_name]

    return best_model_name, best_params, best_model


def model_evaluation(model, test_data, test_targets) :

    y_pred = model.predict(test_data)

    print("\nClassification report :")
    print(classification_report(test_targets, y_pred))
    print("\nMatrice de confusion :")
    print(confusion_matrix(test_targets, y_pred))