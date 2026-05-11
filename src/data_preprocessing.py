import pandas as pd
import numpy as np


TARGET_COLUMN = "y"


def read_data(data_file_path: str) -> pd.DataFrame:

    df = pd.read_csv(data_file_path, sep=";")

    return df


def encode_target(y: pd.Series) -> pd.Series:

    """Encode the target variable for model training only."""
    return y.map({"no": 0, "yes": 1}).astype(int)


def remove_useless_columns(df) :

    """Remove columns intentionally excluded from the model.

    duration is generally excluded because it is only known after the call.
    """

    try :
        df = df.drop(columns=["duration", "emp.var.rate", "nr.employed"])
    except KeyError :
        raise ValueError("Column 'duration' or column 'emp.var.rate' or column 'nr.employed' not found")
    
    return df


def feature_engineering(df) : 

    try:
        df["contacted"] = (df["pdays"] != 999).astype(int)
        # df["pdays"] = df["pdays"].replace(999, -1)
    except KeyError:
        raise ValueError("Column 'pdays' not found")
    
    return df

"""
def group_data(df) :

    job_mapping = {
        # Col blanc (bureau, qualifié)
        "admin.": "white_collar",
        "management": "white_collar",

        # Techniques / services qualifiés
        "technician": "skilled_worker",
        "services": "skilled_worker",

        # Manuel
        "blue-collar": "manual_worker",

        # Indépendants
        "entrepreneur": "self_employed",
        "self-employed": "self_employed",

        # Inactifs
        "retired": "inactive",
        "student": "inactive",
        "unemployed": "inactive",

        # Autres
        "housemaid": "housemaid",
        "unknown": "unknown"
    }

    education_mapping = {
        "basic.4y": "basic",
        "basic.6y": "basic",
        "basic.9y": "basic",

        "high.school": "intermediate",

        "professional.course": "higher",
        "university.degree": "higher",

        "illiterate": "low",
        "unknown": "unknown"
    }

    df["job_grouped"] = df["job"].map(job_mapping)
    df["education_grouped"] = df["education"].map(education_mapping)

    df = df.drop(columns=["job", "education"])

    return df

"""

def data_types_review(df) :
    
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue

        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            df[col] = df[col].astype("category")

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature preprocessing before sklearn encoding.

    This function is inserted inside the sklearn Pipeline via FunctionTransformer,
    so FastAPI and training use exactly the same business logic.
    """
    df = remove_useless_columns(df)
    df = feature_engineering(df)
    df = data_types_review(df)
    return df
