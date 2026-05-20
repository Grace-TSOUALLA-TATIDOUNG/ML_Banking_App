import pandas as pd
import numpy as np

from pandas.api.types import is_integer_dtype, is_float_dtype, is_numeric_dtype


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
        df = df.drop(columns=["duration", "emp.var.rate", "nr.employed", "previous", "campaign", "day_of_week"])
    except KeyError :
        raise ValueError("Column 'duration' or column 'emp.var.rate' or column 'nr.employed' not found")
    
    return df


def feature_engineering(df) : 

    try:
        df["contacted"] = (df["pdays"] != 999).astype(int)
        df["previous_contacted"] = (df["previous"] > 0).astype(int)
        df["previous_log"] = np.log1p(df["previous"])
        # df["pdays"] = df["pdays"].replace(999, -1)
        df["campaign_capped"] = df["campaign"].clip(upper=10)
        df["campaign_log"] = np.log1p(df["campaign_capped"])
    except KeyError:
        raise ValueError("Column 'pdays' or 'previous' or 'campaign' not found")
    
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

def data_types_review(df):
    df = df.copy()

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue

        # Numérique → optimiser le type
        if is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")

        elif is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Tout le reste → catégoriel (en object, pas category)
        elif not is_numeric_dtype(df[col]):
            df[col] = df[col].astype("object")

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature preprocessing before sklearn encoding.

    This function is inserted inside the sklearn Pipeline via FunctionTransformer,
    so FastAPI and training use exactly the same business logic.
    """
    df = feature_engineering(df)
    df = data_types_review(df)
    df = remove_useless_columns(df)
    return df
