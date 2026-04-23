import pandas as pd
import numpy as np


def read_data(data_file_path) :

    df = pd.read_csv(data_file_path, sep=";")

    return df


def remove_useless_columns(df) :

    # "day_of_week"

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
    
    df["y"] = df["y"].map({"no": 0, "yes": 1}).astype(int)
    
    for col in df.columns:

        if np.issubdtype(df[col].dtype, np.integer):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        
        elif np.issubdtype(df[col].dtype, np.floating):
            df[col] = pd.to_numeric(df[col], downcast="float")
        
        else:
            df[col] = df[col].astype("category")

    return df


def data_encoding(df) :

    categorical_columns = df.select_dtypes(include=["category"]).columns

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True).astype("int8")

    return df
