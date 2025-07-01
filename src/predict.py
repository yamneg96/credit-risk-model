# src/predict.py

import joblib
import pandas as pd

def load_model(model_path="models/credit_model.pkl"):
    """Loads the saved model from disk."""
    return joblib.load(model_path)

def predict_from_dataframe(df, model=None):
    """
    Takes a DataFrame with the required columns and returns risk probabilities.
    
    Parameters:
        df (pd.DataFrame): Data with features like Value, Frequency, etc.
        model: Trained model object (if not provided, it will be loaded)
    
    Returns:
        pd.Series: Risk probabilities for each row
    """
    if model is None:
        model = load_model()

    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(df)[:, 1], index=df.index, name="risk_probability")
    else:
        raise ValueError("Model does not support probability prediction")
