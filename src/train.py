import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib


def train_model():
    df = pd.read_csv("data/processed/final_dataset.csv")
    X = df.drop("is_high_risk", axis=1)
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/credit_model.pkl")