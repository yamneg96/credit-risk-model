from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

from sklearn.cluster import KMeans
from datetime import datetime

class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        return X

def get_pipeline():
    date_features = DateFeatures()
    num_features = ['Value']
    cat_features = ['ChannelId', 'ProductCategory']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    full_pipeline = Pipeline([
        ('date', date_features),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

def create_rfm_labels(df):
    snap_date = datetime(2021, 12, 1)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snap_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    rfm_scaled = StandardScaler().fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Assign high-risk to lowest spend cluster
    risk_map = rfm.groupby('cluster')['Monetary'].mean().sort_values().index[0]
    rfm['is_high_risk'] = (rfm['cluster'] == risk_map).astype(int)

    return rfm[['is_high_risk']]

def test_date_features():
    from src.data_processing import DateFeatures
    import pandas as pd

    df = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 08:00', '2023-01-02 14:00']
    })
    transformer = DateFeatures()
    transformed = transformer.fit_transform(df)

    assert 'transaction_month' in transformed.columns
    assert 'transaction_hour' in transformed.columns