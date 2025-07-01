# test_predict.py (for manual testing)

import pandas as pd
from src.predict import predict_from_dataframe

# Sample input
data = pd.DataFrame({
    "Value": [100, 200],
    "Frequency": [5, 10],
    "Monetary": [1000, 2000],
    "ProductCategory": ["A", "B"],
    "ChannelId": ["web", "android"]
})

# You may need to preprocess this first
# e.g., using src.data_processing.preprocess_data()

# Predict
probs = predict_from_dataframe(data)
print(probs)
