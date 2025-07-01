def test_woe_scaling():
    from src.data_processing import preprocess_data
    import pandas as pd

    df = pd.DataFrame({
        'Value': [100, 200, 300],
        'Frequency': [1, 2, 3],
        'Monetary': [10, 20, 30],
        'ProductCategory': ['A', 'B', 'A'],
        'ChannelId': ['web', 'android', 'web']
    })

    result = preprocess_data(df)
    assert result.shape[0] == 3
