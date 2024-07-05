import pandas as pd
import numpy as np

from src.credit_fraud_detector.pipelines.data_split.nodes import split_data2


def test_split_data():
    """Test the split_data function.
    """

    # Create a sample DataFrame
    df = pd.read_csv("./tests/pipelines/sample/sample_creditcard.csv")

    # Define the parameters
    parameters = {
        'target_column': 'Class',
        'random_state': 42,
        'test_fraction': 0.2
    }

    # Call the split_data function
    X_train, X_test, y_train, y_test = split_data2(df, parameters)


    # Assert the existence of the datasets
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None


    # Assert the shapes of the resulting datasets
    assert X_train.shape == (6, 31)
    assert X_test.shape == (2, 31)
    assert y_train.shape == (6,)
    assert y_test.shape == (2,)
