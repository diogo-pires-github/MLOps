import pandas as pd
import numpy as np
import pytest
from src.credit_fraud_detector.pipelines.data_preprocessing.nodes import (
    load_and_preprocess_data,
    shuffle_and_undersample,
    remove_outliers,
    preprocess_data
)


from sklearn.preprocessing import StandardScaler, RobustScaler


def sample_data():
    # Create a DataFrame similar to the sample_creditcard.csv file
    data = {
        'Time': [0, 1, 2, 3, 4, 5, 6, 7],
        'V1': [-1.35980713367380, 1.19185711131486, -1.35835406159823, -0.96627171157209, -1.15823309349523, -0.42596588441245, 0.12345678901234, -0.23456789012345],
        'V2': [-0.07278117330985, 0.26615071205963, -1.34016307473609, -0.18522600808290, 0.87773675484845, 0.96052304488298, -0.34567890123456, 0.45678901234567],
        'V3': [2.53634673796914, 0.16648011335321, 1.77320934263119, 1.79299333957872, 1.54871784651100, 1.14110934232219, 0.56789012345678, -0.67890123456789],
        'V4': [1.37815522427443, 0.44815407846091, 0.37977959303433, -0.86329127503645, 0.40303393395512, -0.16825207976030, -0.78901234567890, 0.89012345678901],
        'V5': [-0.33832076994252, 0.06001764928222, -0.50319813331819, -0.01030887960308, -0.40719337731165, 0.42098688077219, 0.90123456789012, -0.01234567890123],
        'V6': [0.46238777776229, -0.08236080881557, 1.80049938079263, 1.24720316752486, 0.09592146246843, -0.02972755166397, -0.12345678901234, 0.23456789012345],
        'V7': [0.23959855406126, -0.07880298333231, 0.79146095645042, 0.23760893977178, 0.59294074538555, 0.47620094872003, 0.34567890123456, -0.45678901234567],
        'V8': [0.09869790126105, 0.08510165491481, 0.24767578658899, 0.37743587465226, -0.27053267719228, 0.26031433307487, -0.56789012345678, 0.67890123456789],
        'V9': [0.36378696961121, -0.25542512810919, -1.51465432260583, -1.38702406270200, 0.81773930823529, -0.56867137571251, 0.78901234567890, -0.89012345678901],
        'V10': [0.09079417197893, -0.16697441400461, 0.20764286521670, -0.05495192247137, 0.75307443197635, -0.37140719683447, -0.90123456789012, 0.01234567890123],
        'V11': [-0.55159953326081, 1.61272666105479, 0.62450145942490, -0.22648726383540, -0.82284287794636, 1.34126198001957, 0.12345678901234, -0.23456789012345],
        'V12': [-0.61780085576235, 1.06523531137287, 0.06608368526883, 0.17822822587730, 0.53819555014995, 0.35989383703804, -0.34567890123456, 0.45678901234567],
        'V13': [-0.99138984723541, 0.48909501589608, 0.71729273141083, 0.50775686995717, 1.34585159321540, -0.35809065257363, 0.56789012345678, -0.67890123456789],
        'V14': [-0.31116935369988, -0.14377229644152, -0.16594592276355, -0.28792374549456, -1.11966983471731, -0.13713370021761, -0.78901234567890, 0.89012345678901],
        'V15': [1.46817697209427, 0.63555809325821, 2.34586494901581, -0.63141811770905, 0.17512113000899, 0.51761680655574, 0.90123456789012, -0.01234567890123],
        'V16': [-0.47040052525948, 0.46391704102217, -2.89008319444231, -1.05964724543250, -0.45144918281353, 0.40172589558960, -0.12345678901234, 0.23456789012345],
        'V17': [0.20797124192924, -0.11480466310235, 1.10996937869600, -0.68409278634548, -0.23703323936278, -0.05813282336401, 0.34567890123456, -0.45678901234567],
        'V18': [0.02579058019856, -0.18336127012400, -0.12135931319589, 1.96577500349540, -0.03819478703528, 0.06865314944254, -0.56789012345678, 0.67890123456789],
        'V19': [0.40399296025573, -0.14578304132526, -2.26185709530414, -1.23262197008920, 0.80348692496018, -0.03319378778763, 0.78901234567890, -0.89012345678901],
        'V20': [0.25141209823971, -0.06908313522302, 0.52497972522440, -0.20803778116037, 0.40854236039276, 0.08496767206820, -0.90123456789012, 0.01234567890123],
        'V21': [-0.01830677794415, -0.22577524803314, 0.24799815346975, -0.10830045203555, -0.00943069713233, -0.20825351465673, 0.12345678901234, -0.23456789012345],
        'V22': [0.27783757555890, -0.63867195277185, 0.77167940191723, 0.00527359678253, 0.79827849458971, -0.55982479625325, -0.34567890123456, 0.45678901234567],
        'V23': [-0.11047391018877, 0.10128802125323, 0.90941226234772, -0.19032051874284, -0.13745807961906, -0.02639766797954, 0.56789012345678, -0.67890123456789],
        'V24': [0.06692807491467, -0.33984647552913, -0.68928095649069, -1.17557533186320, 0.14126698382477, -0.37142658317435, -0.78901234567890, 0.89012345678901],
        'V25': [0.128539358273528, 0.167170404418143, -0.327641833735251, 0.647376034602038, -0.206009587619756, -0.232793816737034, -0.23456789012345, -0.23456789012345],
        'V26': [-0.01830677794415, -0.22577524803314, 0.24799815346975, -0.10830045203555, -0.00943069713233, -0.20825351465673, 0.31247587117642, -0.25743157939275],
        'V27': [0.27783757555890, -0.63867195277185, 0.77167940191723, 0.00527359678253, 0.79827849458971, -0.55982479625325, -0.47288896281415, 0.67552786712489],
        'V28': [-0.11047391018877, 0.10128802125323, 0.90941226234772, -0.19032051874284, -0.13745807961906, -0.02639766797954, -0.36965187625429, 0.19868229130560],
        'Amount': [149.62, 2.69, 378.66, 123.5, 69.99, 3.67, 9.72, 87.58],
        'Class': [0, 0, 0, 0, 0, 1, 1, 1]
    }
    return pd.DataFrame(data)

#sample_data().to_csv("tests/pipelines/sample/sample_creditcard.csv")


@pytest.fixture

def test_load_and_preprocess_data():
    # Load and preprocess the data
    sample_data = pd.read_csv("./tests/pipelines/sample/sample_creditcard.csv")
    processed_data = load_and_preprocess_data(sample_data)
    
    # Assertions to check the preprocessing
    assert 'Time' not in processed_data.columns
    assert 'Amount' not in processed_data.columns
    assert 'scaled_amount' in processed_data.columns
    assert 'scaled_time' in processed_data.columns

    # Check if the scaling was applied correctly
    rob_scaler = RobustScaler()
    expected_scaled_amount = rob_scaler.fit_transform(sample_data['Amount'].values.reshape(-1,1)).flatten()
    expected_scaled_time = rob_scaler.fit_transform(sample_data['Time'].values.reshape(-1,1)).flatten()

    np.testing.assert_array_almost_equal(processed_data['scaled_amount'].values, expected_scaled_amount, decimal=6)
    np.testing.assert_array_almost_equal(processed_data['scaled_time'].values, expected_scaled_time, decimal=6)

    # Check if the other features are unchanged
    for col in sample_data.columns:
        if col not in ['Time', 'Amount']:
            np.testing.assert_array_almost_equal(processed_data[col].values, sample_data[col].values, decimal=6)

    # Check if the Class column is intact
    assert 'Class' in processed_data.columns
    assert processed_data['Class'].equals(sample_data['Class'])


def test_shuffle_and_undersample():
    sample_data = pd.read_csv("./tests/pipelines/sample/sample_creditcard.csv")
    print("Running test_shuffle_and_undersample...")
    print(f"Initial class distribution:\n{sample_data['Class'].value_counts()}")

    undersampled_df = shuffle_and_undersample(sample_data)
    print(f"Class distribution after undersampling:\n{undersampled_df['Class'].value_counts()}")

    assert not undersampled_df.equals(sample_data)

    class_counts = undersampled_df['Class'].value_counts()
    assert class_counts[0] == 5
    assert class_counts[1] == len(sample_data[sample_data['Class'] == 1])

    assert len(undersampled_df) == 5 + len(sample_data[sample_data['Class'] == 1])

def test_remove_outliers():
    sample_data = pd.read_csv("./tests/pipelines/sample/sample_creditcard.csv")
    print("Running test_remove_outliers...")
    column = 'V14'
    sample_data_with_outliers = sample_data.copy()
    sample_data_with_outliers.loc[sample_data_with_outliers['Class'] == 1, column] = np.random.uniform(-10, 10, size=len(sample_data_with_outliers[sample_data_with_outliers['Class'] == 1]))

    print(f"Number of rows before removing outliers: {len(sample_data_with_outliers)}")
    cleaned_df = remove_outliers(sample_data_with_outliers, column)
    print(f"Number of rows after removing outliers: {len(cleaned_df)}")

    fraud_values = cleaned_df[column].loc[cleaned_df['Class'] == 1]
    q25, q75 = np.percentile(fraud_values, 25), np.percentile(fraud_values, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    outliers_removed = cleaned_df[(cleaned_df[column] > upper) | (cleaned_df[column] < lower)]
    assert outliers_removed.empty

def test_preprocess_data():
    sample_data = pd.read_csv("./tests/pipelines/sample/sample_creditcard.csv")
    print("Running test_preprocess_data...")
    print(f"Initial class distribution:\n{sample_data['Class'].value_counts()}")

    processed_df = preprocess_data(sample_data)
    print(f"Class distribution after preprocessing:\n{processed_df['Class'].value_counts()}")

    class_counts = processed_df['Class'].value_counts()
    print(f"Processed class counts: {class_counts}")

    assert class_counts[0] == 2
    assert class_counts[1] == len(sample_data[sample_data['Class'] == 1])

    assert len(processed_df) == 2 + len(sample_data[sample_data['Class'] == 1])

    for column in ['V14', 'V12', 'V10']:
        fraud_values = processed_df[column].loc[processed_df['Class'] == 1]
        q25, q75 = np.percentile(fraud_values, 25), np.percentile(fraud_values, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

        outliers_removed = processed_df[(processed_df[column] > upper) | (processed_df[column] < lower)]
        assert outliers_removed.empty