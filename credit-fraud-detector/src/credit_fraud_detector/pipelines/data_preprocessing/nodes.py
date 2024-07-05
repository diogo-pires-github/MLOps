import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    
    data = data.dropna()

    # RobustScaler is less prone to outliers.
    rob_scaler = RobustScaler()

    data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
    data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

    data.drop(['Time','Amount'], axis=1, inplace=True)  

    scaled_amount = data['scaled_amount']
    scaled_time = data['scaled_time']

    data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    data.insert(0, 'scaled_amount', scaled_amount)
    data.insert(1, 'scaled_time', scaled_time)

    return preprocess_data(data)


def shuffle_and_undersample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sample(frac=1)
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    return new_df

def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    fraud_values = df[column].loc[df['Class'] == 1].values
    q25, q75 = np.percentile(fraud_values, 25), np.percentile(fraud_values, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    df = df.drop(df[(df[column] > upper) | (df[column] < lower)].index)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = shuffle_and_undersample(df)
    for column in ['V14', 'V12', 'V10']:
        df = remove_outliers(df, column)
    return df
