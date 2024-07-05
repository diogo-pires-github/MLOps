import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from great_expectations.core import ExpectationSuite
from keyword import iskeyword
from typing import Union
from termcolor import cprint

load_dotenv()

def _to_feature_store_logic(
    df: pd.DataFrame,
    group_name: str,
    description: str,
    group_description: dict,
    feature_group_version: Union[int, None] = None,
    validation_expectation_suite: ExpectationSuite = None
):
    '''
    This function takes in a pandas DataFrame and a validation expectation suite,
      performs validation on the data using the suite, and then saves the data to a
      feature store in the feature store.

    Args:
        - data (pd.DataFrame): Dataframe with the data to be stored
        - group_name (str): Name of the feature group.
        - feature_group_version (int): Version of the feature group.
        - description (str): Description for the feature group.
        - group_description (dict): Description of each feature of the feature group.
        - validation_expectation_suite (ExpectationSuite): group of expectations to check data.

    Returns:
        - A dictionary with the feature view version, feature view name and training dataset feature version.
    '''
    ##### DELETE THIS BIT AFTER GX IS IMPLEMENTED #####
    if validation_expectation_suite:
        raise NotImplementedError
    ###################################################

    if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
        raise TypeError(f'Expect pd.DataFrame, got {type(df)}')
    if isinstance(df, pd.Series):
        new_df = pd.DataFrame({'index': df.index, df.name: df})
        df = new_df

    # Create primary key to posteriorly joins
    if 'index' not in df.columns:
        df = df.reset_index()

    # Hopsworks only accepts lowercase column names, better to sanitize beforehand. Also, try to protect from Python's reserved words
    df.columns = list(map(lambda x: x.lower() if not iskeyword(x.lower()) else x.lower() + '_', df.columns))


    # Get credentials
    project_name = os.environ.get('FS_PROJECT_NAME')
    api_key = os.environ.get('FS_API_KEY')


    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=api_key, project=project_name,
    )

    feature_store = project.get_feature_store()


    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        primary_key=['index'],
        description= description,
        online_enabled=False,
        expectation_suite=validation_expectation_suite
    )

    # Upload data.
    object_feature_group.insert(
        features=df,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.
    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def upload_to_feature_store(
    df: pd.DataFrame
):
    '''
    This function takes as argument the preprocessed df, splits into X and y and
      uploads the Features and Target to Hopsworks.

    Args:
        - df (pd.DataFrame): Dataframe with preprocessed features and target.

    Returns:
        - None
    '''
    # Feature / Target split
    X = df.drop(columns=['Class'])
    y = df.Class

    # Features metadata and upload
    feature_descriptions = [{'name': f'v{i}', 'description': 'Anonymized credit card data', 'validation_rules': 'TO DETERMINE'} for i in range(1, 29)]
    feature_descriptions += [
        {'name': 'scaled_amount', 'description': 'Scaled amount of transaction', 'validation_rules': 'TO DETERMINE'},
        {'name': 'scaled_time', 'description': 'Scaled amount of time, relative to first transaction observation', 'validation_rules': 'TO DETERMINE'},
        {'name': 'index', 'description': 'Index of the observations', 'validation_rules': 'Positive integer, unique'},
    ]

    _to_feature_store_logic(
        df=X, group_name='features',
        description='New version of Features',
        group_description=feature_descriptions,
        feature_group_version=1
    )

    # Target metadata and upload
    target_descriptions = [
    {'name': 'index', 'description': 'Index of the observations', 'validation_rules': 'Positive integer, unique'},
    {'name': 'class_', 'description': 'Predicted class of the observation. 1 for fraud, 0 otherwise', 'validation_rules': '0 or 1'}
    ]

    _to_feature_store_logic(
        df=y, group_name='target',
        description='New version of Target.',
        group_description=target_descriptions,
        feature_group_version=1
    )

    cprint('Features and Target uploaded.', color='green')
