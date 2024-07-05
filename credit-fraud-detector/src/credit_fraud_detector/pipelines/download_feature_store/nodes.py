import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
from typing import Union
from operator import attrgetter

load_dotenv()

def _get_features(
    group_name: str,
    version: Union[int, None] = None
):
    '''
    This function takes in the group name of the desired features in Hopsworks and returns a pd.DataFrame with them.

    Args:
        - group_name (str): Name of the feature group.
        - version (int | None): Version number of feature group. If None, latest is returned

    Returns:
        - A pd.DataFrame with the features.
    '''
    project_name = os.environ.get('FS_PROJECT_NAME')
    api_key = os.environ.get('FS_API_KEY')

    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    if version:
        features = fs.get_feature_group(name=group_name, version=version)
    else:
        # Get a list with all the versions and chooses the latest
        features = max(fs.get_feature_groups(name=group_name), key=attrgetter('version'))

    df = features.read()

    return df


def download_from_feature_store():
    '''
    This function gets the features and targets from the Feature Store and returns a pd.DataFrame.

    Args:
        - None

    Returns:
        - A pd.DataFrame with the expected columns for the next pipeline steps.
    '''
    X = _get_features(group_name='features')
    y = _get_features(group_name='target')

    df = pd.merge(X, y, how='inner', on='index')
    df = df.set_index('index').sort_index()
    df = df.rename(columns=lambda x: x.upper() if 'v' in x else 'Class' if x == 'class_' else x)

    return df
