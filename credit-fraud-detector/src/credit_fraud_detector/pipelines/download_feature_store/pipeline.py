from kedro.pipeline import Pipeline, node
from .nodes import download_from_feature_store

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=download_from_feature_store,
            inputs=None,
            outputs='feature_data',
            name="download_feature_store",
        ),
    ], tags='train')
