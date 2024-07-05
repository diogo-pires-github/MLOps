from kedro.pipeline import Pipeline, node
from .nodes import upload_to_feature_store

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=upload_to_feature_store,
            inputs="preprocessed_data",
            outputs=None,
            name="upload_feature_store",
            tags='data'
        ),
    ])
