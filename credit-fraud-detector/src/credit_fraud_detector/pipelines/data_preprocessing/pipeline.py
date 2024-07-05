from kedro.pipeline import Pipeline, node
from .nodes import load_and_preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_preprocess_data,
            inputs="params:filepath",
            outputs="preprocessed_data",
            name="load_and_preprocess_data_node",
        ),
    ], tags=['data', 'drift'])
