from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, data_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs="preprocessed_data",
                outputs=["ref_data", "ana_data"],
                name="split_data",
            ),
            node(
                func=data_drift,
                inputs=["ref_data", "ana_data"],
                outputs="drift_result",
                name="data_drift",
                tags=['drift']
            ),
        ]
    )
