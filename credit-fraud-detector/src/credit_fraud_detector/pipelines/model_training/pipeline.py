from kedro.pipeline import Pipeline, node
from .nodes import train_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_models,
                inputs=["X_train_data", "y_train_data"],
                outputs="trained_models",
                name="train_models_node",
            ),
        ],
        tags='train'
    )
