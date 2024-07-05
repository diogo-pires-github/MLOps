from kedro.pipeline import Pipeline, node
from .nodes import select_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_model,
                inputs=["X_train_data", "y_train_data", "trained_models"],
                outputs="champion_model",
                name="model_selection_node",
            ),
        ],
        tags='model_selection'
    )
