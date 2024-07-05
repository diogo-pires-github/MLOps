from kedro.pipeline import Pipeline, node
from .nodes import evaluate_models

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=evaluate_models,
                inputs=["X_test_data", "y_test_data", "tuned_models"],
                outputs="evaluation_reports",
                name="evaluate_models_node",
            ),
        ],
        tags='train'
    )
