
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node
from .nodes import split_data, split_data2

# def create_pipeline(**kwargs) -> Pipeline:
#     return Pipeline([
#         node(
#             func=split_data,
#             inputs="preprocessed_data",
#             outputs=["original_Xtrain", "original_Xtest", "original_ytrain", "original_ytest"],
#             name="split_data_node",
#         ),
#     ])

def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline([
        node(
            func=split_data2,
            inputs=["feature_data", "parameters"],
            outputs= ["X_train_data","X_test_data","y_train_data","y_test_data"],
            name="split_data_node",
        ),
    ], tags='train')
