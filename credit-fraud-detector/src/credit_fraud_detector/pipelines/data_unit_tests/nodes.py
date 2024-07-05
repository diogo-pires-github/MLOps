import logging
from typing import Any, Dict, Tuple

import pandas as pd
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)

def get_validation_results(checkpoint_result):
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')

    df_validation = pd.DataFrame(columns=["Success", "Expectation Type", "Column", "Column Pair", "Max Value",
                                          "Min Value", "Element Count", "Unexpected Count", "Unexpected Percent",
                                          "Value Set", "Unexpected Value", "Observed Value"])

    for result in results:
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')
        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        unexpected_value = [item for item in observed_value if item not in value_set] if isinstance(observed_value, list) else []

        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict([{
            "Success": success,
            "Expectation Type": expectation_type,
            "Column": column,
            "Column Pair": (column_A, column_B),
            "Max Value": max_value,
            "Min Value": min_value,
            "Element Count": element_count,
            "Unexpected Count": unexpected_count,
            "Unexpected Percent": unexpected_percent,
            "Value Set": value_set,
            "Unexpected Value": unexpected_value,
            "Observed Value": observed_value
        }])], ignore_index=True)

    return df_validation

def test_data(df):
    context = gx.get_context(context_root_dir="src/credit_fraud_detector/pipelines/data_unit_tests/gx")
    datasource_name = "creditcard_datasource"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    suite_creditcard = context.add_or_update_expectation_suite(expectation_suite_name="CreditCard")

    # Define expectations for the dataset
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "scaled_time",
                "min_value": 0,
                "max_value": df["scaled_time"].max()
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "scaled_amount",
                "min_value": df["scaled_amount"].min(), 
                "max_value": df["scaled_amount"].max()
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "Class",
                "value_set": [0, 1]
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "scaled_time"
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "scaled_amount"
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "Class"
            }
        ),
    ]

    for column in df.columns:
        if column.startswith("V"):
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={
                        "column": column
                    }
                )
            )
            expectations.append(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": column,
                        "min_value": df[column].min(),
                        "max_value": df[column].max()
                    }
                )
            )

    for expectation in expectations:
        suite_creditcard.add_expectation(expectation_configuration=expectation)

    context.add_or_update_expectation_suite(expectation_suite=suite_creditcard)

    data_asset_name = "test"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_creditcard",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "CreditCard",
            },
        ],
    )
    checkpoint_result = checkpoint.run()

    df_validation = get_validation_results(checkpoint_result)

    pd_df_ge = gx.from_pandas(df)

    assert pd_df_ge.expect_column_values_to_be_of_type("scaled_time", "float64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("scaled_amount", "float64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("Class", "int64").success == True

    log = logging.getLogger(__name__)
    log.info("Data passed the unit data tests")

    return df_validation

'''
if __name__ == "__main__":
    # Load the data
    data_path = "/mnt/data/creditcard.csv"
    df = pd.read_csv(data_path)

    # Run the data tests
    df_validation = test_data(df)

    # Print validation results
    print(df_validation)
'''