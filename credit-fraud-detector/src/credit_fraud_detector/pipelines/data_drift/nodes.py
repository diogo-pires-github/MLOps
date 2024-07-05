import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
import logging
import nannyml as nml
import matplotlib.pyplot as plt 
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from credit_fraud_detector.pipelines.data_drift.utils import calculate_psi

# Define logging
logger = logging.getLogger(__name__)

def introduce_drift(data: pd.DataFrame) -> pd.DataFrame:
    data_with_drift = data.copy()
    # Example: Shift the scaled_amount feature by adding a constant value
    data_with_drift['scaled_amount'] = data_with_drift['scaled_amount'] + np.random.normal(loc=2.0, scale=0.5, size=len(data_with_drift))
    # Example: Shift the scaled_time feature by multiplying it by a factor
    data_with_drift['scaled_time'] = data_with_drift['scaled_time'] * np.random.uniform(1.5, 2.0, size=len(data_with_drift))
    return data_with_drift

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference_data = data.sample(frac=0.5, random_state=42)
    analysis_data = data.drop(reference_data.index)
    analysis_data_with_drift = introduce_drift(analysis_data)
    return reference_data, analysis_data_with_drift


def plot_psi(train: pd.DataFrame, test: pd.DataFrame, numerical_features: list):
    psis_num = []

    for feature_name in numerical_features:
        psi = calculate_psi(train[feature_name], test[feature_name], buckettype='bins', buckets=50, axis=0)
        psis_num.append(psi)

    # Plot each feature's PSI value
    height = psis_num
    bars = numerical_features
    y_pos = np.arange(len(bars))
    
    plt.figure(figsize=(6, 4))
    plt.barh(y_pos, height)
    plt.axvline(x=0.2, color='red')
    plt.yticks(y_pos, bars)
    plt.xlabel("PSI")
    plt.title("Population Stability Index (PSI) for Numerical Features")
    plt.tight_layout()
    plt.savefig("data/08_reporting/psi_plot.png")
    plt.show()

    return psis_num

def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame) -> pd.DataFrame:
    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    constant_threshold.thresholds(data_reference)

    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=["scaled_amount", "scaled_time"],
        treat_as_categorical=[],
        chunk_size=50,
        categorical_methods=[],
        thresholds={"jensen_shannon": constant_threshold}
    )

    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['scaled_amount', 'scaled_time'], methods=['jensen_shannon']).to_df()

    # Save the NannyML drift plot as HTML
    figure = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['scaled_amount', 'scaled_time'], methods=['jensen_shannon']).plot(kind='drift')
    figure.write_html("data/08_reporting/univariate_nml.html")

    # Generate and save the Evidently report as HTML
    data_drift_report = Report(metrics=[DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])
    data_drift_report.run(current_data=data_analysis[["scaled_amount", "scaled_time"]], reference_data=data_reference[["scaled_amount", "scaled_time"]], column_mapping=None)
    data_drift_report.save_html("data/08_reporting/data_drift_report.html")
    
    # Save individual Evidently plots
    data_drift_report.save_json("data/08_reporting/data_drift_report.json")
    
    # Calculate and plot PSI for numerical features
    numerical_features = ["scaled_amount", "scaled_time"]
    plot_psi(data_reference, data_analysis, numerical_features)
    
    return results