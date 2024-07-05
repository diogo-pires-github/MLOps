from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def evaluate_models(X_test, y_test, models):
    evaluation_reports = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"evaluation_{name}", nested=True):
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            evaluation_reports[name] = report
            
            # Log evaluation metrics
            for metric, value in report.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        mlflow.log_metric(f"{metric}_{sub_metric}", sub_value)
                else:
                    mlflow.log_metric(metric, value)
                    
            # Set a tag for this run
            mlflow.set_tag("evaluation", name)
            
    return evaluation_reports
