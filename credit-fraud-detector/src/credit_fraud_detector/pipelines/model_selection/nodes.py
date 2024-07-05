from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score
import mlflow
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_model(X_train, y_train, models):
    params = {
        "LogisiticRegression": {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        "KNearest": {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
        "Support Vector Classifier": {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']},
        "DecisionTreeClassifier": {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
    }

    best_recall_score = 0
    champion_model = None

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.Loader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    for name, model in models.items():
        logger.info(f"Tuning model: {name}")
        if name not in params:
            logger.error(f"No parameters defined for model: {name}")
            continue

        with mlflow.start_run(experiment_id=experiment_id, run_name=f"tuning_{name}", nested=True) as run:
            mlflow.sklearn.autolog()  # Enable autologging for sklearn models

            try:
                grid = GridSearchCV(model, params[name])
                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                # Log best parameters and performance metrics
                mlflow.log_params(grid.best_params_)
                mlflow.log_metric("best_score", grid.best_score_)

                # Log the tuned model as an artifact
                mlflow.sklearn.log_model(best_model, artifact_path=f"tuned_{name}")

                # Log custom tags or additional information
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("tuning", "GridSearchCV")

                # Set tags for model version (example: experiment name)
                mlflow.set_tag(name, "experiment", "tuning_experiment")

                client = mlflow.MlflowClient()
                try:
                    client.create_registered_model(name)
                except:
                    client.get_registered_model(name)
                model_version = client.create_model_version(name, "champion_model_alias", run.info.run_id)
                # Set registered model alias (example: champion model alias)
                client.set_registered_model_alias(name, "champion_model_alias", model_version.version)

                # Optionally, log specific metrics on training data (example: training accuracy)
                y_train_pred = best_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                mlflow.log_metric("train_accuracy", train_accuracy)

                # Calculate and log recall score
                recall = recall_score(y_train, y_train_pred, average='macro')
                mlflow.log_metric("train_recall", recall)

                # Update the champion model based on recall score
                if recall > best_recall_score:
                    best_recall_score = recall
                    champion_model = best_model

            except Exception as e:
                logger.error(f"Error tuning model {name}: {str(e)}")

    return champion_model
