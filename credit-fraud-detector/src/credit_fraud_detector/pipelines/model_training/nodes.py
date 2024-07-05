from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import yaml

def train_models(X_train, y_train):
    classifiers = {
        "LogisiticRegression": LogisticRegression(),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    trained_models = {}
    for name, model in classifiers.items():
        with mlflow.start_run(run_name=f"training_{name}", nested=True):
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Log model parameters
            mlflow.log_params(model.get_params())
            
            # Save the model as an artifact
            mlflow.sklearn.log_model(model, artifact_path=name)
            
            # Set a tag for this run
            mlflow.set_tag("model_type", name)
            
    return trained_models
