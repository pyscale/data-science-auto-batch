"""
This is an example of how to train this model using the tools provided in this repo
"""
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import tempfile

from pyskoptimize.base import MLPipelineStateModel


config = MLPipelineStateModel.parse_file("/app/config/examples/ml/elastic-binary-tree.json")

mlflow.set_experiment(experiment_name="Elastic Binary Tree")

mlflow.set_tracking_uri(os.environ['MLFLOW_DB'])

mlflow.autolog()
mlflow.xgboost.autolog()

with mlflow.start_run(run_name="Elastic Binary Tree") as run:

    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

    model = config.to_bayes_opt(verbose=3)

    with mlflow.start_run(nested=True):
        model.fit(
                X_train,
                y_train,
            )

    test_score = model.score(X_test, y_test)

    with mlflow.start_run(nested=True):
        mlflow.log_params(model.best_params_)
        mlflow.log_metrics({"train_mse": model.best_score_, "test_mse": test_score})

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.sklearn.save_model(model.best_estimator_, tmp_dir)

            mlflow.log_artifact(
                tmp_dir
            )
