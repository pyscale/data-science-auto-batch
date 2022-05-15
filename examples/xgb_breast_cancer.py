"""
This is an example of how to train this model using the tools provided in this repo
"""
import os
import tempfile

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import plotly.express as px
import mlflow

from pyskoptimize.base import MLPipelineStateModel


config = MLPipelineStateModel.parse_file("config/examples/ml/xgb-classification.json")

mlflow.set_experiment(experiment_name="XGB Clf")

mlflow.set_tracking_uri(os.environ.get('MLFLOW_DB', 'sqlite:///data/mlflow/db/mydb.sqlite'))

mlflow.autolog(silent=True)
mlflow.xgboost.autolog()

cal_housing = load_breast_cancer()
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0, stratify=y)

model = config.to_bayes_opt(verbose=3)

model.fit(
    X_train,
    y_train,
)

test_score = model.score(X_test, y_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

with mlflow.start_run(run_name="XGB Clf") as run:
    mlflow.log_params(model.best_params_)
    mlflow.log_metrics({"train_f1": model.best_score_, "test_f1": test_score})

    train_fig = px.imshow(
        confusion_matrix(y_train, train_pred, normalize="true")
    )

    test_fig = px.scatter(
        confusion_matrix(y_test, test_pred, normalize="true")
    )

    mlflow.log_figure(train_fig, "train_results.html")
    mlflow.log_figure(train_fig, "test_results.html")

    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.sklearn.save_model(model.best_estimator_, tmp_dir)

        mlflow.log_artifact(
            tmp_dir
        )
