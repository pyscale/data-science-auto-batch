

init:
	pip install --upgrade pip
	pip install --upgrade poetry

install:
	poetry install

test:
	pytest

mlflow.server.up:
	mlflow server --backend-store-uri sqlite:///data/mlflow/db/mydb.sqlite --default-artifact-root data/mlflow/mlruns