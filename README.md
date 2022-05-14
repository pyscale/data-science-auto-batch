# {{cookiecutter.project_name}}

This is a project that is set to:

    1. Extract Data
    2. Engineer Data (i.e. joins and filters)
    3. Train a ML Pipeline and optimize the hyperparameter
    4. Ease the transition from Training into Deployments

## Installation

Just run `make init` and follow that with `make install`.  This will install everything you need.

## How To Use

There is an example of how to run this end to end with pre-engineered data under the `examples` folder. 

Docker is there as an option if you need to containerize your solution for training. 

For information regarding how to integrate with mlflow, please go to the `data/mlflow` folder. 

## Deployment

If you trained your model and used MLFlow, then follow these linked commands:
    
    1. https://www.mlflow.org/docs/latest/cli.html#mlflow-models-build-docker
