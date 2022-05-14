FROM python:3.9

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

ENV MLFLOW_DB=sqlite:////app/mlflow/mydb.sqlite
ENV MLFLOW_ARTIFACT=/app/mlflow/mlruns

COPY Makefile .
COPY pyproject.toml .

RUN make init
RUN make install

COPY src .
COPY examples .
COPY config/examples/ml .