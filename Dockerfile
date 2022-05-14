FROM python:3.9

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV MLFLOW_DB=sqlite:////app/mydb.sqlite
ENV MLFLOW_ARTIFACT=/app/mlruns

COPY Makefile .
COPY pyproject.toml .

RUN make install

COPY src .
