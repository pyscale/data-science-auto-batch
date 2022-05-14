

init:
	pip install --upgrade pip
	pip install --upgrade poetry

install:
	poetry install

test:
	pytest

