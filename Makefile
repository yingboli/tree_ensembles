.PHONY: env lint format typecheck test check

env:        ## create the conda env
	conda env create -f environment.yml

lint:       ## check style without changing files
	ruff check .
	ruff format --check .

format:     ## auto-fix style
	ruff format .
	ruff check --fix .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test
