.PHONY: env kernel lint format typecheck test check

ENV := tree_ensembles

env:        ## create the conda env and register its Jupyter kernel
	conda env create -f environment.yml
	$(MAKE) kernel

kernel:     ## register (or refresh) the env as the Jupyter kernel "tree_ensembles"
	conda run -n $(ENV) python -m ipykernel install --user --name $(ENV) --display-name $(ENV)

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
