# Makefile to help automate key steps

.DEFAULT_GOAL := help


# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: checks
checks:  ## run all the linting checks of the codebase
	@echo "=== black docs ==="; poetry run blacken-docs --check book/notebooks/*.md || echo "--- black docs failed ---" >&2; \
		echo "======"

.PHONY: docs
docs:  ## build the docs (also acts as a testing step because of assertions in the notebooks)
	poetry run jupyter-book build book

.PHONY: black-docs
black-docs:  ## format the notebok examples using black
	poetry run blacken-docs book/notebooks/*.md

.PHONY: check-commit-messages
check-commit-messages:  ## check commit messages
	poetry run cz check --rev-range 6ecf76a3..HEAD

virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	# Put virtual environments in the project
	poetry config virtualenvs.in-project true
	poetry install
