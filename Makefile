##
# marmots
#
# @file
# @version 0.0.1

# our testing targets
.PHONY: tests flake black mypy all

all: mypy isort black flake tests

tests:
	python3 -m pytest --cov=marmots tests

flake:
	python3 -m flake8 marmots

black:
	python3 -m black -t py37 marmots tests

# mypy:
	# python3 -m mypy marmots

isort:
	python3 -m isort --atomic -rc -y poinsseta marmots

# end
