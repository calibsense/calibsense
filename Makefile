PY ?= .venv/bin/python

.PHONY: test cov docs binary clean

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=caltrust --cov-report=term-missing --cov-report=html

docs:
	$(PY) -m pdoc caltrust -o docs/api --docformat google

binary:
	$(PY) -m PyInstaller --clean --noconfirm packaging/caltrust.spec

clean:
	rm -rf build dist docs/api htmlcov .coverage .pytest_cache
