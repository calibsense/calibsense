PY ?= .venv/bin/python

# pdoc documents exactly the modules it is handed, so the list is derived from
# the tree rather than maintained by hand. `__main__` and `_version` carry no
# API worth a page. pdoc warns for each module reachable both directly and
# through a package that re-exports it; one page per module is still produced.
MODULES = $(shell find calibsense -name '*.py' ! -name '__main__.py' ! -name '_version.py' \
	| sed -e 's/\.py$$//' -e 's|/__init__$$||' | tr '/' '.' | sort -u)

.PHONY: test fast cov docs binary clean

test:
	$(PY) -m pytest

# Skips the Monte Carlo and image-rendering tests; use before a commit, not
# instead of `make test`.
fast:
	$(PY) -m pytest -m "not slow"

cov:
	$(PY) -m pytest --cov=calibsense --cov-report=term-missing --cov-report=html

docs:
	$(PY) -m pdoc $(MODULES) -o docs/api --docformat google

binary:
	$(PY) -m PyInstaller --clean --noconfirm packaging/calibsense.spec

clean:
	rm -rf build dist docs/api htmlcov .coverage .pytest_cache
