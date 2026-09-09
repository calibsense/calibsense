"""Entry point for `python -m caltrust`, and for the PyInstaller build.

The import is absolute rather than relative on purpose. PyInstaller runs this
file as a top-level script with no parent package, so a relative import fails at
run time *and* leaves the dependency graph empty at build time -- producing a
binary that builds cleanly, weighs a few megabytes, and cannot start.
"""

from __future__ import annotations

import sys

from caltrust.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
