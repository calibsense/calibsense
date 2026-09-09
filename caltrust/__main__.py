# caltrust - metric trust for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

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
