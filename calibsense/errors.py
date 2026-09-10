# calibsense - measurement uncertainty for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

"""Exception hierarchy for calibsense.

Every error raised deliberately by this package derives from `CalibSenseError`, so
a caller — the CLI included — can separate "the user gave us something we cannot
work with" from "calibsense has a bug".
"""

from __future__ import annotations


class CalibSenseError(Exception):
    """Base class for every error calibsense raises on purpose."""


class ValidationError(CalibSenseError):
    """A value failed a precondition: wrong shape, wrong range, wrong count."""


class IngestError(CalibSenseError):
    """Input could not be turned into a calibration session."""


class DetectionError(IngestError):
    """A target detector could not run, or found nothing usable."""


class UnsupportedFormatError(IngestError):
    """No reader recognised the file, or a reader recognised it and choked."""


class SerializationError(CalibSenseError):
    """A session or fit could not be written or read back."""


class RefitError(CalibSenseError):
    """The calibration could not be re-estimated from the given observations."""


class DegenerateSystemError(RefitError):
    """The normal equations are rank deficient beyond what a pseudo-inverse fixes.

    Raised only when calibsense cannot produce *any* trustworthy number. A merely
    ill-conditioned system is not an error — it is the finding, and it is
    reported through the condition number instead.
    """
