"""Exception hierarchy for caltrust.

Every error raised deliberately by this package derives from `CalTrustError`, so
a caller — the CLI included — can separate "the user gave us something we cannot
work with" from "caltrust has a bug".
"""

from __future__ import annotations


class CalTrustError(Exception):
    """Base class for every error caltrust raises on purpose."""


class ValidationError(CalTrustError):
    """A value failed a precondition: wrong shape, wrong range, wrong count."""


class IngestError(CalTrustError):
    """Input could not be turned into a calibration session."""


class DetectionError(IngestError):
    """A target detector could not run, or found nothing usable."""


class UnsupportedFormatError(IngestError):
    """No reader recognised the file, or a reader recognised it and choked."""


class SerializationError(CalTrustError):
    """A session or fit could not be written or read back."""


class RefitError(CalTrustError):
    """The calibration could not be re-estimated from the given observations."""


class DegenerateSystemError(RefitError):
    """The normal equations are rank deficient beyond what a pseudo-inverse fixes.

    Raised only when caltrust cannot produce *any* trustworthy number. A merely
    ill-conditioned system is not an error — it is the finding, and it is
    reported through the condition number instead.
    """
