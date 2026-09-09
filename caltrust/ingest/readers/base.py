"""The reader contract for existing calibration files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ...core.session import CalibrationRecord

#: Bytes of a file offered to `sniff`. Enough for any header worth matching.
SNIFF_BYTES = 4096


class CalibrationReader(ABC):
    """Reads one third-party calibration format.

    A reader is asked to recognise a file before it is asked to parse it, so
    that an unrecognised file produces "no reader matched" rather than the
    parse error of whichever reader happened to be tried last.
    """

    #: Short name used in `--format` and in provenance strings.
    name: ClassVar[str] = ""
    #: One line, shown by the CLI when listing supported formats.
    description: ClassVar[str] = ""

    @abstractmethod
    def sniff(self, path: str, head: str) -> bool:
        """Decide whether this reader should attempt the file.

        Args:
            path: The file path, for extension checks.
            head: The first `SNIFF_BYTES` bytes decoded as text, errors ignored.

        Returns:
            `True` if this reader recognises the format.
        """

    @abstractmethod
    def read(self, path: str) -> CalibrationRecord:
        """Parse the file.

        Args:
            path: The file to read.

        Returns:
            The calibration it describes.

        Raises:
            UnsupportedFormatError: The file matched but could not be parsed.
        """
