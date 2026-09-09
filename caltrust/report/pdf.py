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

"""A minimal PDF writer.

Written rather than depended upon. caltrust ships with three runtime
dependencies so that it can be frozen into one binary, and a PDF of text, rules
and bars needs nothing that is not already here: the format is plain bytes, the
base-14 fonts are guaranteed present in every viewer and need no embedding, and
`zlib` is in the standard library.

What it supports is deliberately narrow — left, right and centred text in three
fonts, horizontal rules, filled rectangles, and multi-page flow with automatic
breaks. That is enough for a report with headings, paragraphs, tables and bar
charts, and stopping there is what keeps it under a few hundred lines.

Glyph widths for Helvetica and Helvetica-Bold are the standard Adobe metrics,
which is what makes wrapping and right-alignment come out right; Courier is
monospaced at 600 units.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..errors import ValidationError

#: Points per inch, the PDF unit.
POINTS_PER_INCH = 72.0

#: A4 in points, which is what a European quality department prints on.
A4 = (595.28, 841.89)

#: Adobe Helvetica advance widths for ASCII 32-126, in 1/1000 em.
_HELVETICA = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
)

#: Adobe Helvetica-Bold advance widths for ASCII 32-126, in 1/1000 em.
_HELVETICA_BOLD = (
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
)

#: The three base-14 fonts this writer exposes.
FONTS = {
    "regular": ("Helvetica", _HELVETICA),
    "bold": ("Helvetica-Bold", _HELVETICA_BOLD),
    "mono": ("Courier", None),
}


def text_width(text: str, size: float, font: str = "regular") -> float:
    """Width of a string when set in one of the available fonts.

    Args:
        text: The string to measure.
        size: Font size in points.
        font: `"regular"`, `"bold"` or `"mono"`.

    Returns:
        The advance width in points.

    Raises:
        ValidationError: The font name is unknown.
    """
    if font not in FONTS:
        raise ValidationError(f"unknown font {font!r}; have {sorted(FONTS)}")
    _, widths = FONTS[font]
    if widths is None:
        return len(text) * 0.6 * size
    total = 0
    for character in text:
        code = ord(character)
        total += widths[code - 32] if 32 <= code <= 126 else widths[ord("?") - 32]
    return total * size / 1000.0


def wrap(text: str, width: float, size: float, font: str = "regular") -> List[str]:
    """Break a paragraph into lines that fit a column.

    Args:
        text: The paragraph.
        width: Column width in points.
        size: Font size in points.
        font: Font name.

    Returns:
        The lines, never empty — an unbreakable long word gets its own line
        rather than being dropped.
    """
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(candidate, size, font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def escape(text: str) -> bytes:
    """Encode and escape a string for a PDF literal.

    The fonts are declared `/WinAnsiEncoding`, which is CP1252, so that is what
    the bytes have to be. Encoding to Latin-1 instead would turn an em dash or a
    curly quote into a question mark, which is exactly the kind of small wrong
    thing a printed report should not contain.

    Args:
        text: The string to encode.

    Returns:
        CP1252 bytes with backslashes and parentheses escaped. Characters CP1252
        cannot represent become `?`, because a base-14 font has no glyph for
        them either.
    """
    encoded = text.encode("cp1252", errors="replace")
    for source, target in ((b"\\", b"\\\\"), (b"(", b"\\("), (b")", b"\\)")):
        encoded = encoded.replace(source, target)
    return encoded


@dataclass
class Page:
    """One page's content stream, built up as drawing operators."""

    width: float
    height: float
    operators: List[str] = field(default_factory=list)

    def stream(self) -> bytes:
        """The page's content stream, deflated."""
        return zlib.compress("\n".join(self.operators).encode("cp1252"))


class Document:
    """A multi-page PDF being built up.

    Coordinates are in points with the origin at the top-left and `y` growing
    downwards, which is how a report is written; the PDF's own bottom-left
    origin is handled on the way out.

    Attributes:
        size: Page size as `(width, height)` in points.
        margin: Margin on all four sides, in points.
        title: Document title, written into the PDF metadata.
    """

    def __init__(
        self,
        size: Tuple[float, float] = A4,
        margin: float = 56.0,
        title: str = "caltrust report",
    ):
        self.size = size
        self.margin = margin
        self.title = title
        self.pages: List[Page] = []
        self._page: Optional[Page] = None
        self.y = margin
        self.new_page()

    @property
    def content_width(self) -> float:
        """Usable width between the margins."""
        return self.size[0] - 2 * self.margin

    @property
    def bottom(self) -> float:
        """The `y` at which content must stop."""
        return self.size[1] - self.margin

    def new_page(self) -> None:
        """Start a new page and reset the cursor to the top margin."""
        self._page = Page(self.size[0], self.size[1])
        self.pages.append(self._page)
        self.y = self.margin

    def space(self, amount: float) -> None:
        """Advance the cursor, breaking the page if it would overflow.

        Args:
            amount: Points to advance.
        """
        if self.y + amount > self.bottom:
            self.new_page()
        else:
            self.y += amount

    def ensure(self, amount: float) -> None:
        """Break the page unless `amount` points remain.

        Args:
            amount: Points the next block needs.
        """
        if self.y + amount > self.bottom:
            self.new_page()

    def text(
        self,
        content: str,
        size: float = 9.5,
        font: str = "regular",
        x: Optional[float] = None,
        align: str = "left",
        colour: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        advance: bool = True,
        leading: float = 1.35,
    ) -> None:
        """Draw one line of text at the cursor.

        Args:
            content: The line. Never wrapped; use `paragraph` for that.
            size: Font size in points.
            font: `"regular"`, `"bold"` or `"mono"`.
            x: Left edge, or `None` for the margin.
            align: `"left"`, `"right"` or `"centre"`, relative to `x` and the
                content width.
            colour: RGB in `[0, 1]`.
            advance: Move the cursor down by one line afterwards.
            leading: Line spacing as a multiple of the font size.
        """
        self.ensure(size * leading)
        name, _ = FONTS[font]
        left = self.margin if x is None else x
        if align == "right":
            left = (self.size[0] - self.margin) - text_width(content, size, font)
        elif align == "centre":
            left = (self.size[0] - text_width(content, size, font)) / 2.0
        baseline = self.size[1] - (self.y + size)
        self._page.operators.append(
            f"BT /F{name} {size:.2f} Tf {colour[0]:.3f} {colour[1]:.3f} "
            f"{colour[2]:.3f} rg 1 0 0 1 {left:.2f} {baseline:.2f} Tm "
            f"({escape(content).decode('cp1252')}) Tj ET"
        )
        if advance:
            self.y += size * leading

    def paragraph(
        self,
        content: str,
        size: float = 9.5,
        font: str = "regular",
        indent: float = 0.0,
        colour: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        leading: float = 1.35,
    ) -> None:
        """Draw a wrapped paragraph.

        Args:
            content: The paragraph text.
            size: Font size in points.
            font: Font name.
            indent: Left indent in points.
            colour: RGB in `[0, 1]`.
            leading: Line spacing as a multiple of the font size.
        """
        for line in wrap(content, self.content_width - indent, size, font):
            self.text(
                line, size, font, x=self.margin + indent, colour=colour, leading=leading
            )

    def rule(
        self,
        thickness: float = 0.6,
        colour: Tuple[float, float, float] = (0.75, 0.75, 0.75),
        indent: float = 0.0,
    ) -> None:
        """Draw a horizontal rule across the content width.

        Args:
            thickness: Line width in points.
            colour: RGB in `[0, 1]`.
            indent: Left and right inset in points.
        """
        self.ensure(thickness + 2)
        y = self.size[1] - self.y
        self._page.operators.append(
            f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} RG "
            f"{thickness:.2f} w {self.margin + indent:.2f} {y:.2f} m "
            f"{self.size[0] - self.margin - indent:.2f} {y:.2f} l S"
        )
        self.y += thickness + 2

    def bar(
        self,
        x: float,
        width: float,
        height: float,
        colour: Tuple[float, float, float] = (0.2, 0.35, 0.6),
    ) -> None:
        """Fill a rectangle whose top-left is at `(x, cursor)`.

        Args:
            x: Left edge in points.
            width: Width in points; a non-positive width draws nothing.
            height: Height in points.
            colour: RGB fill in `[0, 1]`.
        """
        if width <= 0 or height <= 0:
            return
        y = self.size[1] - (self.y + height)
        self._page.operators.append(
            f"{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f} rg "
            f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re f"
        )

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        widths: Sequence[float],
        size: float = 8.5,
        aligns: Optional[Sequence[str]] = None,
    ) -> None:
        """Draw a table with a ruled header.

        Args:
            headers: Column titles.
            rows: Cell text, already formatted.
            widths: Column widths in points.
            size: Font size in points.
            aligns: Per-column `"left"` or `"right"`; defaults to left for the
                first column and right for the rest, which is what a table of
                numbers wants.

        Raises:
            ValidationError: The column counts disagree.
        """
        if len(headers) != len(widths):
            raise ValidationError(
                f"{len(headers)} headers against {len(widths)} column widths"
            )
        if not rows:
            # A header with nothing under it is noise, not information.
            return
        alignment = list(aligns) if aligns else ["left"] + ["right"] * (len(headers) - 1)
        line_height = size * 1.5

        def draw(cells: Sequence[str], font: str) -> None:
            self.ensure(line_height)
            left = self.margin
            for cell, width, align in zip(cells, widths, alignment):
                offset = (
                    width - text_width(str(cell), size, font) if align == "right" else 0.0
                )
                self.text(
                    str(cell), size, font, x=left + offset, advance=False
                )
                left += width
            self.y += line_height

        draw(headers, "bold")
        self.rule(0.5, (0.6, 0.6, 0.6))
        for row in rows:
            if len(row) != len(headers):
                raise ValidationError(
                    f"row has {len(row)} cells against {len(headers)} headers"
                )
            draw(row, "regular")

    def render(self) -> bytes:
        """Serialise the document.

        Returns:
            The complete PDF file as bytes.
        """
        objects: List[bytes] = []

        def add(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        font_ids: Dict[str, int] = {}
        for key, (name, _) in FONTS.items():
            font_ids[name] = add(
                b"<< /Type /Font /Subtype /Type1 /BaseFont /"
                + name.encode("latin-1")
                + b" /Encoding /WinAnsiEncoding >>"
            )
        resources = (
            b"<< /Font << "
            + b" ".join(
                b"/F" + name.encode("latin-1") + b" " + str(ident).encode("latin-1") + b" 0 R"
                for name, ident in font_ids.items()
            )
            + b" >> >>"
        )

        pages_id = len(objects) + 1 + 2 * len(self.pages)
        page_ids: List[int] = []
        for page in self.pages:
            stream = page.stream()
            content_id = add(
                b"<< /Length "
                + str(len(stream)).encode("latin-1")
                + b" /Filter /FlateDecode >>\nstream\n"
                + stream
                + b"\nendstream"
            )
            page_ids.append(
                add(
                    b"<< /Type /Page /Parent "
                    + str(pages_id).encode("latin-1")
                    + b" 0 R /MediaBox [0 0 "
                    + f"{page.width:.2f} {page.height:.2f}".encode("latin-1")
                    + b"] /Resources "
                    + resources
                    + b" /Contents "
                    + str(content_id).encode("latin-1")
                    + b" 0 R >>"
                )
            )
        tree_id = add(
            b"<< /Type /Pages /Count "
            + str(len(page_ids)).encode("latin-1")
            + b" /Kids ["
            + b" ".join(str(i).encode("latin-1") + b" 0 R" for i in page_ids)
            + b"] >>"
        )
        info_id = add(
            b"<< /Title (" + escape(self.title) + b") "
            b"/Producer (caltrust) >>"
        )
        catalog_id = add(
            b"<< /Type /Catalog /Pages " + str(tree_id).encode("latin-1") + b" 0 R >>"
        )

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: List[int] = []
        for index, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += str(index).encode("latin-1") + b" 0 obj\n" + body + b"\nendobj\n"
        start = len(out)
        out += b"xref\n0 " + str(len(objects) + 1).encode("latin-1") + b"\n"
        out += b"0000000000 65535 f \n"
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode("latin-1")
        out += (
            b"trailer\n<< /Size "
            + str(len(objects) + 1).encode("latin-1")
            + b" /Root "
            + str(catalog_id).encode("latin-1")
            + b" 0 R /Info "
            + str(info_id).encode("latin-1")
            + b" 0 R >>\nstartxref\n"
            + str(start).encode("latin-1")
            + b"\n%%EOF\n"
        )
        return bytes(out)
