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

"""M7 — the HTML page a quality manager opens instead of a terminal."""

from __future__ import annotations

import json
import re

import pytest

from calibsense.core.session import CalibrationSession
from calibsense.errors import SerializationError
from calibsense.report import (
    ReportMetadata,
    render_html,
    render_json,
    run_audit,
    write_html,
)
from calibsense.report.html import _page

from . import rigs

_PAYLOAD = re.compile(
    r'<script type="application/json" id="calibsense-audit">(.*?)</script>', re.S
)


def embedded(page: str):
    """The JSON a page carries, parsed the way the browser will parse it."""
    match = _PAYLOAD.search(page)
    assert match, "the page carries no report payload"
    return json.loads(match.group(1))


def test_the_empty_viewer_carries_no_report():
    assert embedded(render_html(None)) is None


def test_the_page_fetches_nothing_from_the_network():
    """It gets opened on plant PCs with no route out, so nothing may be remote."""
    page = render_html(None)
    # The SVG namespace is an identifier, not a request.
    stripped = page.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in stripped and "https://" not in stripped
    for pattern in (r"<link\b", r"<script[^>]+src=", r"@import", r"\bfetch\(", r"XMLHttpRequest"):
        assert not re.search(pattern, page), pattern


def test_free_text_cannot_close_the_payload_element():
    hostile = "</script><script>alert(1)</script> & <b>"
    payload = json.dumps({"metadata": {"camera_name": hostile}})
    page = _page(payload, hostile)
    assert embedded(page)["metadata"]["camera_name"] == hostile
    assert "<script>alert(1)" not in page
    assert "<title>&lt;/script&gt;" in page


@pytest.fixture(scope="module")
def audit():
    capture = rigs.with_bad_view()
    session = CalibrationSession(observations=capture.observations)
    return run_audit(
        session, n_samples=300, forecast_samples=200,
        metadata=ReportMetadata(camera_name="line-3 </script> gauge", contact="qa@example.com"),
    )


@pytest.mark.slow
def test_the_page_carries_exactly_what_the_json_does(audit):
    assert embedded(render_html(audit)) == json.loads(render_json(audit))


@pytest.mark.slow
def test_the_page_is_titled_after_the_camera(audit):
    page = render_html(audit)
    assert "<title>Camera calibration measurement audit - line-3 &lt;/script&gt; gauge</title>" in page


@pytest.mark.slow
def test_write_html_writes_a_page(audit, tmp_path):
    path = tmp_path / "audit.html"
    assert write_html(audit, str(path)) == str(path)
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert embedded(page)["metadata"]["contact"] == "qa@example.com"


def test_an_unwritable_destination_is_a_serialization_error(tmp_path):
    with pytest.raises(SerializationError):
        write_html(None, str(tmp_path / "missing" / "audit.html"))
