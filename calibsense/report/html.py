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

"""The report as one HTML file a quality manager can double-click.

The PDF is for signing and the JSON is for machines; this is for the person in
between, who needs to read the result, check it against their own tolerance, and
never open a terminal to do it. The page is fully self-contained — no network,
no fonts or scripts fetched from anywhere — because the PC it gets opened on is
often on a plant network with no route out.

The audit's JSON is embedded in the page and rendered in the browser, so the
page shows exactly the numbers `write_json` would have written. The same page
also opens any other `audit.json` dropped onto it, which means one copy of the
file is enough for a quality manager to read every report the engineers send.

The template lives in this module rather than beside it as a `.html` file,
because the package ships no data files and so needs no frozen-path handling
when built into a single binary.
"""

from __future__ import annotations

from typing import Optional

from ..errors import SerializationError
from .audit import Audit
from .render import render_json

#: Where the audit's JSON goes in the template.
_PAYLOAD = "__CALIBSENSE_AUDIT_JSON__"


def _embed(payload: str) -> str:
    """Make JSON text safe to place inside a `<script>` element.

    A camera name or contact line is free text, so it can contain `</script>`.
    Escaping `<`, `>` and `&` as JSON unicode escapes keeps the text identical
    once parsed while making it impossible for it to end the element early.
    """
    return payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _page(payload: str, title: str) -> str:
    escaped = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return _TEMPLATE.replace("__CALIBSENSE_TITLE__", escaped).replace(
        _PAYLOAD, _embed(payload)
    )


def render_html(audit: Optional[Audit]) -> str:
    """Render an audit as a self-contained HTML page.

    Args:
        audit: The audit to render, or `None` for an empty viewer that asks for
            an `audit.json` to be opened.

    Returns:
        The complete HTML document.
    """
    if audit is None:
        return _page("null", "calibsense report viewer")
    metadata = audit.metadata
    return _page(render_json(audit, indent=0), f"{metadata.title} - {metadata.camera_name}")


def write_html(audit: Optional[Audit], path: str) -> str:
    """Write an audit as a self-contained HTML page.

    Args:
        audit: The audit to render, or `None` for an empty viewer.
        path: Destination path.

    Returns:
        The path written.

    Raises:
        SerializationError: The file could not be written.
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(render_html(audit))
    except OSError as exc:
        raise SerializationError(f"could not write {path}: {exc}") from exc
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__CALIBSENSE_TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f4f4f2;
  --surface: #fcfcfb;
  --sunken: #efefec;
  --line: #deddd8;
  --ink: #161615;
  --ink-2: #52514e;
  --ink-3: #7a7974;
  --accent: #256abf;
  --good: #0f7a0f;
  --good-bg: #e8f4e6;
  --warn: #8a5a00;
  --warn-mark: #fab219;
  --warn-bg: #fdf3dc;
  --crit: #b3261e;
  --crit-mark: #d03b3b;
  --crit-bg: #fbe9e7;
  --note: #52514e;
  --src-calibration: #2a78d6;
  --src-hand-eye: #eb6834;
  --src-robot: #1baf7a;
  --src-noise: #c9c8c2;
  --bar: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #121211;
    --surface: #1a1a19;
    --sunken: #232321;
    --line: #34332f;
    --ink: #f4f4f2;
    --ink-2: #c3c2b7;
    --ink-3: #95948c;
    --accent: #6da7ec;
    --good: #5cc45c;
    --good-bg: #16261a;
    --warn: #f3c152;
    --warn-bg: #2b2414;
    --crit: #f08a82;
    --crit-bg: #331b19;
    --note: #c3c2b7;
    --src-calibration: #3987e5;
    --src-hand-eye: #d95926;
    --src-robot: #199e70;
    --src-noise: #55544f;
    --bar: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #121211; --surface: #1a1a19; --sunken: #232321; --line: #34332f;
  --ink: #f4f4f2; --ink-2: #c3c2b7; --ink-3: #95948c; --accent: #6da7ec;
  --good: #5cc45c; --good-bg: #16261a; --warn: #f3c152; --warn-bg: #2b2414;
  --crit: #f08a82; --crit-bg: #331b19; --note: #c3c2b7;
  --src-calibration: #3987e5; --src-hand-eye: #d95926; --src-robot: #199e70;
  --src-noise: #55544f; --bar: #3987e5;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
main { max-width: 900px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; line-height: 1.25; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-2);
     margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 16px; margin: 0 0 4px; }
p { margin: 0 0 10px; }
.muted { color: var(--ink-2); }
.small { font-size: 13px; }
.num { font-variant-numeric: tabular-nums; }
.toolbar { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; margin-bottom: 20px; }
button, .button {
  font: inherit; font-size: 14px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--line); border-radius: 8px; padding: 7px 14px; cursor: pointer;
}
button:hover, .button:hover { border-color: var(--ink-3); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.meta { color: var(--ink-2); font-size: 14px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }

.status { border-radius: 12px; padding: 18px 20px; margin: 20px 0 8px; border: 1px solid transparent; }
.status .label { font-size: 20px; font-weight: 650; display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.status ul { margin: 8px 0 0; padding-left: 20px; }
.status.good { background: var(--good-bg); color: var(--good); border-color: color-mix(in srgb, var(--good) 30%, transparent); }
.status.warn { background: var(--warn-bg); color: var(--warn); border-color: color-mix(in srgb, var(--warn) 30%, transparent); }
.status.crit { background: var(--crit-bg); color: var(--crit); border-color: color-mix(in srgb, var(--crit) 30%, transparent); }
.status p, .status li { color: var(--ink); }
.icon { width: 22px; height: 22px; flex: none; }

.headline p { font-size: 17px; line-height: 1.55; }
.headline p:first-child { font-size: 19px; font-weight: 600; }

.figures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
.figure { background: var(--sunken); border-radius: 10px; padding: 12px 14px; }
.figure .value { font-size: 24px; font-weight: 650; font-variant-numeric: tabular-nums; line-height: 1.2; }
.figure .value small { font-size: 14px; font-weight: 500; color: var(--ink-2); }
.figure .what { font-size: 13px; color: var(--ink-2); }
.quantity + .quantity { margin-top: 22px; padding-top: 18px; border-top: 1px dashed var(--line); }

.split { display: flex; height: 14px; gap: 2px; margin: 8px 0 6px; }
.split span { display: block; height: 100%; min-width: 0; }
.split span:first-child { border-radius: 4px 0 0 4px; }
.split span:last-child { border-radius: 0 4px 4px 0; }
.split span:only-child { border-radius: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13px; color: var(--ink-2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.legend b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }

.tolerance { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-top: 14px;
             padding: 12px 14px; border-radius: 10px; border: 1px solid var(--line); }
.tolerance label { font-weight: 600; font-size: 14px; }
.tolerance input { font: inherit; width: 110px; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px;
                   background: var(--surface); color: var(--ink); }
.tolerance .result { flex: 1 1 260px; font-size: 14px; }
.tolerance .result.good { color: var(--good); }
.tolerance .result.crit { color: var(--crit); }
.tolerance .result strong { font-weight: 650; }

.finding { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; }
.finding .badge { grid-row: span 3; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 650; letter-spacing: 0.03em;
         padding: 3px 9px; border-radius: 999px; white-space: nowrap; height: fit-content; }
.badge.CRITICAL { background: var(--crit-bg); color: var(--crit); }
.badge.WARNING { background: var(--warn-bg); color: var(--warn); }
.badge.NOTE { background: var(--sunken); color: var(--note); }
.badge.OK { background: var(--good-bg); color: var(--good); }
.badge svg { width: 12px; height: 12px; }
.todo { margin-top: 6px; padding: 8px 12px; background: var(--sunken); border-radius: 8px; font-size: 14px; }
.todo b { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-2); }
.passed { display: flex; flex-wrap: wrap; gap: 6px; }
.passed span { font-size: 13px; padding: 3px 10px; border-radius: 999px; background: var(--good-bg); color: var(--good); }

table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { font-size: 12px; color: var(--ink-2); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.scroll { overflow-x: auto; }
dl.kv { display: grid; grid-template-columns: minmax(140px, max-content) 1fr; gap: 6px 20px; margin: 0; font-size: 14px; }
dl.kv dt { color: var(--ink-2); }
dl.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
details { margin-top: 12px; }
details > summary { cursor: pointer; font-weight: 600; padding: 10px 0; }
details > summary .muted { font-weight: 400; }

.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--ink-2); font-size: 11px; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .ref { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 4 3; }
.chart .bar { fill: var(--bar); }
.chart .bar.outlier { fill: var(--crit-mark); }
.chart .hit { fill: transparent; }
.chart .hit:hover + .bar, .chart .bar:hover { opacity: 0.8; }

.question { font-size: 17px; }
footer { margin-top: 40px; font-size: 13px; color: var(--ink-3); }

.empty { text-align: center; padding: 72px 20px; border: 2px dashed var(--line); border-radius: 16px; background: var(--surface); margin-top: 40px; }
.empty h1 { margin-bottom: 8px; }
.error { color: var(--crit); margin-top: 14px; }
body.dragging main { outline: 3px dashed var(--accent); outline-offset: 8px; border-radius: 12px; }

@media (max-width: 640px) {
  h1 { font-size: 22px; }
  .figures { grid-template-columns: 1fr; }
  .finding { grid-template-columns: 1fr; }
  .finding .badge { grid-row: auto; }
  dl.kv { grid-template-columns: 1fr; gap: 0 0; }
  dl.kv dd { margin-bottom: 8px; }
}
@media print {
  :root { color-scheme: light; --page: #fff; --surface: #fff; }
  .toolbar, .tolerance.unset, .no-print { display: none !important; }
  h2 { break-after: avoid; }
  main { max-width: none; padding: 0; }
  .card, .status, .figure { break-inside: avoid; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
</style>
</head>
<body>
<main id="app"></main>
<input type="file" id="picker" accept=".json,application/json" hidden>
<script type="application/json" id="calibsense-audit">__CALIBSENSE_AUDIT_JSON__</script>
<script>
(function () {
  "use strict";

  var RANK = { OK: 0, NOTE: 1, WARNING: 2, CRITICAL: 3 };
  var SOURCES = [
    ["calibration", "Camera calibration", "--src-calibration"],
    ["hand_eye", "Hand-eye solve", "--src-hand-eye"],
    ["robot", "Robot repeatability", "--src-robot"],
    ["pixel_noise", "Pixel noise when measuring", "--src-noise"]
  ];
  var MODELS = {
    pinhole_brown_conrady: "Pinhole, Brown-Conrady distortion",
    fisheye_kannala_brandt: "Fisheye, Kannala-Brandt distortion"
  };
  var ICONS = {
    good: "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z",
    warn: "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
    crit: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z",
    note: "M11 7h2v2h-2zm0 4h2v6h-2zm1-9a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"
  };

  // Every string in a report is data, and some of it (camera name, contact,
  // the open question) is typed by whoever ran the audit, so nothing from the
  // payload is ever assigned as markup.
  function h(tag, props) {
    var node = document.createElement(tag);
    if (props) {
      for (var key in props) {
        if (key === "class") node.className = props[key];
        else if (key === "text") node.textContent = props[key];
        else if (key.slice(0, 2) === "on") node.addEventListener(key.slice(2), props[key]);
        else node.setAttribute(key, props[key]);
      }
    }
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }
  function append(node, child) {
    if (child == null || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(node, c); }); return; }
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  function svg(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var key in attrs || {}) node.setAttribute(key, attrs[key]);
    for (var i = 2; i < arguments.length; i++) if (arguments[i]) node.appendChild(arguments[i]);
    return node;
  }
  function icon(kind) {
    return svg("svg", { "class": "icon", viewBox: "0 0 24 24", "aria-hidden": "true" },
      svg("path", { d: ICONS[kind], fill: "currentColor" }));
  }

  // Matches Python's "{:.3g}", which is how the PDF and the text report print.
  function fmt(x, digits) {
    if (x == null || typeof x !== "number" || !isFinite(x)) return "–";
    digits = digits || 3;
    var a = Math.abs(x);
    if (a !== 0 && (a < 1e-4 || a >= Math.pow(10, digits + 3))) return x.toExponential(digits - 1);
    return String(Number(x.toPrecision(digits)));
  }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function signed(x, digits) { return (x > 0 ? "+" : "") + fmt(x, digits); }
  function when(iso) {
    var d = new Date(iso);
    if (isNaN(d)) return iso || "";
    return d.toLocaleString(undefined, { dateStyle: "long", timeStyle: "short" });
  }
  function section(title) {
    return h("h2", { text: title });
  }

  function status(report) {
    var caveats = report.caveats || [];
    var kind, label, lead;
    if (caveats.length) {
      kind = "crit"; label = "Do not sign off on these figures yet";
      lead = "The report says its numbers cannot be read at face value:";
    } else if (report.severity === "CRITICAL") {
      kind = "crit"; label = "Critical problems found";
      lead = "Resolve the critical findings below before relying on this calibration.";
    } else if (report.severity === "WARNING") {
      kind = "warn"; label = "Usable, with warnings";
      lead = "The figures can be read as they stand. The warnings below say what would make them better.";
    } else {
      kind = "good"; label = "No problems found";
      lead = "Every check passed, and the figures below can be read as they stand.";
    }
    return h("div", { "class": "status " + kind, role: "status" },
      h("div", { "class": "label" }, icon(kind === "good" ? "good" : kind === "warn" ? "warn" : "crit"), label),
      h("p", { text: lead }),
      caveats.length ? h("ul", null, caveats.map(function (c) { return h("li", { text: c + "." }); })) : null);
  }

  function split(q, sourceLabel) {
    var shares = q.variance_sources || {
      calibration: q.variance_share.calibration, pixel_noise: q.variance_share.pixel_noise
    };
    var parts = SOURCES.filter(function (s) { return shares[s[0]] > 0.0005; });
    var bar = h("div", { "class": "split", role: "img",
      "aria-label": parts.map(function (s) { return s[1] + " " + pct(shares[s[0]]); }).join(", ") },
      parts.map(function (s) {
        return h("span", { style: "flex:" + shares[s[0]] + ";background:var(" + s[2] + ")",
          title: s[1] + ": " + pct(shares[s[0]]) + " of the variance" });
      }));
    var legend = h("div", { "class": "legend" }, parts.map(function (s) {
      var name = s[0] === "calibration" && sourceLabel !== "the calibration" ? "Calibration (" + sourceLabel + ")" : s[1];
      return h("span", null, h("i", { style: "background:var(" + s[2] + ")" }), name + " ", h("b", { text: pct(shares[s[0]]) }));
    }));
    // The split is of variance, so "at most N%" is the most a perfect
    // calibration could remove, not what a recapture will.
    var improvable = 1 - (shares.pixel_noise || 0);
    var advice;
    if (improvable > 0.95) {
      advice = "Almost all of this error comes from the calibration, so a better calibration is what will reduce it.";
    } else if (improvable < 0.05) {
      advice = "Almost all of this error is pixel noise in the images taken at measurement time. Re-calibrating will not reduce it.";
    } else {
      advice = "Re-calibrating can reduce this error by at most " + pct(improvable) +
        ". The rest is pixel noise in the images taken at measurement time, which a better calibration cannot remove.";
    }
    return h("div", null,
      h("div", { "class": "small", style: "font-weight:600;margin-top:4px", text: "Where the error comes from" }),
      bar, legend,
      h("p", { "class": "small muted", style: "margin-top:6px", text: advice }));
  }

  function toleranceCheck(q, trustworthy) {
    var unit = q.quantity.unit;
    var result = h("div", { "class": "result muted", "aria-live": "polite",
      text: "Enter the tolerance your acceptance test allows to see whether this camera meets it." });
    var input = h("input", { type: "number", min: "0", step: "any", inputmode: "decimal", placeholder: "e.g. 0.5",
      "aria-label": "Tolerance in " + unit });
    function update() {
      var tol = parseFloat(input.value);
      result.className = "result";
      result.textContent = "";
      box.classList.toggle("unset", !(tol > 0));
      if (!(tol > 0)) {
        result.className = "result muted";
        result.textContent = "Enter the tolerance your acceptance test allows to see whether this camera meets it.";
        return;
      }
      var hw = q.half_width;
      if (!q.bounded) {
        result.className = "result crit";
        append(result, [h("strong", { text: "Cannot be checked. " }),
          "This figure is a lower bound, so the real error could exceed any tolerance. Fix the calibration first."]);
      } else if (hw <= tol) {
        result.className = "result good";
        append(result, [h("strong", { text: "Within tolerance. " }),
          "95% of measurements should land within ±" + fmt(hw) + " " + unit + ", inside your ±" + fmt(tol) + " " + unit + " (" + pct(hw / tol) + " of it)."]);
        if (!trustworthy) append(result, " Read the warning at the top before relying on this.");
      } else {
        result.className = "result crit";
        append(result, [h("strong", { text: "Does not meet tolerance. " }),
          "The 95% interval of ±" + fmt(hw) + " " + unit + " is " + fmt(hw / tol, 2) + "× your ±" + fmt(tol) + " " + unit + "."]);
      }
    }
    var box = h("div", { "class": "tolerance unset" },
      h("label", null, "Your tolerance ± ", input, " " + unit), result);
    input.addEventListener("input", update);
    return box;
  }

  function quantity(task, q, trustworthy) {
    var unit = q.quantity.unit;
    var desc = q.quantity.description;
    return h("div", { "class": "quantity" },
      h("h3", { text: desc.charAt(0).toUpperCase() + desc.slice(1) }),
      h("div", { "class": "figures" },
        h("div", { "class": "figure" },
          h("div", { "class": "value" }, "±" + fmt(q.half_width) + " ", h("small", { text: unit })),
          h("div", { "class": "what", text: q.bounded ? "95% of measurements fall within this" : "Lower bound only; the real spread may be far larger" })),
        h("div", { "class": "figure" },
          h("div", { "class": "value" }, fmt(q.expected_error) + " ", h("small", { text: unit })),
          h("div", { "class": "what", text: "Typical (expected) error of one measurement" })),
        h("div", { "class": "figure" },
          h("div", { "class": "value" }, signed(q.bias) + " ", h("small", { text: unit })),
          h("div", { "class": "what", text: "Average offset (bias): how far readings lean one way" }))),
      split(q, task.parameter_source || "the calibration"),
      q.task_space_widening > 1.02 ? h("p", { "class": "small muted", text:
        "These intervals are " + fmt(q.task_space_widening, 3) + "× wider than the simple noise model gives, because the views disagree with each other more than that model expects. Widening the interval is the honest figure; it does not make the calibration itself better." }) : null,
      toleranceCheck(q, trustworthy));
  }

  function tasks(report) {
    if (!report.tasks || !report.tasks.length) return null;
    return [section("Measurement error"), report.tasks.map(function (task) {
      return h("div", { "class": "card" },
        h("div", { "class": "muted small", style: "margin-bottom:10px", text: task.task.title + " — " + task.task.description }),
        task.quantities.map(function (q) { return quantity(task, q, report.trustworthy); }));
    })];
  }

  function badge(severity) {
    var kind = { CRITICAL: "crit", WARNING: "warn", NOTE: "note", OK: "good" }[severity] || "note";
    var i = icon(kind); i.setAttribute("class", "");
    return h("span", { "class": "badge " + severity }, i, severity === "OK" ? "PASSED" : severity);
  }

  function findings(report) {
    var all = [];
    (report.diagnosis.findings || []).forEach(function (f) { all.push([f, null]); });
    if (report.hand_eye_diagnosis) {
      report.hand_eye_diagnosis.findings.forEach(function (f) { all.push([f, "Hand-eye"]); });
    }
    all.sort(function (a, b) { return (RANK[b[0].severity] || 0) - (RANK[a[0].severity] || 0); });
    var open = all.filter(function (p) { return RANK[p[0].severity] >= RANK.NOTE; });
    var passed = all.filter(function (p) { return RANK[p[0].severity] < RANK.NOTE; });
    return [
      section(open.length ? "What needs attention" : "Checks"),
      open.length ? h("p", { "class": "muted", text: report.diagnosis.verdict }) : null,
      open.map(function (p) {
        var f = p[0];
        return h("div", { "class": "card finding" },
          badge(f.severity),
          h("h3", { text: (p[1] ? p[1] + ": " : "") + f.title }),
          h("p", { style: "margin:0", text: f.summary.charAt(0).toUpperCase() + f.summary.slice(1) + "." }),
          f.action ? h("div", { "class": "todo" }, h("b", { text: "What to do" }), f.action) : null);
      }),
      passed.length ? h("div", { style: "margin-top:14px" },
        h("div", { "class": "small muted", style: "margin-bottom:6px", text: "Checks that passed" }),
        h("div", { "class": "passed" }, passed.map(function (p) {
          return h("span", { title: p[0].summary, text: "✓ " + (p[1] ? p[1] + ": " : "") + p[0].title });
        }))) : null
    ];
  }

  function forecast(report) {
    var f = report.forecast;
    if (!f) return null;
    var q = report.tasks && report.tasks[0] && report.tasks[0].quantities[0];
    var unit = q ? q.quantity.unit : "mm";
    var rec = f.recommendation || {};
    var sentence = f.worthwhile
      ? rec.description + " would take the 95% interval from ±" + fmt(f.current_half_width) + " to about ±" + fmt(f.forecast_half_width) + " " + unit + "."
      : "The most useful change, " + (rec.description || "more views").replace(/^Adding/, "adding") + ", would move the interval only from ±" + fmt(f.current_half_width) + " to about ±" + fmt(f.forecast_half_width) + " " + unit + ". Recapturing is not worth it for this.";
    return [section("What a recapture would buy"), h("div", { "class": "card" },
      h("p", { text: sentence }),
      h("dl", { "class": "kv" },
        h("dt", { text: "Views" }), h("dd", { text: f.n_views_before + " → " + f.n_views_after }),
        h("dt", { text: "95% interval" }), h("dd", { text: "±" + fmt(f.current_half_width) + " → ±" + fmt(f.forecast_half_width) + " " + unit }),
        h("dt", { text: "Every parameter determined" }), h("dd", { text: (f.currently_identifiable ? "yes" : "no") + " → " + (f.forecast_identifiable ? "yes" : "no") })),
      h("p", { "class": "small muted", style: "margin:10px 0 0", text:
        "This predicts how much tighter the interval would get, not that the answer would then be right: the extra views are simulated from the calibration already in hand." }))];
  }

  function outliers(report) {
    var c = report.without_outliers;
    if (!c) return null;
    return [section("What the outlier views are costing"), h("div", { "class": "card" },
      h("p", { text: "Leaving out " + c.dropped.join(", ") + " and refitting on the other " + c.n_views_after +
        " views gives the figures below. The report does not decide this for you: whether a badly fitting image should be thrown away is something only a person looking at it can judge." }),
      h("dl", { "class": "kv" },
        h("dt", { text: "Residual noise" }), h("dd", { text: fmt(c.sigma_before_px, 4) + " → " + fmt(c.sigma_after_px, 4) + " px" }),
        h("dt", { text: "Focal length uncertainty" }), h("dd", { text: fmt(c.fx_std_before, 4) + " → " + fmt(c.fx_std_after, 4) + " px" }),
        h("dt", { text: "Focal length moves by" }), h("dd", { text: signed(c.fx_shift, 4) + " px" })))];
  }

  function handEye(report) {
    var he = report.hand_eye;
    if (!he) return null;
    var t = he.camera.translation_mm, ts = he.translation_std_mm, rs = he.rotation_std_deg;
    var label = he.mounting === "eye_in_hand" ? "Flange to camera" : "Base to camera";
    return [section("Hand-eye"), h("div", { "class": "card" },
      h("dl", { "class": "kv" },
        h("dt", { text: "Mounting" }), h("dd", { text: he.mounting.replace(/_/g, "-") + ", " + he.n_views + " robot poses" }),
        h("dt", { text: label + " (mm)" }), h("dd", { text: t.map(function (v, i) { return signed(v, 4) + " ± " + fmt(ts[i], 2); }).join(",  ") }),
        h("dt", { text: "Rotation uncertainty" }), h("dd", { text: rs.map(function (v) { return fmt(v, 2); }).join(", ") + " deg" }),
        h("dt", { text: "Residual" }), h("dd", { text: fmt(he.rotation_rms_deg, 3) + " deg, " + fmt(he.translation_rms_mm, 3) + " mm RMS" }),
        h("dt", { text: "Every parameter determined" }), h("dd", { text: he.identifiable ? "yes" : "NO" })))];
  }

  function viewChart(fit) {
    var views = fit.views || [];
    if (!views.length) return null;
    var W = 860, H = 220, L = 44, R = 10, T = 12, B = 30;
    var max = Math.max.apply(null, views.map(function (v) { return v.rms; }).concat([fit.rms])) * 1.1;
    var step = Math.pow(10, Math.floor(Math.log10(max / 4)));
    var tick = [1, 2, 5, 10].map(function (m) { return m * step; }).filter(function (s) { return max / s <= 5; })[0];
    var y = function (v) { return T + (H - T - B) * (1 - v / max); };
    var bw = (W - L - R) / views.length;
    var chart = svg("svg", { "class": "chart", viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Reprojection error per view; outlier views are marked in red" });
    for (var g = 0; g <= max; g += tick) {
      chart.appendChild(svg("line", { "class": "grid", x1: L, x2: W - R, y1: y(g), y2: y(g) }));
      var label = svg("text", { x: L - 6, y: y(g) + 4, "text-anchor": "end" });
      label.textContent = fmt(g, 2);
      chart.appendChild(label);
    }
    views.forEach(function (v, i) {
      var x = L + i * bw, top = y(v.rms), width = Math.max(2, bw - 2);
      var r = Math.min(4, width / 2);
      var d = "M" + (x + 1) + "," + y(0) + "V" + (top + r) + "q0,-" + r + " " + r + ",-" + r +
              "H" + (x + 1 + width - r) + "q" + r + ",0 " + r + "," + r + "V" + y(0) + "Z";
      var bar = svg("path", { "class": "bar" + (v.is_outlier ? " outlier" : ""), d: d });
      var title = svg("title");
      title.textContent = v.view_id + ": " + fmt(v.rms, 3) + " px RMS over " + v.n_points + " points at " +
        Math.round(v.distance_mm) + " mm" + (v.is_outlier ? " (outlier)" : "");
      var hit = svg("rect", { "class": "hit", x: x, y: T, width: bw, height: H - T - B });
      hit.appendChild(title.cloneNode(true));
      bar.appendChild(title);
      chart.appendChild(hit);
      chart.appendChild(bar);
      if (v.is_outlier) {
        var tag = svg("text", { x: x + bw / 2, y: top - 4, "text-anchor": "middle" });
        tag.textContent = "outlier";
        chart.appendChild(tag);
      }
    });
    chart.appendChild(svg("line", { "class": "ref", x1: L, x2: W - R, y1: y(fit.rms), y2: y(fit.rms) }));
    var ref = svg("text", { x: W - R, y: y(fit.rms) - 4, "text-anchor": "end" });
    ref.textContent = "overall " + fmt(fit.rms, 3) + " px";
    chart.appendChild(ref);
    var axis = svg("text", { x: L, y: H - 8 });
    axis.textContent = "each bar is one calibration image, in capture order";
    chart.appendChild(axis);
    return h("div", null,
      h("div", { "class": "small", style: "font-weight:600;margin:14px 0 4px", text: "Reprojection error per image (px)" }),
      chart);
  }

  function technical(report) {
    var fit = report.fit, s = report.session, cv = fit.cross_validation, c = fit.conditioning;
    var rows = [
      ["Camera model", (MODELS[fit.camera.kind] || fit.camera.kind.replace(/_/g, " ")) + ", " + fit.camera.distortion.length + " distortion terms"],
      ["Target", s.target_description],
      ["Images used", s.detection && s.detection.attempted ? fit.n_views + " of " + s.detection.attempted + " (" + s.detection.detector + ")" : String(fit.n_views)],
      ["Corner points", String(fit.total_points)],
      ["Reprojection error", fmt(fit.rms, 4) + " px RMS in sample"],
      cv ? ["Held-out error", fmt(cv.out_of_sample_rms, 4) + " px over " + cv.n_folds + " folds (ratio " + cv.ratio.toFixed(2) + "×)"] : null,
      fit.prior_rms != null ? ["Shipped file claimed", fmt(fit.prior_rms, 4) + " px"] : null,
      ["Residual noise", fmt(fit.sigma, 4) + " px per coordinate"],
      ["Every parameter determined", c.identifiable ? "yes" : "NO (rank " + c.rank + " of " + c.n_intrinsic + ")"],
      ["Scaled condition number", fmt(c.scaled_condition_number, 3)],
      ["Working distance", Math.round(fit.working_distance_mm.min) + " to " + Math.round(fit.working_distance_mm.max) + " mm"]
    ].filter(Boolean);
    var failures = (s.detection && s.detection.failures) || [];
    var params = h("div", { "class": "scroll" }, h("table", null,
      h("thead", null, h("tr", null, ["Parameter", "Value", "Std dev", "Relative", "Weak"].map(function (t) { return h("th", { text: t }); }))),
      h("tbody", null, fit.parameters.map(function (p) {
        return h("tr", null,
          h("td", { text: p.name }), h("td", { text: fmt(p.value, 6) }), h("td", { text: fmt(p.std_dev, 4) }),
          h("td", { text: p.value ? (100 * Math.abs(p.std_dev / p.value)).toFixed(2) + "%" : "–" }),
          h("td", { text: p.weak_participation.toFixed(2) }));
      }))));
    return h("details", { "class": "card", id: "technical" },
      h("summary", null, "Technical details ", h("span", { "class": "muted small", text: "for the calibration engineer" })),
      h("dl", { "class": "kv", style: "margin-top:6px" }, rows.map(function (r) { return [h("dt", { text: r[0] }), h("dd", { text: r[1] })]; })),
      failures.length ? h("p", { "class": "small muted", style: "margin-top:10px", text: failures.length + " image(s) were rejected during detection." }) : null,
      h("div", { style: "margin-top:16px" }, params),
      !c.identifiable ? h("p", { "class": "small muted", style: "margin-top:8px", text:
        "A weak value near 1 means the capture does not constrain that parameter. Its standard deviation is then not meaningful: it reads as small when it is really unbounded." }) : null,
      viewChart(fit));
  }

  function render(report) {
    var m = report.metadata;
    document.title = m.title + " — " + m.camera_name;
    var app = document.getElementById("app");
    app.textContent = "";
    var sub = [m.camera_name, when(m.created)];
    if (m.prepared_by) sub.push("prepared by " + m.prepared_by);
    append(app, [
      toolbar(),
      h("h1", { text: m.title }),
      h("div", { "class": "meta", text: sub.join("  ·  ") }),
      status(report),
      section("What this camera can measure"),
      h("div", { "class": "headline" }, (report.headline || []).map(function (sn) { return h("p", { text: sn }); })),
      tasks(report),
      findings(report),
      forecast(report),
      outliers(report),
      handEye(report),
      section("More detail"),
      technical(report),
      section("The open question"),
      h("p", { "class": "question", text: m.open_question }),
      h("p", null, h("strong", { text: "Questions about this report: " }), m.contact),
      h("footer", null,
        "Produced by calibsense " + m.calibsense_version + ". Every figure is reproducible from the session bundle and the seed recorded in the report data. This page reads the report only; it does not recompute anything.")
    ]);
    window.scrollTo(0, 0);
  }

  function toolbar() {
    return h("div", { "class": "toolbar no-print" },
      h("button", { type: "button", onclick: pick, text: "Open another report…" }),
      h("button", { type: "button", "class": "primary", onclick: function () { window.print(); }, text: "Print or save as PDF" }));
  }

  function empty(message) {
    var app = document.getElementById("app");
    app.textContent = "";
    append(app, h("div", { "class": "empty" },
      h("h1", { text: "calibsense report viewer" }),
      h("p", { "class": "muted", text: "Drop an audit .json file anywhere on this page, or choose one." }),
      h("button", { type: "button", "class": "primary", onclick: pick, text: "Choose a report…" }),
      message ? h("p", { "class": "error", text: message }) : null));
  }

  function valid(report) {
    return report && typeof report === "object" && report.metadata && report.diagnosis &&
      report.fit && Array.isArray(report.fit.parameters);
  }

  function load(file) {
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      var report;
      try { report = JSON.parse(reader.result); } catch (e) { report = null; }
      if (valid(report)) render(report);
      else empty("“" + file.name + "” is not a calibsense report. Open the .json written by the report step (the --json file).");
    };
    reader.onerror = function () { empty("Could not read “" + file.name + "”."); };
    reader.readAsText(file);
  }

  var picker = document.getElementById("picker");
  function pick() { picker.value = ""; picker.click(); }
  picker.addEventListener("change", function () { load(picker.files[0]); });

  var depth = 0;
  window.addEventListener("dragenter", function (e) { e.preventDefault(); depth++; document.body.classList.add("dragging"); });
  window.addEventListener("dragleave", function () { if (--depth <= 0) { depth = 0; document.body.classList.remove("dragging"); } });
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) {
    e.preventDefault(); depth = 0; document.body.classList.remove("dragging");
    load(e.dataTransfer.files[0]);
  });

  // A printed copy has no way to expand a disclosure, so print everything.
  window.addEventListener("beforeprint", function () {
    document.querySelectorAll("details").forEach(function (d) { d.setAttribute("open", ""); });
  });

  var embedded = null;
  try { embedded = JSON.parse(document.getElementById("calibsense-audit").textContent); } catch (e) { embedded = null; }
  if (valid(embedded)) render(embedded); else empty();
})();
</script>
</body>
</html>
"""
