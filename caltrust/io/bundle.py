"""A single-file container for a JSON manifest plus NumPy arrays.

Sessions and fits are both "some structured metadata and some large arrays".
Writing them as one compressed `.npz` keeps a result to one file the engineer
can attach to a ticket, while `allow_pickle=False` throughout means loading a
bundle can never execute code from it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ..errors import SerializationError

_MANIFEST_KEY = "__manifest__"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def write_bundle(
    path: str,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    compress: bool = True,
) -> str:
    """Write a manifest and arrays to one file.

    Args:
        path: Destination path. A `.npz` suffix is appended when missing.
        manifest: JSON-serialisable metadata.
        arrays: Named arrays. A name may not collide with the manifest key.
        compress: Use deflate. Off is faster, on is roughly 3x smaller for
            residual arrays.

    Returns:
        The path actually written.

    Raises:
        SerializationError: The manifest is not JSON-serialisable, an array name
            is reserved, or the write failed.
    """
    if _MANIFEST_KEY in arrays:
        raise SerializationError(f"{_MANIFEST_KEY!r} is reserved for the manifest")
    destination = path if path.endswith(".npz") else path + ".npz"
    try:
        encoded = json.dumps(manifest, default=_json_default, sort_keys=True)
    except TypeError as exc:
        raise SerializationError(f"manifest is not JSON-serialisable: {exc}") from exc

    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload[_MANIFEST_KEY] = np.asarray(encoded)
    parent = os.path.dirname(os.path.abspath(destination))
    if parent:
        os.makedirs(parent, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    try:
        save(destination, **payload)
    except OSError as exc:
        raise SerializationError(f"could not write {destination}: {exc}") from exc
    return destination


def read_bundle(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Read a bundle back.

    Args:
        path: Path to a file written by `write_bundle`.

    Returns:
        The manifest dictionary and the arrays, materialised eagerly so the
        underlying file handle is closed before returning.

    Raises:
        SerializationError: The file is missing, not an npz, or carries no
            caltrust manifest.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _MANIFEST_KEY not in archive.files:
                raise SerializationError(
                    f"{path} is not a caltrust bundle: no manifest inside"
                )
            manifest = json.loads(str(archive[_MANIFEST_KEY]))
            arrays = {
                name: archive[name] for name in archive.files if name != _MANIFEST_KEY
            }
    except FileNotFoundError as exc:
        raise SerializationError(f"no such file: {path}") from exc
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"could not read {path}: {exc}") from exc
    return manifest, arrays


def check_format(manifest: Mapping[str, Any], expected: str, max_version: int) -> None:
    """Validate a bundle's declared format before trusting its contents.

    Args:
        manifest: The manifest read back from a bundle.
        expected: The `format` string this reader handles.
        max_version: Highest `format_version` this reader understands.

    Raises:
        SerializationError: The format or version does not match.
    """
    found = manifest.get("format")
    if found != expected:
        raise SerializationError(
            f"expected a {expected!r} bundle, found {found!r}"
        )
    version = int(manifest.get("format_version", 0))
    if version > max_version:
        raise SerializationError(
            f"bundle format version {version} is newer than this caltrust "
            f"understands (max {max_version}); upgrade caltrust"
        )
