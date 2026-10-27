"""Installs ``usd-core`` (pxr) for Fusion's Python, which does not ship it, so binary .usdc can be written.

The wheel is unpacked into the user's config directory; nothing inside the Fusion install is touched."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional

from .ui.settings import config_dir

PYPI_URL = "https://pypi.org/pypi/usd-core/json"
# macOS framework Pythons (Fusion's included) ship without CA certificates.
_CA_FILES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt")


def lib_dir() -> str:
    # Per Python version: a Fusion update to a newer Python needs a new wheel.
    return os.path.join(config_dir(), f"lib-py{sys.version_info[0]}{sys.version_info[1]}")


def activate() -> None:
    path = lib_dir()
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def has_pxr() -> bool:
    return import_error() is None


def import_error() -> Optional[str]:
    try:
        from pxr import Sdf  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        return str(exc)
    return None


def pick_wheel(files: List[Dict[str, Any]], python: str, system: str, machine: str) -> Optional[Dict[str, Any]]:
    """The wheel for ``python`` (e.g. "cp314") on ``system``/``machine`` from PyPI's file list."""
    machine = machine.lower()
    if system == "Darwin":
        platforms = ("universal2", "arm64" if machine == "arm64" else "x86_64")
        ok = lambda tag: tag.startswith("macosx") and tag.endswith(platforms)  # noqa: E731
    elif system == "Windows":
        ok = lambda tag: tag == "win_amd64"  # noqa: E731
    else:
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        ok = lambda tag: "manylinux" in tag and tag.endswith(arch)  # noqa: E731
    for f in files:
        name = f.get("filename", "")
        if f.get("packagetype") != "bdist_wheel" or not name.endswith(".whl"):
            continue
        _, _, py, _, tag = name[:-4].split("-")[:5]
        if py == python and any(ok(t) for t in tag.split(".")):
            return f
    return None


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not ctx.cert_store_stats().get("x509_ca"):
        cafile = next((p for p in _CA_FILES if os.path.exists(p)), None)
        if cafile:
            ctx.load_verify_locations(cafile)
    return ctx


def install() -> str:
    """Download and unpack the matching usd-core wheel; returns its version."""
    ctx = _ssl_context()
    with urllib.request.urlopen(PYPI_URL, context=ctx, timeout=60) as r:
        info = json.load(r)
    python = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    wheel = pick_wheel(info["urls"], python, platform.system(), platform.machine())
    if wheel is None:
        raise RuntimeError(f"usd-core {info['info']['version']} has no wheel for {python} on "
                           f"{platform.system()} {platform.machine()}")
    os.makedirs(config_dir(), exist_ok=True)
    work = tempfile.mkdtemp(prefix="usd-core-", dir=config_dir())
    try:
        archive = os.path.join(work, wheel["filename"])
        digest = hashlib.sha256()
        with urllib.request.urlopen(wheel["url"], context=ctx, timeout=600) as r, open(archive, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != wheel["digests"]["sha256"]:
            raise RuntimeError(f"checksum mismatch for {wheel['filename']}")
        unpacked = os.path.join(work, "lib")
        with zipfile.ZipFile(archive) as z:
            z.extractall(unpacked)
        target = lib_dir()
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.replace(unpacked, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    activate()
    return info["info"]["version"]
