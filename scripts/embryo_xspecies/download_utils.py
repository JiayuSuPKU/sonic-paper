"""Shared HTTP download helpers for embryo_xspecies data scripts."""

from __future__ import annotations

import fnmatch
import http.client
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config as C

logger = logging.getLogger("embryo_xspecies.download")

_HEADERS = {"User-Agent": "sonic-paper-embryo-xspecies/1.0"}


def matches_section(section_id: str, patterns: list[str]) -> bool:
    """Match a section by exact ID, substring, or shell-style glob."""
    return any(
        pattern == section_id or pattern in section_id or fnmatch.fnmatchcase(section_id, pattern)
        for pattern in patterns
    )


def repo_relative_path(path: Path) -> str:
    """Return a repo-relative path when possible; otherwise return the input path."""
    try:
        return str(path.relative_to(C.PROJECT_ROOT))
    except ValueError:
        return str(path)


def open_url(url: str, *, method: str = "GET", timeout: int = 60):
    return urllib.request.urlopen(
        urllib.request.Request(url, method=method, headers=_HEADERS), timeout=timeout
    )


def remote_size(url: str, *, timeout: int = 30) -> int | None:
    try:
        with open_url(url, method="HEAD", timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except (
        urllib.error.URLError,
        ValueError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ):
        pass
    try:
        headers = dict(_HEADERS)
        headers["Range"] = "bytes=0-0"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            cr = resp.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)\s*$", cr)
            return int(m.group(1)) if m else None
    except (
        urllib.error.URLError,
        ValueError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ):
        return None


def _attempt_one_pass(
    url: str,
    part: Path,
    *,
    expected_size: int | None,
    chunk: int,
    tag: str,
    log_every: float,
) -> int:
    """One streaming pass into ``part``. Returns additional bytes pulled."""
    start_byte = part.stat().st_size if part.exists() else 0
    if expected_size is not None and start_byte >= expected_size:
        return 0
    headers = dict(_HEADERS)
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    req = urllib.request.Request(url, headers=headers)
    prefix = f"[{tag}] " if tag else "    "
    new_bytes = 0
    t0 = time.time()
    last_log = t0
    with urllib.request.urlopen(req, timeout=60) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        if start_byte > 0 and status != 206:
            logger.warning(
                "%s server returned %s, not 206 Partial Content — restarting from 0",
                prefix.strip(),
                status,
            )
            start_byte = 0
            part.unlink(missing_ok=True)
        mode = "ab" if start_byte > 0 else "wb"
        with open(part, mode) as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                new_bytes += len(buf)
                now = time.time()
                if now - last_log > log_every:
                    written = start_byte + new_bytes
                    rate = new_bytes / max(now - t0, 1e-9) / 1e6
                    pct = f"{100.0 * written / expected_size:5.1f}%" if expected_size else "  ?  "
                    logger.info(
                        "%s%s  %8.1f MB  (%.1f MB/s)",
                        prefix,
                        pct,
                        written / 1e6,
                        rate,
                    )
                    last_log = now
    return new_bytes


def _backoff_seconds(attempt: int, *, base: float = 5.0, cap: float = 60.0) -> float:
    return min(cap, base * (2 ** (attempt - 1)))


def download(
    url: str,
    dest: Path,
    *,
    chunk: int = 1 << 22,
    tag: str = "",
    log_every: float = 15.0,
    expected_size: int | None = None,
    max_attempts: int = 8,
) -> int:
    """Stream ``url`` to ``dest`` via ``dest.part`` with size check and resume."""
    part = dest.with_name(dest.name + ".part")
    if expected_size is None:
        expected_size = remote_size(url)
    pulled_total = 0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            pulled_total += _attempt_one_pass(
                url,
                part,
                expected_size=expected_size,
                chunk=chunk,
                tag=tag,
                log_every=log_every,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_exc = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_exc = exc
        else:
            actual = part.stat().st_size if part.exists() else 0
            if expected_size is None or actual == expected_size:
                part.replace(dest)
                return pulled_total
            last_exc = None
            if attempt < max_attempts:
                sleep_s = _backoff_seconds(attempt)
                logger.warning(
                    "[%s] attempt %d/%d truncated: have %.1f / %.1f MB; "
                    "sleeping %.0fs then resuming.",
                    tag,
                    attempt,
                    max_attempts,
                    actual / 1e6,
                    expected_size / 1e6,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
        if attempt == max_attempts:
            break
        sleep_s = _backoff_seconds(attempt)
        logger.warning(
            "[%s] attempt %d/%d errored: %s — sleeping %.0fs then retrying.",
            tag,
            attempt,
            max_attempts,
            last_exc,
            sleep_s,
        )
        time.sleep(sleep_s)
    raise OSError(
        f"[{tag}] failed after {max_attempts} attempts "
        f"(last error: {last_exc}); .part is {part.stat().st_size if part.exists() else 0} bytes."
    )
