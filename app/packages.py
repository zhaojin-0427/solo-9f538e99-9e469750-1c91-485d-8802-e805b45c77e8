"""Frozen input packages: one ZIP per model run.

A package freezes, for a UTC day range of 1-7 days, the latest published
mosaic version of every day at one consistent snapshot, together with a
manifest (day, version, SHA-256 per member).  The lifecycle mirrors
:func:`app.store.publish`:

1. one ``SELECT`` picks the newest version per day (a single WAL snapshot);
2. the ZIP is rendered to a temporary file *without holding the write lock*,
    so submissions keep publishing while packaging runs.  Every source file
    must exist and hash to the SHA-256 registered for its version, otherwise
    the whole package fails;
3. a ``BEGIN IMMEDIATE`` transaction re-checks that every selected version is
   still the latest for its day.  If a newer version was published meanwhile
   the build is aborted with a conflict and the temporary file is removed, so
   no downloadable half-package is left behind;
4. only then is the archive renamed to its immutable final path and the
   package row inserted in the same transaction.  A package therefore becomes
   downloadable by number only after both the file and the registration
   succeeded.

Packages are never overwritten or deleted.  Versions published *after*
registration do not change an old package; query endpoints mark it as
``stale`` and list the superseded days instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import store
from .config import Settings
from .rasters import iso_utc, sha256_file

log = logging.getLogger(__name__)

MAX_PACKAGE_DAYS = 7
PACKAGE_FORMAT = "rainmosaic-package/1"
MANIFEST_NAME = "manifest.json"
# Fixed ZIP member timestamps keep archives byte-reproducible for identical
# inputs (and ZIP timestamps cannot represent dates before 1980 anyway).
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class PackageMissingDays(Exception):
    """Some days in the requested range have no published version."""

    def __init__(self, days: list[str]):
        super().__init__(f"no published version for: {', '.join(days)}")
        self.days = days


class PackageConflict(Exception):
    """A newer version appeared for a selected day before registration."""

    def __init__(self, conflicts: list[dict[str, Any]]):
        days = ", ".join(item["day"] for item in conflicts)
        super().__init__(f"newer versions published while packaging: {days}")
        self.conflicts = conflicts


class PackageSourceMissing(Exception):
    """The image file for a selected version is absent on disk."""

    def __init__(self, day: str, version: int):
        super().__init__(f"source image for {day} version {version} is missing on disk")
        self.day = day
        self.version = version


class PackageSourceMismatch(Exception):
    """A source file's content does not match its registered SHA-256."""

    def __init__(self, day: str, version: int, expected: str, actual: str):
        super().__init__(
            f"source image for {day} version {version} does not match its registered SHA-256"
        )
        self.day = day
        self.version = version
        self.expected = expected
        self.actual = actual


class PackageBusy(Exception):
    """The write lock was held by another writer for longer than the timeout."""


class PackageError(Exception):
    """The package could not be rendered or registered."""


@dataclass(frozen=True)
class SelectedVersion:
    day: str
    version: int
    sha256: str
    size_bytes: int
    filename: str
    path: Path


@dataclass(frozen=True)
class PackageDownload:
    package_id: int
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    stale: bool


def enumerate_days(start_day: str, end_day: str) -> list[str]:
    first = date.fromisoformat(start_day)
    last = date.fromisoformat(end_day)
    return [(first + timedelta(days=offset)).isoformat() for offset in range((last - first).days + 1)]


def _selection_from_row(settings: Settings, row: sqlite3.Row) -> SelectedVersion:
    version = int(row["version"])
    return SelectedVersion(
        day=row["day"],
        version=version,
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
        filename=f"rainmosaic-{row['day']}-v{version}.tif",
        path=settings.versions_dir / f"v{version}.tif",
    )


def select_latest_versions(
    settings: Settings, *, start_day: str, end_day: str, days: list[str]
) -> list[SelectedVersion]:
    """Pick the newest version per day in one consistent read snapshot."""

    conn = store.connect(settings)
    try:
        rows = conn.execute(
            "SELECT * FROM versions WHERE day >= ? AND day <= ? ORDER BY day, version",
            (start_day, end_day),
        ).fetchall()
    finally:
        conn.close()

    latest: dict[str, sqlite3.Row] = {}
    for row in rows:  # ascending version: the last row per day wins
        latest[row["day"]] = row

    missing = [day for day in days if day not in latest]
    if missing:
        raise PackageMissingDays(missing)
    return [_selection_from_row(settings, latest[day]) for day in days]


def build_package_zip(
    settings: Settings,
    selections: list[SelectedVersion],
    temp_path: Path,
    *,
    start_day: str,
    end_day: str,
    created_at: str,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Render the ZIP to ``temp_path`` and verify it against the registry.

    Returns ``(sha256, size_bytes, manifest_entries)``.  Raises
    :class:`PackageSourceMissing` or :class:`PackageSourceMismatch` when a
    source file is absent or its content differs from the registered
    SHA-256; any other failure raises :class:`PackageError`.  Nothing is
    registered here, so a failure leaves no downloadable package behind.
    """

    payloads: list[tuple[SelectedVersion, bytes]] = []
    entries: list[dict[str, Any]] = []
    for selection in selections:
        if not selection.path.is_file():
            raise PackageSourceMissing(selection.day, selection.version)
        data = selection.path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != selection.sha256:
            raise PackageSourceMismatch(selection.day, selection.version, selection.sha256, actual)
        payloads.append((selection, data))
        entries.append(
            {
                "day": selection.day,
                "version": selection.version,
                "sha256": selection.sha256,
                "filename": selection.filename,
                "size_bytes": len(data),
            }
        )

    manifest = {
        "format": PACKAGE_FORMAT,
        "start_day": start_day,
        "end_day": end_day,
        "created_at": created_at,
        "entries": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for selection, data in payloads:
                info = zipfile.ZipInfo(selection.filename, date_time=ZIP_EPOCH)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, data)
            manifest_info = zipfile.ZipInfo(MANIFEST_NAME, date_time=ZIP_EPOCH)
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            manifest_info.external_attr = 0o644 << 16
            zf.writestr(manifest_info, manifest_bytes)

        # Re-open the finished archive and confirm every member still hashes
        # to the registered SHA-256 before the package may be registered.
        with zipfile.ZipFile(temp_path) as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                raise PackageError(f"freshly built archive failed CRC check at {corrupt}")
            names = set(zf.namelist())
            expected_names = {selection.filename for selection, _ in payloads} | {MANIFEST_NAME}
            if names != expected_names:
                raise PackageError("freshly built archive has unexpected members")
            for selection, _ in payloads:
                digest = hashlib.sha256(zf.read(selection.filename)).hexdigest()
                if digest != selection.sha256:
                    raise PackageError(
                        f"archived {selection.filename} does not match its registered SHA-256"
                    )
    except PackageError:
        raise
    except (PackageSourceMissing, PackageSourceMismatch):
        raise
    except Exception as exc:
        raise PackageError(f"could not render package archive: {exc}") from exc

    store.fsync_file(temp_path)
    return sha256_file(temp_path), temp_path.stat().st_size, entries


def create_package(settings: Settings, *, start_day: str, end_day: str) -> dict[str, Any]:
    """Freeze the latest version of every day in the range into one ZIP."""

    days = enumerate_days(start_day, end_day)
    selections = select_latest_versions(settings, start_day=start_day, end_day=end_day, days=days)

    created_at = iso_utc(datetime.now(timezone.utc))
    temp_path = settings.tmp_dir / f"package-{uuid.uuid4().hex}.zip"
    final_path: Path | None = None
    package_id: int | None = None

    try:
        sha256, size_bytes, entries = build_package_zip(
            settings,
            selections,
            temp_path,
            start_day=start_day,
            end_day=end_day,
            created_at=created_at,
        )
    except (PackageSourceMissing, PackageSourceMismatch):
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        if isinstance(exc, PackageError):
            raise
        raise PackageError("package archive could not be generated") from exc

    conn = store.connect(settings)
    try:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise PackageBusy() from exc
            raise

        # The archive is built; confirm no selected day gained a newer
        # version while packaging ran.  Holding the write lock means no
        # publication can interleave with this check and the insert below.
        rows = conn.execute(
            "SELECT day, MAX(version) AS v FROM versions WHERE day >= ? AND day <= ? GROUP BY day",
            (start_day, end_day),
        ).fetchall()
        current = {row["day"]: int(row["v"]) for row in rows}
        conflicts = [
            {
                "day": selection.day,
                "packaged_version": selection.version,
                "current_version": current[selection.day],
            }
            for selection in selections
            if current.get(selection.day) != selection.version
        ]
        if conflicts:
            raise PackageConflict(conflicts)

        row = conn.execute("SELECT COALESCE(MAX(package_id), 0) + 1 AS n FROM packages").fetchone()
        package_id = int(row["n"])

        final_path = settings.packages_dir / f"p{package_id}.zip"
        os.replace(temp_path, final_path)
        store.fsync_dir(settings.packages_dir)

        conn.execute(
            """
            INSERT INTO packages (
                package_id, start_day, end_day, status, created_at,
                sha256, size_bytes, manifest
            ) VALUES (?, ?, ?, 'ready', ?, ?, ?, ?)
            """,
            (
                package_id,
                start_day,
                end_day,
                created_at,
                sha256,
                size_bytes,
                json.dumps(entries),
            ),
        )
        row = conn.execute("SELECT * FROM packages WHERE package_id = ?", (package_id,)).fetchone()
        assert row is not None
        conn.execute("COMMIT")
        log.info(
            "registered package %s for %s..%s (%s days, %s bytes)",
            package_id,
            start_day,
            end_day,
            len(entries),
            size_bytes,
        )
        return _row_to_payload(conn, row)
    except PackageConflict:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        raise
    except PackageBusy:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        _rollback(conn)
        temp_path.unlink(missing_ok=True)
        if final_path is not None and package_id is not None:
            # Only remove the renamed file if the package really is not in
            # the database (guards against an ambiguous commit outcome).
            try:
                row = conn.execute(
                    "SELECT package_id FROM packages WHERE package_id = ?", (package_id,)
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row is None:
                final_path.unlink(missing_ok=True)
        log.exception("package registration failed")
        raise PackageError("package could not be registered") from exc
    finally:
        conn.close()


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            log.warning("rollback failed", exc_info=True)


def _superseded_days(conn: sqlite3.Connection, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    superseded = []
    for entry in entries:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM versions WHERE day = ?", (entry["day"],)
        ).fetchone()
        latest = int(row["v"]) if row is not None and row["v"] is not None else None
        if latest is not None and latest > int(entry["version"]):
            superseded.append(
                {
                    "day": entry["day"],
                    "packaged_version": int(entry["version"]),
                    "latest_version": latest,
                }
            )
    return superseded


def _row_to_payload(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    package_id = int(row["package_id"])
    entries = json.loads(row["manifest"])
    superseded = _superseded_days(conn, entries)
    return {
        "package_id": package_id,
        "start_day": row["start_day"],
        "end_day": row["end_day"],
        "status": row["status"],
        "created_at": row["created_at"],
        "day_count": len(entries),
        "entries": entries,
        "archive": {
            "filename": f"rainmosaic-package-{package_id}.zip",
            "sha256": row["sha256"],
            "size_bytes": int(row["size_bytes"]),
        },
        "stale": bool(superseded),
        "superseded": superseded,
        "download_url": f"/api/v1/packages/{package_id}/download",
    }


def get_package(settings: Settings, package_id: int) -> dict[str, Any] | None:
    conn = store.connect(settings)
    try:
        row = conn.execute(
            "SELECT * FROM packages WHERE package_id = ?", (package_id,)
        ).fetchone()
        return _row_to_payload(conn, row) if row is not None else None
    finally:
        conn.close()


def list_packages(
    settings: Settings, *, limit: int = 20, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    conn = store.connect(settings)
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM packages").fetchone()["n"]
        rows = conn.execute(
            "SELECT * FROM packages ORDER BY package_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_payload(conn, row) for row in rows], int(total)
    finally:
        conn.close()


def get_package_download(settings: Settings, package_id: int) -> PackageDownload | None:
    conn = store.connect(settings)
    try:
        row = conn.execute(
            "SELECT * FROM packages WHERE package_id = ?", (package_id,)
        ).fetchone()
        if row is None:
            return None
        entries = json.loads(row["manifest"])
        stale = bool(_superseded_days(conn, entries))
        return PackageDownload(
            package_id=int(row["package_id"]),
            path=settings.packages_dir / f"p{int(row['package_id'])}.zip",
            filename=(
                f"rainmosaic-package-{int(row['package_id'])}"
                f"-{row['start_day']}-to-{row['end_day']}.zip"
            ),
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            stale=stale,
        )
    finally:
        conn.close()
