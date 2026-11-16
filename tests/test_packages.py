"""Package behaviour: freezing, conflicts, staleness, integrity, durability."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from app import packages, rasters, store
from tests.conftest import make_geotiff, submit_payload, utc

DAY1 = "2026-01-01"
DAY2 = "2026-01-02"
DAY3 = "2026-01-03"


def _submit(client, *, expected_version: int, day: str = DAY1, fill: int = 1):
    data, files = submit_payload(
        block_a=make_geotiff(np.full((2, 2), fill, dtype=np.int16)),
        block_b=make_geotiff(np.full((2, 2), fill + 1, dtype=np.int16), x0=0.01),
        expected_version=expected_version,
        day=day,
        a_time=f"{day}T01:00:00Z",
        b_time=f"{day}T02:00:00Z",
    )
    return client.post("/api/v1/mosaics", data=data, files=files)


def _publish_direct(settings, *, day: str, fill: int = 1):
    """Publish a new version straight through the store (simulates a submitter)."""
    left = rasters.load_block(
        make_geotiff(np.full((2, 2), fill, dtype=np.int16)),
        slot="a",
        block_no=1,
        captured_at=utc(f"{day}T01:00:00Z"),
        original_filename="a.tif",
    )
    right = rasters.load_block(
        make_geotiff(np.full((2, 2), fill + 1, dtype=np.int16), x0=0.01),
        slot="b",
        block_no=2,
        captured_at=utc(f"{day}T02:00:00Z"),
        original_filename="b.tif",
    )
    return store.publish(
        settings,
        day=day,
        expected_version=store.get_current_version(settings),
        left=left,
        right=right,
    )


def _create(client, start_day: str = DAY1, end_day: str = DAY3):
    return client.post(
        "/api/v1/packages", json={"start_day": start_day, "end_day": end_day}
    )


def _read_zip(content: bytes) -> tuple[dict[str, bytes], dict]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    return members, manifest


def test_package_freezes_latest_version_of_each_day(client):
    assert _submit(client, expected_version=0, day=DAY1, fill=1).status_code == 201
    assert _submit(client, expected_version=1, day=DAY1, fill=3).status_code == 201  # v2, same day
    assert _submit(client, expected_version=2, day=DAY2, fill=5).status_code == 201  # v3
    assert _submit(client, expected_version=3, day=DAY3, fill=7).status_code == 201  # v4

    response = _create(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["package_id"] == 1
    assert body["start_day"] == DAY1 and body["end_day"] == DAY3
    assert body["status"] == "ready"
    assert body["day_count"] == 3
    assert body["stale"] is False
    assert body["superseded"] == []
    assert body["download_url"] == "/api/v1/packages/1/download"
    # Day 1 must be frozen at its latest version (v2), not the older v1.
    assert [entry["version"] for entry in body["entries"]] == [2, 3, 4]
    assert [entry["day"] for entry in body["entries"]] == [DAY1, DAY2, DAY3]

    download = client.get("/api/v1/packages/1/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.headers["x-package-id"] == "1"
    assert download.headers["x-package-stale"] == "false"
    assert "rainmosaic-package-1-2026-01-01-to-2026-01-03.zip" in download.headers[
        "content-disposition"
    ]
    assert hashlib.sha256(download.content).hexdigest() == body["archive"]["sha256"]
    assert len(download.content) == body["archive"]["size_bytes"]

    members, manifest = _read_zip(download.content)
    assert sorted(members) == [
        "rainmosaic-2026-01-01-v2.tif",
        "rainmosaic-2026-01-02-v3.tif",
        "rainmosaic-2026-01-03-v4.tif",
    ]
    # The packaged bytes are exactly the bytes of the frozen versions.
    for version in (2, 3, 4):
        original = client.get(f"/api/v1/mosaics/{version}/download").content
        day = {2: DAY1, 3: DAY2, 4: DAY3}[version]
        assert members[f"rainmosaic-{day}-v{version}.tif"] == original

    assert manifest["format"] == "rainmosaic-package/1"
    assert manifest["start_day"] == DAY1 and manifest["end_day"] == DAY3
    assert [entry["day"] for entry in manifest["entries"]] == [DAY1, DAY2, DAY3]
    for entry in manifest["entries"]:
        data = members[entry["filename"]]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert len(data) == entry["size_bytes"]
        # The manifest SHA-256 is the one registered for that version.
        registered = client.get(f"/api/v1/mosaics/{entry['version']}").json()["version"]
        assert entry["sha256"] == registered["image"]["sha256"]


def test_single_day_range_is_allowed(client):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    response = _create(client, DAY1, DAY1)
    assert response.status_code == 201, response.text
    assert response.json()["day_count"] == 1


@pytest.mark.parametrize(
    "start_day, end_day",
    [
        (DAY3, DAY1),  # end before start
        ("2026-01-01", "2026-01-08"),  # 8 days > 7
        ("2026-01-01", "2026-01-31"),  # far beyond the limit
    ],
)
def test_invalid_range_is_rejected(client, start_day, end_day):
    response = _create(client, start_day, end_day)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_range"
    assert client.get("/api/v1/packages").json()["total"] == 0


def test_malformed_days_are_rejected(client):
    response = _create(client, "01-01-2026", DAY1)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_day"
    assert client.post("/api/v1/packages", json={}).status_code == 422


def test_missing_days_fail_the_whole_package(client):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    response = _create(client, DAY1, DAY3)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "missing_days"
    assert detail["days"] == [DAY2, DAY3]
    assert client.get("/api/v1/packages").json()["total"] == 0
    assert client.get("/api/v1/packages/1").status_code == 404


def test_conflict_when_new_version_appears_during_build(client, settings, monkeypatch):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    assert _submit(client, expected_version=1, day=DAY2).status_code == 201

    real_builder = packages.build_package_zip

    def racing_builder(settings_arg, selections, temp_path, **kwargs):
        # A submitter publishes a newer version for DAY1 mid-build.
        _publish_direct(settings, day=DAY1, fill=9)
        return real_builder(settings_arg, selections, temp_path, **kwargs)

    monkeypatch.setattr(packages, "build_package_zip", racing_builder)
    response = _create(client, DAY1, DAY2)

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "package_conflict"
    assert detail["conflicts"] == [
        {"day": DAY1, "packaged_version": 1, "current_version": 3}
    ]
    # No downloadable half-package is left behind.
    assert client.get("/api/v1/packages").json()["total"] == 0
    assert client.get("/api/v1/packages/1").status_code == 404
    assert client.get("/api/v1/packages/1/download").status_code == 404
    assert list(settings.packages_dir.glob("*")) == []
    # The racing publication itself succeeded and is current.
    assert client.get("/api/v1/mosaics/current").json()["current_version"] == 3


def test_unrelated_publish_during_build_does_not_conflict(client, settings, monkeypatch):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    assert _submit(client, expected_version=1, day=DAY2).status_code == 201

    real_builder = packages.build_package_zip

    def racing_builder(settings_arg, selections, temp_path, **kwargs):
        _publish_direct(settings, day="2026-01-10", fill=9)  # outside the range
        return real_builder(settings_arg, selections, temp_path, **kwargs)

    monkeypatch.setattr(packages, "build_package_zip", racing_builder)
    response = _create(client, DAY1, DAY2)
    assert response.status_code == 201, response.text
    assert response.json()["stale"] is False


def test_package_not_downloadable_until_registered(client, settings, monkeypatch):
    import threading

    assert _submit(client, expected_version=0, day=DAY1).status_code == 201

    started = threading.Event()
    release = threading.Event()
    real_builder = packages.build_package_zip

    def slow_builder(settings_arg, selections, temp_path, **kwargs):
        started.set()
        assert release.wait(timeout=30), "package build never released"
        return real_builder(settings_arg, selections, temp_path, **kwargs)

    monkeypatch.setattr(packages, "build_package_zip", slow_builder)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_create, client, DAY1, DAY1)
        assert started.wait(timeout=10), "package build did not start"
        try:
            assert client.get("/api/v1/packages").json()["total"] == 0
            assert client.get("/api/v1/packages/1").status_code == 404
            assert client.get("/api/v1/packages/1/download").status_code == 404
        finally:
            release.set()
        response = pending.result(timeout=60)

    assert response.status_code == 201, response.text
    assert client.get("/api/v1/packages/1/download").status_code == 200


def test_package_marked_stale_after_later_publish_but_stays_downloadable(client):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    assert _submit(client, expected_version=1, day=DAY2).status_code == 201
    created = _create(client, DAY1, DAY2)
    assert created.status_code == 201, created.text
    assert created.json()["stale"] is False
    original_zip = client.get("/api/v1/packages/1/download").content

    # A version published after registration supersedes DAY1 in the package.
    assert _submit(client, expected_version=2, day=DAY1, fill=9).status_code == 201

    shown = client.get("/api/v1/packages/1")
    assert shown.status_code == 200
    package = shown.json()["package"]
    assert package["stale"] is True
    assert package["superseded"] == [
        {"day": DAY1, "packaged_version": 1, "latest_version": 3}
    ]
    # The frozen entries still reference the original versions.
    assert [entry["version"] for entry in package["entries"]] == [1, 2]

    listing = client.get("/api/v1/packages").json()
    assert listing["total"] == 1
    assert listing["items"][0]["stale"] is True
    assert listing["items"][0]["superseded"][0]["day"] == DAY1

    download = client.get("/api/v1/packages/1/download")
    assert download.status_code == 200
    assert download.headers["x-package-stale"] == "true"
    assert download.content == original_zip
    members, manifest = _read_zip(download.content)
    assert [entry["version"] for entry in manifest["entries"]] == [1, 2]


def test_missing_source_file_fails_package(client, settings):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    (settings.versions_dir / "v1.tif").unlink()

    response = _create(client, DAY1, DAY1)
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "package_source_missing"
    assert detail["day"] == DAY1 and detail["version"] == 1
    assert client.get("/api/v1/packages").json()["total"] == 0
    assert list(settings.packages_dir.glob("*")) == []


def test_corrupted_source_file_fails_package(client, settings):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    original = (settings.versions_dir / "v1.tif").read_bytes()
    (settings.versions_dir / "v1.tif").write_bytes(b"tampered" + original[8:])

    response = _create(client, DAY1, DAY1)
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "package_source_mismatch"
    assert detail["day"] == DAY1 and detail["version"] == 1
    assert detail["expected_sha256"] != detail["actual_sha256"]
    assert client.get("/api/v1/packages").json()["total"] == 0
    assert list(settings.packages_dir.glob("*")) == []


def test_packages_survive_restart(restartable_client):
    first = restartable_client()
    assert _submit(first, expected_version=0, day=DAY1).status_code == 201
    assert _submit(first, expected_version=1, day=DAY2).status_code == 201
    created = _create(first, DAY1, DAY2)
    assert created.status_code == 201, created.text
    before = created.json()
    zip_before = first.get("/api/v1/packages/1/download").content
    first.__exit__(None, None, None)

    restarted = restartable_client()
    shown = restarted.get("/api/v1/packages/1")
    assert shown.status_code == 200
    assert shown.json()["package"] == before
    listing = restarted.get("/api/v1/packages").json()
    assert listing["total"] == 1
    assert listing["items"][0]["package_id"] == 1
    download = restarted.get("/api/v1/packages/1/download")
    assert download.status_code == 200
    assert download.content == zip_before
    members, manifest = _read_zip(download.content)
    assert [entry["version"] for entry in manifest["entries"]] == [1, 2]


def test_unknown_packages_return_404(client):
    assert client.get("/api/v1/packages").json()["total"] == 0
    assert client.get("/api/v1/packages/99").status_code == 404
    assert client.get("/api/v1/packages/99").json()["detail"]["code"] == "package_not_found"
    assert client.get("/api/v1/packages/99/download").status_code == 404
    assert client.get("/api/v1/packages/0").status_code == 422
    assert client.get("/api/v1/packages/0/download").status_code == 422


def test_registered_package_with_missing_file_returns_410(client, settings):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    assert _create(client, DAY1, DAY1).status_code == 201
    (settings.packages_dir / "p1.zip").unlink()

    response = client.get("/api/v1/packages/1/download")
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "package_file_missing"
    # Metadata remains available even though the archive is gone.
    assert client.get("/api/v1/packages/1").status_code == 200


def test_packages_list_paginates(client):
    assert _submit(client, expected_version=0, day=DAY1).status_code == 201
    assert _create(client, DAY1, DAY1).status_code == 201
    assert _create(client, DAY1, DAY1).status_code == 201

    listing = client.get("/api/v1/packages").json()
    assert listing["total"] == 2
    assert [item["package_id"] for item in listing["items"]] == [2, 1]
    page = client.get("/api/v1/packages", params={"limit": 1, "offset": 1}).json()
    assert [item["package_id"] for item in page["items"]] == [1]
