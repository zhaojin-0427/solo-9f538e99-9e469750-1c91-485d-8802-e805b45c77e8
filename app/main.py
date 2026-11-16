"""FastAPI application: submit two daily rainfall blocks, browse, download, freeze packages."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import __version__, packages, store
from .config import Settings, load_settings
from .limits import BodyLimitMiddleware
from .rasters import RasterInputError, load_block, validate_pair

log = logging.getLogger(__name__)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def parse_day(raw: str) -> str:
    value = (raw or "").strip()
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_day",
                "message": "day must be a calendar date formatted YYYY-MM-DD",
            },
        ) from exc
    if parsed.isoformat() != value:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_day",
                "message": "day must be a calendar date formatted YYYY-MM-DD",
            },
        )
    return value


def parse_captured_at(raw: str, slot: str) -> datetime:
    text = (raw or "").strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_captured_at",
                "slot": slot,
                "message": "captured_at must be an ISO-8601 timestamp with a UTC offset",
            },
        ) from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_captured_at",
                "slot": slot,
                "message": (
                    "captured_at must include a UTC offset "
                    "(for example 2026-01-01T12:00:00Z)"
                ),
            },
        )
    return value.astimezone(timezone.utc)


async def read_upload(upload: UploadFile, *, slot: str, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    await upload.close()
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "file_too_large",
                "slot": slot,
                "message": f"each uploaded file must be at most {limit} bytes",
            },
        )
    if not data:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_file", "slot": slot, "message": "uploaded file is empty"},
        )
    return data


def error_response(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return JSONResponse(status_code=status, content={"detail": detail})


class PackageRequest(BaseModel):
    start_day: str
    end_day: str


def validate_day_range(start_day: str, end_day: str) -> tuple[str, str]:
    start = parse_day(start_day)
    end = parse_day(end_day)
    span = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    if span < 1 or span > packages.MAX_PACKAGE_DAYS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_range",
                "message": (
                    "start_day must be on or before end_day and the range must cover "
                    f"1 to {packages.MAX_PACKAGE_DAYS} days; got {span} days"
                ),
                "start_day": start,
                "end_day": end,
            },
        )
    return start, end


def _validate_version_number(value: int) -> int:
    if value < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_expected_version",
                "message": "expected_version must be 0 or a positive version number",
            },
        )
    return value


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved = settings if settings is not None else load_settings()
        store.init_store(resolved)
        app.state.settings = resolved
        # Two files plus multipart framing, with a little headroom.
        app.state.body_limit_bytes = 2 * resolved.max_upload_bytes + 1024 * 1024
        log.info("rain-mosaic service ready; data dir %s", resolved.data_dir)
        yield

    app = FastAPI(
        title="Rainfall Mosaic Service",
        version=__version__,
        summary=(
            "Combine two daily single-band rainfall GeoTIFF blocks "
            "into one research-area mosaic."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    static_dir = Path(__file__).parent / "static" / "swagger-ui"
    app.mount("/static/swagger-ui", StaticFiles(directory=static_dir), name="swagger-ui")

    # Rejects oversized bodies before the multipart parser spools them to disk.
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Flatten request-model errors into the same {code, message} shape."""
        problems = exc.errors()
        first = problems[0] if problems else {}
        location = ".".join(
            str(part) for part in first.get("loc", ()) if part not in ("body", "query", "path")
        )
        message = first.get("msg", "invalid request")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": message,
                    "errors": [
                        {
                            "field": ".".join(
                                str(part)
                                for part in problem.get("loc", ())
                                if part not in ("body", "query", "path")
                            ),
                            "message": problem.get("msg", ""),
                            "type": problem.get("type", ""),
                        }
                        for problem in problems
                    ],
                }
            },
        )

    @app.get("/docs", include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        # Assets are served locally so /docs also works without internet access.
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — API documentation",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters={"persistAuthorization": True, "displayRequestDuration": True},
        )

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/healthz", tags=["ops"], summary="Liveness, database and version check")
    def healthz(request: Request) -> dict:
        current = get_settings(request)
        conn = store.connect(current)
        try:
            conn.execute("SELECT 1").fetchone()
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        finally:
            conn.close()
        return {"status": "ok", "current_version": store.get_current_version(current)}

    @app.post(
        "/api/v1/mosaics",
        status_code=201,
        tags=["mosaics"],
        summary="Submit two blocks for one day and publish a new mosaic version",
        description=(
            "Upload two single-band int16 GeoTIFF blocks that belong to the same observation "
            "day. Both must use "
            "the same CRS, the same resolution and an aligned pixel grid, with at most 64x64 "
            "pixels each, NoData = -32768.\n\n"
            "The union bounding rectangle of the two blocks is mosaicked. In overlapping "
            "pixels the block with the later UTC acquisition time wins; where that block is "
            "NoData the other block is used; when acquisition times are equal the larger "
            "block number wins. Pixels covered by neither block stay NoData.\n\n"
            "`day` labels the observation and is optional; when omitted it is taken from the "
            "earlier of the two UTC acquisition timestamps. Capture instants may straddle UTC "
            "midnight (daily totals are often read just after it).\n\n"
            "`expected_version` is the version the submitter last saw (0 when nothing is "
            "published). Publishing is compare-and-swap: concurrent submissions that started "
            "from the same expected version produce exactly one new version and the other "
            "receives 409. A version only becomes current after its image is completely "
            "written and registered, so a failed submission never disturbs the current "
            "version."
        ),
        responses={
            201: {"description": "New version published"},
            409: {"description": "Another submission published first"},
            413: {"description": "Uploaded file too large"},
            422: {"description": "Invalid parameters or raster blocks"},
        },
    )
    async def submit_mosaic(
        request: Request,
        block_a: Annotated[
            UploadFile,
            File(
                description="GeoTIFF for block A",
                json_schema_extra={"format": "binary"},
            ),
        ],
        block_b: Annotated[
            UploadFile,
            File(
                description="GeoTIFF for block B",
                json_schema_extra={"format": "binary"},
            ),
        ],
        block_a_no: Annotated[int, Form(description="Block number of block A")],
        block_a_captured_at: Annotated[
            str, Form(description="UTC acquisition time of block A, ISO-8601 with offset")
        ],
        block_b_no: Annotated[int, Form(description="Block number of block B")],
        block_b_captured_at: Annotated[
            str, Form(description="UTC acquisition time of block B, ISO-8601 with offset")
        ],
        expected_version: Annotated[
            int,
            Form(
                description=(
                    "Version the submitter last saw; 0 when no version exists yet"
                )
            ),
        ],
        day: Annotated[
            str | None,
            Form(description="UTC day of the observation, YYYY-MM-DD (derived if omitted)"),
        ] = None,
    ):
        current_settings = get_settings(request)
        _validate_version_number(expected_version)
        a_time = parse_captured_at(block_a_captured_at, "a")
        b_time = parse_captured_at(block_b_captured_at, "b")
        # The observation day labels the pair; when omitted it is taken from
        # the earlier of the two acquisition instants.  Capture instants may
        # straddle UTC midnight (daily totals are often read just after it).
        day_value = (
            parse_day(day)
            if day is not None
            else min(a_time, b_time).date().isoformat()
        )
        if block_a_no < 0 or block_b_no < 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_block_no",
                    "message": "block numbers must not be negative",
                },
            )
        if block_a_no == block_b_no:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "duplicate_block_no",
                    "message": (
                        "the two blocks must have different block numbers; "
                        f"both are {block_a_no}"
                    ),
                },
            )

        raw_a = await read_upload(block_a, slot="a", limit=current_settings.max_upload_bytes)
        raw_b = await read_upload(block_b, slot="b", limit=current_settings.max_upload_bytes)

        def work() -> dict[str, Any]:
            try:
                left = load_block(
                    raw_a,
                    slot="a",
                    block_no=block_a_no,
                    captured_at=a_time,
                    original_filename=block_a.filename or "block_a.tif",
                )
                right = load_block(
                    raw_b,
                    slot="b",
                    block_no=block_b_no,
                    captured_at=b_time,
                    original_filename=block_b.filename or "block_b.tif",
                )
                validate_pair(left, right)
            except RasterInputError as exc:
                raise HTTPException(status_code=422, detail=exc.as_detail()) from exc

            try:
                return store.publish(
                    current_settings,
                    day=day_value,
                    expected_version=expected_version,
                    left=left,
                    right=right,
                )
            except store.VersionConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "version_conflict",
                        "message": (
                            "another submission published first; "
                            f"current version is {exc.current_version}"
                        ),
                        "expected_version": exc.expected_version,
                        "current_version": exc.current_version,
                        "current": exc.current,
                    },
                ) from exc
            except store.PublishBusy as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "publish_busy",
                        "message": "another publication is still in progress; retry shortly",
                    },
                ) from exc
            except store.PublishError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "code": "publish_failed",
                        "message": (
                            "mosaic could not be generated or registered; "
                            "the current version is unchanged"
                        ),
                    },
                ) from exc

        return await run_in_threadpool(work)

    @app.get(
        "/api/v1/mosaics",
        tags=["mosaics"],
        summary="List published versions (newest first)",
    )
    def list_mosaics(
        request: Request,
        day: Annotated[str | None, Query(description="Filter by UTC day, YYYY-MM-DD")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        current_settings = get_settings(request)
        day_value = parse_day(day) if day is not None else None
        items, total = store.list_versions(
            current_settings, day=day_value, limit=limit, offset=offset
        )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get(
        "/api/v1/mosaics/current",
        tags=["mosaics"],
        summary="Show the currently published version",
        responses={404: {"description": "Nothing published yet"}},
    )
    def current_mosaic(request: Request) -> JSONResponse:
        version, payload = store.get_current(get_settings(request))
        if payload is None:
            return error_response(
                404, "no_current_version", "no mosaic version has been published yet"
            )
        return JSONResponse({"current_version": version, "version": payload})

    @app.get(
        "/api/v1/mosaics/{version}",
        tags=["mosaics"],
        summary="Show one version by number",
        responses={404: {"description": "Unknown version"}},
    )
    def get_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
    ) -> dict:
        payload = store.get_version(get_settings(request), version)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            )
        return {"version": payload}

    @app.api_route(
        "/api/v1/mosaics/{version}/download",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Image file missing"},
        },
    )
    @app.get(
        "/api/v1/mosaics/{version}/download",
        tags=["mosaics"],
        summary="Download the GeoTIFF for a fixed version",
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown version"},
            410: {"description": "Image file missing"},
        },
    )
    def download_mosaic_version(
        request: Request,
        version: Annotated[int, PathParam(ge=1, description="Version number")],
    ) -> FileResponse:
        record = store.get_download(get_settings(request), version)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "version_not_found",
                    "message": f"version {version} does not exist",
                },
            )
        return _file_response(record)

    @app.api_route(
        "/api/v1/download/current",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=FileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    @app.get(
        "/api/v1/download/current",
        tags=["mosaics"],
        summary="Download the current mosaic",
        response_class=FileResponse,
        responses={404: {"description": "Nothing published yet"}},
    )
    def download_current(request: Request) -> FileResponse:
        record = store.get_current_download(get_settings(request))
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "no_current_version",
                    "message": "no mosaic version has been published yet",
                },
            )
        return _file_response(record)

    @app.post(
        "/api/v1/packages",
        status_code=201,
        tags=["packages"],
        summary="Freeze the latest published version of each day into one ZIP package",
        description=(
            "Selects, at one consistent point in time, the newest published mosaic version "
            f"for every UTC day in `[start_day, end_day]` (1 to {packages.MAX_PACKAGE_DAYS} "
            "days), verifies each source GeoTIFF against its registered SHA-256 and packs "
            "the original files plus a `manifest.json` (day, version, SHA-256 per member) "
            "into one ZIP.\n\n"
            "The whole package fails when a day has no published version (422 "
            "`missing_days`) or a source file is missing/corrupted (500). Publishing keeps "
            "working while the package is built; if a newer version for a selected day is "
            "published before the package is registered, the build is aborted with 409 "
            "`package_conflict` and no downloadable half-package remains. A package becomes "
            "downloadable by number only after both the archive and its registration "
            "succeeded. Versions published after registration leave the package downloadable "
            "but mark it `stale` in query responses."
        ),
        responses={
            201: {"description": "Package registered and downloadable"},
            409: {"description": "A newer version appeared while packaging"},
            422: {"description": "Invalid range or days without a published version"},
            500: {"description": "Source file missing/corrupt or build failed"},
            503: {"description": "Another writer held the store too long"},
        },
    )
    async def create_package(request: Request, body: PackageRequest) -> JSONResponse:
        current_settings = get_settings(request)
        start_day, end_day = validate_day_range(body.start_day, body.end_day)

        def work() -> dict[str, Any]:
            return packages.create_package(current_settings, start_day=start_day, end_day=end_day)

        try:
            payload = await run_in_threadpool(work)
        except packages.PackageMissingDays as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_days",
                    "message": (
                        "no published version exists for these days in the range: "
                        + ", ".join(exc.days)
                    ),
                    "days": exc.days,
                },
            ) from exc
        except packages.PackageConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "package_conflict",
                    "message": (
                        "a newer version was published for a selected day while the "
                        "package was being built; no package was registered, retry to "
                        "freeze the newer versions"
                    ),
                    "conflicts": exc.conflicts,
                },
            ) from exc
        except packages.PackageSourceMissing as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "package_source_missing",
                    "message": str(exc),
                    "day": exc.day,
                    "version": exc.version,
                },
            ) from exc
        except packages.PackageSourceMismatch as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "package_source_mismatch",
                    "message": str(exc),
                    "day": exc.day,
                    "version": exc.version,
                    "expected_sha256": exc.expected,
                    "actual_sha256": exc.actual,
                },
            ) from exc
        except packages.PackageBusy as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "package_busy",
                    "message": "another writer is still in progress; retry shortly",
                },
            ) from exc
        except packages.PackageError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "package_failed",
                    "message": "package could not be generated or registered",
                },
            ) from exc
        return JSONResponse(payload, status_code=201)

    @app.get(
        "/api/v1/packages",
        tags=["packages"],
        summary="List frozen packages, newest first (stale flags included)",
    )
    def list_packages(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        items, total = packages.list_packages(get_settings(request), limit=limit, offset=offset)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.get(
        "/api/v1/packages/{package_id}",
        tags=["packages"],
        summary="Show one package, its manifest and staleness",
        responses={404: {"description": "Unknown package"}},
    )
    def get_package(
        request: Request,
        package_id: Annotated[int, PathParam(ge=1, description="Package number")],
    ) -> dict:
        payload = packages.get_package(get_settings(request), package_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "package_not_found",
                    "message": f"package {package_id} does not exist",
                },
            )
        return {"package": payload}

    @app.api_route(
        "/api/v1/packages/{package_id}/download",
        methods=["GET", "HEAD"],
        include_in_schema=False,
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown package"},
            410: {"description": "Archive file missing"},
        },
    )
    @app.get(
        "/api/v1/packages/{package_id}/download",
        tags=["packages"],
        summary="Download the frozen ZIP for a package",
        response_class=FileResponse,
        responses={
            404: {"description": "Unknown package"},
            410: {"description": "Archive file missing"},
        },
    )
    def download_package(
        request: Request,
        package_id: Annotated[int, PathParam(ge=1, description="Package number")],
    ) -> FileResponse:
        record = packages.get_package_download(get_settings(request), package_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "package_not_found",
                    "message": f"package {package_id} does not exist",
                },
            )
        if not record.path.is_file():
            log.error("archive for package %s is missing at %s", record.package_id, record.path)
            raise HTTPException(
                status_code=410,
                detail={
                    "code": "package_file_missing",
                    "message": f"archive for package {record.package_id} is unavailable on disk",
                },
            )
        return FileResponse(
            path=record.path,
            media_type="application/zip",
            filename=record.filename,
            headers={
                "ETag": f'"{record.sha256}"',
                "X-Package-Id": str(record.package_id),
                "X-Package-SHA256": record.sha256,
                "X-Package-Stale": "true" if record.stale else "false",
                # The bytes are immutable, but the staleness header is
                # computed live, so caches must revalidate (ETag makes that
                # cheap) instead of serving a stale flag.
                "Cache-Control": "no-cache",
            },
        )

    return app


def _file_response(record: store.DownloadRecord) -> FileResponse:
    if not record.path.is_file():
        log.error("image for version %s is missing at %s", record.version, record.path)
        raise HTTPException(
            status_code=410,
            detail={
                "code": "image_missing",
                "message": f"image for version {record.version} is unavailable on disk",
            },
        )
    return FileResponse(
        path=record.path,
        media_type="image/tiff",
        filename=record.filename,
        headers={
            "ETag": f'"{record.sha256}"',
            "X-Mosaic-Version": str(record.version),
            "X-Mosaic-SHA256": record.sha256,
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


app = create_app()
