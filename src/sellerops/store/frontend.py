"""Serve the built web app (``web/dist``, from ``npm run build``) next to the API.

Vite writes hashed files under ``assets/``; those are cached forever. Every other
path that is not an API route returns ``index.html`` so the client-side router can
handle it (``/p/<id>``, ``/seller/<id>``). Missing ``/api`` paths stay JSON 404s.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

# Served by FastAPI itself or reserved for the API; never answered with index.html.
_API_PREFIXES = ("api/", "docs", "redoc", "openapi.json")


class _ImmutableStatic(StaticFiles):
    """Hashed build output: safe to cache for a year because a change means a new file name."""

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def install_frontend(app: FastAPI, dist: Path) -> bool:
    """Mount ``dist`` on ``app``. Returns False (and mounts nothing) if there is no build yet."""
    index = dist / "index.html"
    if not index.is_file():
        return False
    root = dist.resolve()
    index = root / "index.html"
    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", _ImmutableStatic(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith(_API_PREFIXES):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (root / path).resolve() if path else index
        # A real file in the build (favicon, robots.txt) wins; anything outside dist is refused.
        if candidate != index and root in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return True
