"""The built web app is served next to the API: real files, SPA fallback, and no leaks."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sellerops.store.frontend import install_frontend


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><div id=root></div>")
    (root / "assets" / "app-abc123.js").write_text("console.log(1)")
    (root / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside dist")
    return root


@pytest.fixture
def client(dist: Path) -> TestClient:
    app = FastAPI()

    @app.get("/api/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    assert install_frontend(app, dist) is True
    return TestClient(app)


def test_index_for_client_routes(client: TestClient) -> None:
    for path in ("/", "/p/1", "/seller", "/seller/CP-0001"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "<div id=root>" in response.text
        assert response.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_cached_for_a_year(client: TestClient) -> None:
    response = client.get("/assets/app-abc123.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]


def test_real_files_in_dist_are_served(client: TestClient) -> None:
    assert client.get("/favicon.svg").text == "<svg/>"


def test_unknown_api_paths_stay_json_404(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_existing_api_routes_win(client: TestClient) -> None:
    assert client.get("/api/ping").json() == {"ok": "yes"}


def test_path_traversal_does_not_leave_dist(client: TestClient) -> None:
    response = client.get("/..%2Fsecret.txt")
    assert "outside dist" not in response.text
    assert "<div id=root>" in response.text  # falls back to index.html


def test_no_build_mounts_nothing(tmp_path: Path) -> None:
    app = FastAPI()
    assert install_frontend(app, tmp_path / "missing") is False
    assert TestClient(app).get("/").status_code == 404
