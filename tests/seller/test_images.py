"""Product images as model input (C3b): host allow-list, shrink, content-addressed store, vision flag."""

from __future__ import annotations

import base64
import hashlib
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image

from sellerops.seller import SellerError
from sellerops.seller.images import (
    LocalImageStore,
    ProductImages,
    check_url,
    fetch_url,
    shrink,
    to_content_block,
)

URL = "https://cdn.dummyjson.com/product-images/dresses/dress-pea/thumbnail.webp"


def picture(size=(2000, 1000), mode="RGB", color=(30, 60, 200), fmt="PNG") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color).save(out, fmt)
    return out.getvalue()


class Counter:
    def __init__(self, data: bytes) -> None:
        self.data, self.calls = data, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self.data


@pytest.fixture
def store(tmp_path):
    return LocalImageStore(tmp_path, store_id="kala-studio")


def test_prepare_shrinks_hashes_and_stores(store, tmp_path) -> None:
    fetch = Counter(picture())
    loaded = ProductImages(store, fetch=fetch).prepare(URL)
    ref = loaded.ref
    assert (ref.width, ref.height, ref.mime_type, ref.source_url) == (1024, 512, "image/jpeg", URL)
    assert ref.sha256 == hashlib.sha256(loaded.data).hexdigest() and ref.size_bytes == len(loaded.data)
    assert (tmp_path / "stores" / "kala-studio" / "images" / f"{ref.sha256}.jpg").read_bytes() == loaded.data
    assert Image.open(io.BytesIO(loaded.data)).format == "JPEG"

    again = ProductImages(store, fetch=fetch).prepare(URL)      # cache hit: no second download
    assert again == loaded and fetch.calls == [URL]


def test_stores_are_separate(tmp_path) -> None:
    fetch = Counter(picture())
    ref = ProductImages(LocalImageStore(tmp_path, store_id="a"), fetch=fetch).prepare(URL).ref
    other = LocalImageStore(tmp_path, store_id="b")
    assert other.get(ref.sha256) is None and other.lookup(URL) is None


def test_transparent_image_is_flattened_on_white() -> None:
    data, width, height = shrink(picture((100, 100), "RGBA", (0, 0, 0, 0)))
    assert (width, height) == (100, 100)
    assert Image.open(io.BytesIO(data)).convert("RGB").getpixel((50, 50)) == (255, 255, 255)


def test_small_image_is_not_enlarged() -> None:
    _, width, height = shrink(picture((300, 200)))
    assert (width, height) == (300, 200)


@pytest.mark.parametrize("url", [
    "http://cdn.dummyjson.com/x.png",              # not https
    "https://evil.example.com/x.png",              # unknown host
    "https://cdn.dummyjson.com.evil.com/x.png",    # look-alike host
    "https://user@cdn.dummyjson.com/x.png",        # credentials
    "https://cdn.dummyjson.com:8443/x.png",        # odd port
    "file:///etc/passwd",
    "https://169.254.169.254/latest/meta-data",
])
def test_only_catalog_hosts_are_fetched(url, store) -> None:
    fetch = Counter(picture())
    with pytest.raises(SellerError) as error:
        ProductImages(store, fetch=fetch).prepare(url)
    assert error.value.code == "image_unavailable" and fetch.calls == []
    with pytest.raises(SellerError):
        fetch_url(url)                               # refused before any network call


def test_redirect_to_another_host_is_refused() -> None:
    from sellerops.seller.images import _AllowedRedirects

    handler = _AllowedRedirects(frozenset({"cdn.dummyjson.com"}))
    with pytest.raises(SellerError):
        handler.redirect_request(None, None, 302, "Found", {}, "http://10.0.0.1/admin")
    check_url("https://cdn.dummyjson.com/ok.png", {"cdn.dummyjson.com"})   # allowed: no error


@pytest.mark.parametrize("data", [b"not an image", b"", picture(fmt="BMP")])
def test_unreadable_or_unsupported_images_are_refused(data) -> None:
    with pytest.raises(SellerError) as error:
        shrink(data)
    assert error.value.code == "image_unavailable"


def test_load_skips_failures_and_respects_limit(store) -> None:
    def fetch(url: str) -> bytes:
        return b"broken" if url.endswith("1.webp") else picture()

    urls = [URL.replace("thumbnail", "1"), URL, URL.replace("thumbnail", "2")]
    loaded = ProductImages(store, fetch=fetch).load(urls, limit=1)
    assert [item.ref.source_url for item in loaded] == [URL]


def test_content_block_is_langchain_standard(store) -> None:
    loaded = ProductImages(store, fetch=Counter(picture())).prepare(URL)
    block = to_content_block(loaded)
    assert block["type"] == "image" and block["mime_type"] == "image/jpeg"
    assert base64.b64decode(block["base64"]) == loaded.data


def test_service_lists_catalog_images_and_records_inputs(service, store) -> None:
    urls = service.product_images("FS180")
    assert urls[0].endswith("thumbnail.webp") and len(urls) == len(set(urls)) > 1
    assert all(url.startswith("https://cdn.dummyjson.com/") for url in urls)

    ref = ProductImages(store, fetch=Counter(picture())).prepare(urls[0]).ref
    from dataclasses import asdict

    proposal = service.propose_caption("FS180", "describe the colour", actor="agent",
                                       description="A blue dress.", inputs={"images": [asdict(ref)]})
    assert service.get_proposal(proposal["proposal_id"])["inputs"]["images"][0]["sha256"] == ref.sha256
    plain = service.propose_caption("FS181", "shorter", actor="agent", description="Short.")
    assert plain["inputs"] is None


def test_old_database_gets_the_inputs_column(tmp_path) -> None:
    import sqlite3

    from sellerops.seller.db import SCHEMA, init_db

    old_schema = SCHEMA.replace("reverted_by          TEXT,\n    inputs", "reverted_by          TEXT\n    -- inputs")
    assert old_schema != SCHEMA
    conn = sqlite3.connect(tmp_path / "old.sqlite")
    conn.executescript(old_schema)               # a database from before C3b
    assert "inputs" not in {row[1] for row in conn.execute("PRAGMA table_info(caption_proposals)")}
    init_db(conn)
    assert "inputs" in {row[1] for row in conn.execute("PRAGMA table_info(caption_proposals)")}
    init_db(conn)   # idempotent


def test_supports_vision() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from sellerops.harness.models import supports_vision

    class Claude(FakeListChatModel):
        @property
        def _llm_type(self) -> str:
            return "anthropic-chat"

    text_only = FakeListChatModel(responses=["x"])
    assert supports_vision(Claude(responses=["x"]))
    assert supports_vision(Claude(responses=["x"]).with_config(tags=["t"]))
    assert not supports_vision(text_only)
    assert not supports_vision(Claude(responses=["x"]).with_fallbacks([text_only]))


def test_seller_api_serves_stored_images(service, store) -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app

    ref = ProductImages(store, fetch=Counter(picture())).prepare(URL).ref
    app.dependency_overrides[seller_api.get_service] = lambda: service
    app.dependency_overrides[seller_api.get_image_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.get(f"/api/seller/images/{ref.sha256}")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert hashlib.sha256(response.content).hexdigest() == ref.sha256
        for bad in ("0" * 64, "../../etc/passwd", ref.sha256.upper()):
            missing = client.get(f"/api/seller/images/{bad}")
            assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()
