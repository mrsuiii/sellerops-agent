"""Product images as model input (C3b): fetch, shrink, store by content hash.

The model never gets an image URL. The backend fetches the catalog image
itself, from known hosts only, shrinks it, stores the result under its
SHA-256 and hands the bytes to the caller, which builds the model message.
The same bytes always give the same hash, so a proposal can record exactly
which images the model saw (``ImageRef.sha256``), and a replay or test sees
the same input.

    store = LocalImageStore(settings.project_root / "var" / "images", store_id=service.store_id)
    images = ProductImages(store)
    loaded = images.load(service.product_images("FS180"), limit=1)
    blocks = [to_content_block(item) for item in loaded]   # LangChain standard image blocks

Storage sits behind ``ImageStore``. ``LocalImageStore`` writes files under
``<root>/stores/<store_id>/images/<sha256>.jpg``; the same key layout works
for an object store (S3, R2, MinIO) when images must outlive one machine or
sellers upload their own photos: add an implementation, keep the callers.

Preparing images needs Pillow (``[agent]`` extra), imported only in
``shrink``; reading stored images does not, so the store image (``[store]``)
can serve them to the seller page.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .analytics import SellerError

logger = logging.getLogger(__name__)

ALLOWED_HOSTS = frozenset({"cdn.dummyjson.com"})  # where the catalog's images live
MAX_SIDE = 1024            # longest side sent to the model: enough for a caption, ~1-1.5k tokens
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000   # refuse decompression bombs before decoding
SOURCE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF"})
OUTPUT_MIME = "image/jpeg"   # every provider with vision reads JPEG
JPEG_QUALITY = 85

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class ImageRef:
    """What a proposal records about an image the model saw."""

    sha256: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    source_url: str | None = None


@dataclass(frozen=True)
class LoadedImage:
    ref: ImageRef
    data: bytes


class ImageStore(Protocol):
    """Content-addressed image storage for one store (``store_id``)."""

    def put(self, data: bytes, ref: ImageRef) -> None: ...
    def get(self, sha256: str) -> bytes | None: ...
    def lookup(self, source_url: str) -> ImageRef | None:
        """The prepared image for a source URL, so it is fetched only once."""
        ...


class LocalImageStore:
    """Files under ``<root>/stores/<store_id>/images/`` (``var/images`` by default)."""

    def __init__(self, root: Path, *, store_id: str) -> None:
        self.dir = Path(root) / "stores" / store_id / "images"
        self.sources = Path(root) / "stores" / store_id / "sources"

    def put(self, data: bytes, ref: ImageRef) -> None:
        _write_atomic(self.dir / f"{ref.sha256}.jpg", data)
        if ref.source_url:
            _write_atomic(self.sources / f"{_url_key(ref.source_url)}.json", json.dumps(asdict(ref)).encode())

    def get(self, sha256: str) -> bytes | None:
        if not is_sha256(sha256):
            return None
        path = self.dir / f"{sha256}.jpg"
        return path.read_bytes() if path.is_file() else None

    def lookup(self, source_url: str) -> ImageRef | None:
        path = self.sources / f"{_url_key(source_url)}.json"
        if not path.is_file():
            return None
        ref = ImageRef(**json.loads(path.read_text()))
        return ref if (self.dir / f"{ref.sha256}.jpg").is_file() else None


class ProductImages:
    """Turns catalog image URLs into prepared, stored images."""

    def __init__(self, store: ImageStore, *, fetch: Fetch | None = None,
                 allowed_hosts: Iterable[str] = ALLOWED_HOSTS, max_side: int = MAX_SIDE) -> None:
        self.store = store
        self.allowed_hosts = frozenset(allowed_hosts)
        self.fetch = fetch or (lambda url: fetch_url(url, allowed_hosts=self.allowed_hosts))
        self.max_side = max_side

    def prepare(self, url: str) -> LoadedImage:
        """Fetch (or reuse) one image; raises ``SellerError("image_unavailable")``."""
        check_url(url, self.allowed_hosts)
        cached = self.store.lookup(url)
        if cached is not None and (data := self.store.get(cached.sha256)) is not None:
            return LoadedImage(cached, data)
        data, width, height = shrink(self.fetch(url), self.max_side)
        ref = ImageRef(sha256=hashlib.sha256(data).hexdigest(), mime_type=OUTPUT_MIME, width=width,
                       height=height, size_bytes=len(data), source_url=url)
        self.store.put(data, ref)
        return LoadedImage(ref, data)

    def load(self, urls: Iterable[str], *, limit: int = 1) -> list[LoadedImage]:
        """Up to ``limit`` images; one that fails is logged and skipped (the caller goes on with text)."""
        loaded: list[LoadedImage] = []
        for url in urls:
            if len(loaded) >= limit:
                break
            try:
                loaded.append(self.prepare(url))
            except SellerError as error:
                logger.warning("product image skipped", extra={"url": url, "reason": error.message})
        return loaded


def to_content_block(image: LoadedImage) -> dict[str, Any]:
    """A LangChain standard image content block (works with every provider that has vision)."""
    return {"type": "image", "base64": base64.b64encode(image.data).decode("ascii"),
            "mime_type": image.ref.mime_type}


# ---------------------------------------------------------------- helpers


def check_url(url: str, allowed_hosts: Iterable[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in set(allowed_hosts) or parts.username or parts.port:
        raise SellerError("image_unavailable", f"image host not allowed: {url}")


class _AllowedRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another allowed host (no detour to an internal address)."""

    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        check_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(url: str, *, allowed_hosts: Iterable[str] = ALLOWED_HOSTS, timeout: float = 10.0,
              max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    hosts = frozenset(allowed_hosts)
    check_url(url, hosts)
    opener = urllib.request.build_opener(_AllowedRedirects(hosts))
    request = urllib.request.Request(url, headers={"User-Agent": "sellerops-image-fetch/1.0"})
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except SellerError:
        raise
    except OSError as error:   # URLError, HTTPError, timeouts
        raise SellerError("image_unavailable", f"could not fetch {url}: {error}") from error
    if len(data) > max_bytes:
        raise SellerError("image_unavailable", f"image larger than {max_bytes} bytes: {url}")
    return data


def shrink(data: bytes, max_side: int = MAX_SIDE) -> tuple[bytes, int, int]:
    """Decode, check, flatten onto white, fit into ``max_side``, re-encode as JPEG (drops metadata)."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in SOURCE_FORMATS:
                raise SellerError("image_unavailable", f"unsupported image format {source.format}")
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise SellerError("image_unavailable", "image has too many pixels")
            source.seek(0)                       # first frame of a GIF / animated WebP
            image = source.convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise SellerError("image_unavailable", f"not a readable image: {error}") from error
    flat = Image.new("RGB", image.size, "white")
    flat.paste(image, mask=image.getchannel("A"))
    flat.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    flat.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue(), flat.width, flat.height


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
