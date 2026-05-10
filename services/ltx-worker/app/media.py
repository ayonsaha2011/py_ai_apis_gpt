from __future__ import annotations

import asyncio
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import settings

_ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp"},
    "video": {".mp4", ".mov", ".webm", ".mkv"},
    "audio": {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"},
}

_ALLOWED_MIME_PREFIXES = {
    "image": ("image/",),
    "video": ("video/",),
    "audio": ("audio/",),
}


async def fetch_media(url: str, job_dir: Path, name: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".bin"
    media_kind = "image" if name.startswith("keyframe") else name
    path = job_dir / f"{name}{suffix}"
    if parsed.scheme in {"r2", "s3"}:
        raise ValueError("direct r2/s3 media fetch is not enabled in the worker; provide signed https URLs")
    if parsed.scheme != "https":
        raise ValueError("media URLs must be https")
    if media_kind not in _ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported media kind {media_kind}")
    if suffix.lower() not in _ALLOWED_EXTENSIONS[media_kind]:
        raise ValueError(f"{media_kind} media extension {suffix or '<none>'} is not allowed")
    await _reject_private_destination(parsed.hostname)
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, read=600.0), follow_redirects=False) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type and not content_type.startswith(_ALLOWED_MIME_PREFIXES[media_kind]):
                raise ValueError(f"{media_kind} media content-type {content_type} is not allowed")
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = 0
                if declared_size > settings.max_media_bytes:
                    raise ValueError("media file exceeds configured byte limit")
            total = 0
            with path.open("wb") as handle:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > settings.max_media_bytes:
                        raise ValueError("media file exceeds configured byte limit")
                    handle.write(chunk)
    return path


async def _reject_private_destination(hostname: str | None) -> None:
    if not hostname:
        raise ValueError("media URL must include a hostname")
    infos = await asyncio.to_thread(socket.getaddrinfo, hostname, None, type=socket.SOCK_STREAM)
    if not infos:
        raise ValueError("media URL hostname could not be resolved")
    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("media URL resolves to a blocked private or local network address")
