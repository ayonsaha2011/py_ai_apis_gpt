from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import settings


async def fetch_media(url: str, job_dir: Path, name: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".bin"
    path = job_dir / f"{name}{suffix}"
    if parsed.scheme in {"r2", "s3"}:
        raise ValueError("direct r2/s3 media fetch is not enabled in the worker; provide signed https URLs")
    if parsed.scheme != "https":
        raise ValueError("media URLs must be https")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=600.0)) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = 0
            with path.open("wb") as handle:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > settings.max_media_bytes:
                        raise ValueError("media file exceeds configured byte limit")
                    handle.write(chunk)
    return path

