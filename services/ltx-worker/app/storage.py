from __future__ import annotations

import shutil
from pathlib import Path

import aioboto3

from .config import settings


async def store_video(path: Path, key: str) -> str:
    if settings.r2_bucket and settings.r2_account_id and settings.r2_access_key_id and settings.r2_secret_access_key:
        endpoint = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
        session = aioboto3.Session()
        async with session.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
        ) as client:
            await client.upload_file(str(path), settings.r2_bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        return f"{settings.r2_public_base_url.rstrip('/')}/{key}" if settings.r2_public_base_url else f"r2://{settings.r2_bucket}/{key}"

    local_path = settings.local_storage_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, local_path)
    return str(local_path.resolve())

