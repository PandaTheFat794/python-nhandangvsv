"""
Download files from a *public* Figshare article using the Figshare REST API (v2).

Why:
- Figshare web pages may block automated crawlers.
- The REST API typically returns a direct, stable `download_url` for each file.

References:
- Figshare REST API is accessible at https://api.figshare.com/v2 (public docs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests
from tqdm import tqdm

FIGSHARE_API = "https://api.figshare.com/v2"


class FigshareError(RuntimeError):
    pass


def _get_json(url: str, timeout: int = 60) -> object:
    r = requests.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise FigshareError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    # Figshare returns JSON
    return r.json()


def list_article_files(article_id: int) -> List[dict]:
    """
    Returns a list of file metadata dicts.

    Common fields include:
      - id
      - name
      - size
      - is_link_only
    """
    url = f"{FIGSHARE_API}/articles/{article_id}/files"
    data = _get_json(url)
    if isinstance(data, list):
        return data
    raise FigshareError(f"Unexpected response from {url}: {type(data)}")


def get_file_details(article_id: int, file_id: int) -> dict:
    """
    Returns file metadata including:
      - download_url (often https://ndownloader.figshare.com/files/<id>)
    """
    url = f"{FIGSHARE_API}/articles/{article_id}/files/{file_id}"
    data = _get_json(url)
    if isinstance(data, dict):
        return data
    raise FigshareError(f"Unexpected response from {url}: {type(data)}")


def download_url_to_path(download_url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(download_url, stream=True, timeout=120) as r:
        if r.status_code >= 400:
            raise FigshareError(f"GET {download_url} -> {r.status_code}: {r.text[:200]}")

        total = int(r.headers.get("content-length", "0") or "0")
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name)

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def compile_patterns(patterns: Optional[Sequence[str]]) -> List[re.Pattern]:
    if not patterns:
        return []
    return [re.compile(p) for p in patterns]


def name_matches(name: str, include: List[re.Pattern], exclude: List[re.Pattern]) -> bool:
    if include and not any(p.search(name) for p in include):
        return False
    if exclude and any(p.search(name) for p in exclude):
        return False
    return True


def download_article_files(
    article_id: int,
    out_dir: Path,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    include_p = compile_patterns(include)
    exclude_p = compile_patterns(exclude)

    files = list_article_files(article_id)
    selected = [f for f in files if name_matches(f.get("name", ""), include_p, exclude_p)]
    if max_files is not None:
        selected = selected[:max_files]

    downloaded: List[Path] = []

    for f in selected:
        fid = int(f["id"])
        name = str(f["name"])
        details = get_file_details(article_id, fid)
        download_url = details.get("download_url") or details.get("download_url", "")
        if not download_url:
            # Some responses may include 'download_url' but if not, try 'download_url' on parent list entry.
            download_url = f.get("download_url", "")

        if not download_url:
            raise FigshareError(f"Missing download_url for file id={fid} name={name}")

        out_path = out_dir / name
        if out_path.exists() and out_path.stat().st_size == int(details.get("size", out_path.stat().st_size)):
            print(f"[skip] {out_path} already exists")
            downloaded.append(out_path)
            continue

        print(f"[download] {name}")
        download_url_to_path(download_url, out_path)
        downloaded.append(out_path)

    return downloaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Download files from a public Figshare article (v2 API).")
    ap.add_argument("--article-id", type=int, required=True, help="Figshare article id, e.g. 22022540")
    ap.add_argument("--out-dir", type=str, default="data/raw", help="Output directory")
    ap.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Regex patterns to include (e.g. '.*\\.jpg$' 'annot_YOLO\\.zip')",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Regex patterns to exclude",
    )
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of files (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    downloaded = download_article_files(
        article_id=args.article_id,
        out_dir=out_dir,
        include=args.include,
        exclude=args.exclude,
        max_files=args.max_files,
    )
    print(f"Downloaded {len(downloaded)} file(s) to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
