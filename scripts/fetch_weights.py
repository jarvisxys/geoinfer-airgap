"""Download pretrained encoder weights once into the local registry (needs internet).

The air-gapped platform never downloads at run time: ``src/`` loads weights only from
``registry`` in configs/models.yaml. This script is the one place that fetches them,
verifies the SHA-256 (torchvision names its files after the hash prefix), and pins the
hash in the config on first use.

Usage: uv run python scripts/fetch_weights.py [--name resnet34_imagenet]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path("configs/models.yaml")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="resnet34_imagenet")
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text())
    entry = cfg["pretrained"][a.name]
    dst = Path(cfg["registry"]) / entry["file"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        part = dst.with_name(dst.name + ".part")
        print(f"downloading {entry['url']}")
        urllib.request.urlretrieve(entry["url"], part)
        part.replace(dst)
    digest = _sha256(dst)
    prefix = re.search(r"-([0-9a-f]{8,})\.pth$", entry["url"])
    if prefix and not digest.startswith(prefix.group(1)):
        dst.unlink()
        raise SystemExit(f"hash mismatch: {digest} does not start with {prefix.group(1)}")
    if entry.get("sha256") is None:
        text = CONFIG.read_text()
        old = f"    url: {entry['url']}"
        start = text.index(old)
        line_end = text.index("sha256: null", start)
        text = text[:line_end] + f"sha256: {digest}" + text[line_end + len("sha256: null") :]
        CONFIG.write_text(text)
        print(f"pinned sha256 {digest} in {CONFIG}")
    elif entry["sha256"] != digest:
        raise SystemExit(f"{dst}: sha256 {digest} != pinned {entry['sha256']}")
    print(f"{dst}: {dst.stat().st_size / 2**20:.1f} MiB, sha256 ok")


if __name__ == "__main__":
    main()
