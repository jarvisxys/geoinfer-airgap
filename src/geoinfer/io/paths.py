"""Write guards: nothing is ever written under a read-only data root."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def assert_writable(path: str | Path, read_only_roots: Iterable[Path]) -> None:
    """Raise if ``path`` resolves to, or inside, any read-only root.

    Symlinks are resolved first, so a link pointing into a read-only root is caught.

    Parameters
    ----------
    path
        Destination about to be written (need not exist yet).
    read_only_roots
        Protected directories, e.g. ``/mnt/d/geo/raw``.

    Raises
    ------
    PermissionError
        If ``path`` is inside a read-only root.
    """
    target = Path(path).expanduser().resolve()
    for root in read_only_roots:
        r = Path(root).expanduser().resolve()
        if target == r or r in target.parents:
            raise PermissionError(f"refusing to write {target}: inside read-only root {r}")
