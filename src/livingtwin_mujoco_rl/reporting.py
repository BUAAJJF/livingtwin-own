from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"FILE_MANIFEST.json", "FILE_MANIFEST.sha256"}:
            entries.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {"schema_version": "file_manifest_v1", "files": entries}
    manifest_path = output / "FILE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256(manifest_path)
    (output / "FILE_MANIFEST.sha256").write_text(
        f"{digest}  FILE_MANIFEST.json\n", encoding="utf-8"
    )
    return manifest


def last_csv_row(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[-1]

