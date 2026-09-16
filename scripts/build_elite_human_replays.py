from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<format>.+?)_(?P<rating>\d+)_.*_(?P<date>\d{2}-\d{2}-\d{4})(?:_\d{2}:\d{2}:\d{2})?_(?:WIN|LOSS)\.json(?:\.lz4)?$"
)


def rating_from_name(path: Path) -> int | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group("rating"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a hard-linked elite Gen 3 OU human replay subset.")
    parser.add_argument("--source", required=True, help="Parsed replay root containing gen3ou/")
    parser.add_argument("--destination", required=True, help="Destination root for custom MetamonDataset")
    parser.add_argument("--min-rating", type=int, default=1400)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--format", default="gen3ou")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    source_format = source / args.format
    destination = Path(args.destination).resolve()
    destination_format = destination / args.format

    if not source_format.is_dir():
        raise SystemExit(f"Source format directory does not exist: {source_format}")

    destination_format.mkdir(parents=True, exist_ok=True)

    scanned = 0
    selected = 0
    skipped_unparseable = 0
    skipped_rating = 0

    for path in source_format.rglob("*"):
        if not path.is_file() or not path.name.endswith((".json", ".json.lz4")):
            continue
        scanned += 1
        rating = rating_from_name(path)
        if rating is None:
            skipped_unparseable += 1
            continue
        if rating < args.min_rating or (args.max_rating is not None and rating > args.max_rating):
            skipped_rating += 1
            continue

        destination_path = destination_format / path.name
        if destination_path.exists():
            selected += 1
            continue

        # Hard links avoid duplicating the large replay payload on the same NTFS volume.
        os.link(path, destination_path)
        selected += 1

    metadata = {
        "format": args.format,
        "min_rating": args.min_rating,
        "max_rating": args.max_rating,
        "source": str(source_format),
        "destination": str(destination_format),
        "scanned_files": scanned,
        "selected_files": selected,
        "skipped_rating": skipped_rating,
        "skipped_unparseable": skipped_unparseable,
        "storage_mode": "hardlink",
    }
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / "subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
