from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


USER_AGENT = (
    "netdisk;P2SP;3.0.0.8;netdisk;11.12.3;ANG-AN00;android-android;10.0;"
    "JSbridge4.4.0;jointBridge;1.1.0;"
)
SEGMENT_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class Download:
    name: str
    size: int
    url: str


def read_manifest(path: Path) -> list[Download]:
    downloads = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("\t", 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed manifest line {number}")
        name, size, url = parts
        if Path(name).name != name or not url.startswith("https://"):
            raise ValueError(f"Unsafe manifest line {number}")
        downloads.append(Download(name, int(size), url))
    return downloads


def download_one(item: Download, output: Path) -> str:
    destination = output / item.name
    if destination.is_file() and destination.stat().st_size == item.size:
        return f"SKIP {item.name}"
    if destination.exists() and destination.stat().st_size > item.size:
        raise RuntimeError(f"Existing file is too large: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output / f".{item.name}.segments"
    part_dir.mkdir(exist_ok=True)
    segments: list[tuple[Path, int]] = []
    for index, start in enumerate(range(0, item.size, SEGMENT_SIZE)):
        end = min(item.size - 1, start + SEGMENT_SIZE - 1)
        expected = end - start + 1
        segment = part_dir / f"{index:05d}.part"
        segments.append((segment, expected))
        if segment.is_file() and segment.stat().st_size == expected:
            continue
        if segment.exists():
            segment.unlink()
        command = [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-delay",
            "2",
            "--range",
            f"{start}-{end}",
            "--output",
            str(segment),
            "--user-agent",
            USER_AGENT,
            item.url,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"curl status {error.returncode} for segment {index} ({start}-{end})"
            ) from None
        actual = segment.stat().st_size if segment.exists() else 0
        if actual != expected:
            raise RuntimeError(
                f"Segment size mismatch {item.name}#{index}: expected {expected}, got {actual}"
            )

    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as target:
        for segment, expected in segments:
            if not segment.is_file() or segment.stat().st_size != expected:
                raise RuntimeError(f"Missing completed segment: {segment}")
            with segment.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(destination)
    actual = destination.stat().st_size
    if actual != item.size:
        raise RuntimeError(f"Size mismatch for {item.name}: expected {item.size}, got {actual}")
    for segment, _ in segments:
        segment.unlink()
    part_dir.rmdir()
    return f"DONE {item.name} ({actual / 2**30:.2f} GiB)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download temporary Baidu links without account cookies")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    downloads = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, item, args.output): item for item in downloads
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append((item.name, error))
                print(f"FAILED {item.name}: {error}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; rerun with a fresh manifest to resume")
    print(f"DOWNLOAD_COMPLETE files={len(downloads)}", flush=True)


if __name__ == "__main__":
    main()
