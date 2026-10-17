from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
    "ff14d7c2fd38bdfc143389786993d0f0236d4eb8/data/data.zip"
)
DEFAULT_BYTES = 56_278_154
DEFAULT_GIT_BLOB_SHA1 = "2d8d7749caf622dd07e0df954413dc698129c766"
REQUIRED_FILES = (
    "Train_Dst_NoAuction_DecPre_CF_7.txt",
    "Test_Dst_NoAuction_DecPre_CF_7.txt",
    "Test_Dst_NoAuction_DecPre_CF_8.txt",
    "Test_Dst_NoAuction_DecPre_CF_9.txt",
)


class DownloadError(RuntimeError):
    """The archive could not be downloaded and verified."""


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    """Return (start, end, total) for a valid HTTP Content-Range header."""
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", (value or "").strip())
    if not match:
        raise DownloadError(f"Missing or invalid Content-Range header: {value!r}")
    start, end, total = map(int, match.groups())
    if start > end or end >= total:
        raise DownloadError(f"Invalid Content-Range bounds: {value!r}")
    return start, end, total


def git_blob_sha1(path: Path) -> str:
    """Compute the Git object SHA-1 for a file, including its object header."""
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(
    path: Path,
    expected_bytes: int | None,
    expected_git_blob_sha1: str | None,
) -> dict:
    if not path.is_file():
        raise DownloadError(f"Archive does not exist: {path}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise DownloadError(
            f"Archive is incomplete: got {size:,} bytes; expected {expected_bytes:,}"
        )
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"Downloaded file is not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise DownloadError(f"ZIP CRC check failed for member: {bad_member}")
        names = {Path(name).name for name in archive.namelist() if not name.endswith("/")}
    missing = sorted(set(REQUIRED_FILES) - names)
    if missing:
        raise DownloadError(f"FI-2010 archive is missing expected files: {missing}")
    actual_blob_sha = git_blob_sha1(path)
    if expected_git_blob_sha1 and actual_blob_sha.lower() != expected_git_blob_sha1.lower():
        raise DownloadError(
            "Archive Git blob SHA-1 mismatch: "
            f"{actual_blob_sha}; expected {expected_git_blob_sha1}"
        )
    return {
        "bytes": size,
        "sha256": sha256(path),
        "git_blob_sha1": actual_blob_sha,
        "required_files": list(REQUIRED_FILES),
    }


def _fetch_range(
    url: str,
    start: int,
    end: int,
    total: int,
    timeout: int,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "Topology-LOB research benchmark data fetcher",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status != 206:
            raise DownloadError(
                f"Server did not honor byte range {start}-{end}; HTTP status {status}"
            )
        actual_start, actual_end, actual_total = parse_content_range(
            response.headers.get("Content-Range")
        )
        if (actual_start, actual_end, actual_total) != (start, end, total):
            raise DownloadError(
                "Server returned a different byte range: "
                f"{actual_start}-{actual_end}/{actual_total}; "
                f"expected {start}-{end}/{total}"
            )
        payload = response.read()
    expected_length = end - start + 1
    if len(payload) != expected_length:
        raise DownloadError(
            f"Short byte range at {start}: received {len(payload):,}; "
            f"expected {expected_length:,}"
        )
    return payload


def download_archive(
    url: str,
    output: Path,
    *,
    expected_bytes: int = DEFAULT_BYTES,
    expected_git_blob_sha1: str | None = DEFAULT_GIT_BLOB_SHA1,
    chunk_size: int = 4 * 1024 * 1024,
    retries: int = 6,
    timeout: int = 90,
) -> dict:
    """Download with checked HTTP ranges, retries, resume, and archive verification."""
    if expected_bytes <= 0 or chunk_size <= 0 or retries < 1 or timeout <= 0:
        raise ValueError("expected_bytes, chunk_size, retries and timeout must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            info = validate_archive(output, expected_bytes, expected_git_blob_sha1)
            info["reused_existing_archive"] = True
            return info
        except DownloadError:
            output.unlink()

    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()

    with partial.open("ab") as stream:
        start = stream.tell()
        while start < expected_bytes:
            end = min(start + chunk_size, expected_bytes) - 1
            last_error: Exception | None = None
            payload = b""
            for attempt in range(retries):
                try:
                    payload = _fetch_range(url, start, end, expected_bytes, timeout)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < retries:
                        delay = min(2 ** attempt, 15)
                        print(
                            f"Range {start:,}-{end:,} failed "
                            f"(attempt {attempt + 1}/{retries}): {exc}; "
                            f"retrying in {delay}s",
                            flush=True,
                        )
                        time.sleep(delay)
            if last_error is not None:
                raise DownloadError(
                    f"Could not fetch bytes {start}-{end} after {retries} attempts"
                ) from last_error
            stream.write(payload)
            stream.flush()
            start += len(payload)
            print(f"Downloaded {start:,}/{expected_bytes:,} bytes", flush=True)

    info = validate_archive(partial, expected_bytes, expected_git_blob_sha1)
    partial.replace(output)
    info["reused_existing_archive"] = False
    return info


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract only after validation and reject ZIP path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise DownloadError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(root)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise DownloadError(f"Extraction did not produce the required FI-2010 files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the CC BY 4.0 FI-2010 Decimal-Precision benchmark"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/external/fi2010.zip")
    parser.add_argument("--expected-bytes", type=int, default=DEFAULT_BYTES)
    parser.add_argument(
        "--expected-git-blob-sha1",
        default=DEFAULT_GIT_BLOB_SHA1,
        help="Expected Git blob object ID for the pinned public archive; pass an empty string to skip this check",
    )
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Extraction directory (default: a sibling directory named fi2010)",
    )
    args = parser.parse_args()
    output = Path(args.out)
    info = download_archive(
        args.url,
        output,
        expected_bytes=args.expected_bytes,
        expected_git_blob_sha1=args.expected_git_blob_sha1 or None,
        chunk_size=args.chunk_size,
        retries=args.retries,
        timeout=args.timeout,
    )
    result = {
        "source_url": args.url,
        "archive": str(output),
        "data_license": "CC BY 4.0",
        "citation": (
            "Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., and Iosifidis, A. "
            "(2018), Benchmark Dataset for Mid-Price Forecasting of Limit Order Book "
            "Data with Machine Learning Methods, Journal of Forecasting 37(8), 852-866. "
            "https://doi.org/10.1002/for.2543"
        ),
        **info,
    }
    if args.extract:
        destination = (
            Path(args.extract_dir)
            if args.extract_dir
            else output.parent / "fi2010"
        )
        extract_archive(output, destination)
        result["extract_root"] = str(destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
