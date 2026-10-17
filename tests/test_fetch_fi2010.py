from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import fetch_fi2010 as fetch


class FakeResponse:
    def __init__(self, payload: bytes, start: int, end: int, total: int):
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        self._payload = payload

    def read(self):
        return self._payload

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def build_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in fetch.REQUIRED_FILES:
            archive.writestr(name, f"fixture for {name}\n")
    return buffer.getvalue()


def test_parse_content_range_valid_and_invalid():
    assert fetch.parse_content_range("bytes 4-9/20") == (4, 9, 20)
    with pytest.raises(fetch.DownloadError):
        fetch.parse_content_range(None)
    with pytest.raises(fetch.DownloadError):
        fetch.parse_content_range("bytes 9-4/20")
    with pytest.raises(fetch.DownloadError):
        fetch.parse_content_range("bytes 0-100/20")


def test_git_blob_sha1_matches_git_object_header(tmp_path):
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"test bytes")
    expected = hashlib.sha1(b"blob 10\0test bytes").hexdigest()
    assert fetch.git_blob_sha1(path) == expected


def test_download_archive_fetches_checked_ranges_and_verifies_zip(tmp_path):
    data = build_archive()
    source_file = tmp_path / "source.zip"
    source_file.write_bytes(data)
    expected_sha = fetch.git_blob_sha1(source_file)
    output = tmp_path / "data.zip"
    chunk_size = 128
    requests_seen = []

    def fake_urlopen(request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
        assert match is not None
        start, end = map(int, match.groups())
        assert timeout == 10
        requests_seen.append((start, end))
        return FakeResponse(data[start : end + 1], start, end, len(data))

    with patch("scripts.fetch_fi2010.urllib.request.urlopen", side_effect=fake_urlopen):
        result = fetch.download_archive(
            "https://unit.test/data.zip",
            output,
            expected_bytes=len(data),
            expected_git_blob_sha1=expected_sha,
            chunk_size=chunk_size,
            retries=1,
            timeout=10,
        )

    assert output.read_bytes() == data
    assert not output.with_suffix(".zip.part").exists()
    assert result["bytes"] == len(data)
    assert result["git_blob_sha1"] == expected_sha
    assert len(requests_seen) == (len(data) + chunk_size - 1) // chunk_size
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None


def test_incomplete_or_wrong_source_archives_are_rejected(tmp_path):
    path = tmp_path / "small.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(fetch.DownloadError, match="incomplete"):
        fetch.validate_archive(path, expected_bytes=10, expected_git_blob_sha1=None)
    with pytest.raises(fetch.DownloadError, match="not a valid ZIP"):
        fetch.validate_archive(path, expected_bytes=len(b"not a zip"), expected_git_blob_sha1=None)


def test_extractor_rejects_zip_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    dest = tmp_path / "extract"
    with pytest.raises(fetch.DownloadError, match="Unsafe ZIP member"):
        fetch.extract_archive(archive_path, dest)
