from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "assets" / "figures" / "a100_fi2010_validation_dashboard.svg"


def test_a100_dashboard_is_valid_and_keeps_key_metrics_readable() -> None:
    root = ET.parse(DASHBOARD).getroot()
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "1600"
    assert root.attrib["height"] == "1120"
    assert root.attrib["viewBox"] == "0 0 1600 1120"

    texts = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("text")
    ]
    joined = " ".join(texts)
    assert len(texts) >= 35
    assert any(node.tag.endswith("title") for node in root.iter())
    assert any(node.tag.endswith("desc") for node in root.iter())
    for label in ("A100", "0.064", "numba-cuda", "147.8× faster", "0.15194", "0.21495"):
        assert label in joined


def test_readme_local_image_links_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"!\\[[^\\]]*\\]\\(([^)]+)\\)", readme)
    assert targets, "README should show its analysis images"
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        clean = target.split("#", 1)[0].split("?", 1)[0]
        assert (ROOT / clean).is_file(), f"missing README image: {target}"


def test_obsolete_main_png_and_duplicate_license_files_are_absent() -> None:
    assert not (ROOT / "main.png").exists()
    assert not (ROOT / "assets" / "main.png").exists()
    license_like = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name.lower() in {"license", "copying"}
        and ".git" not in path.parts
    ]
    assert license_like == [ROOT / "LICENSE"]


def test_project_author_is_pirate_emperor() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'authors = [{ name = "Pirate-Emperor" }]' in pyproject
    author_section = readme.split("## Author", 1)[1].split("## ", 1)[0]
    assert "**Pirate-Emperor**" in author_section
