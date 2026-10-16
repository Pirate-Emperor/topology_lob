from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "assets" / "figures" / "a100_fi2010_validation_dashboard.svg"
SUMMARY_PLOT = ROOT / "assets" / "figures" / "benchmark_summary.svg"


def test_simple_summary_plot_is_valid_and_uses_measured_values() -> None:
    root = ET.parse(SUMMARY_PLOT).getroot()
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "1400"
    assert root.attrib["height"] == "650"
    assert root.attrib["viewBox"] == "0 0 1400 650"

    texts = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("text")
    ]
    joined = " ".join(texts)
    assert any(node.tag.endswith("title") for node in root.iter())
    assert any(node.tag.endswith("desc") for node in root.iter())
    for label in (
        "Pearson IC",
        "Rank IC",
        "Day 8",
        "Day 9",
        "Day 10",
        "Target = 0.064",
        "0.151939",
        "61.34 M/s",
        "0.415 M/s",
        "147.83× measured speedup",
        "2.22 × 10",
    ):
        assert label in joined, f"missing label from benchmark plot: {label}"


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
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
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
    author_section = readme.split("## Citation and license", 1)[1]
    assert "**Author:** Pirate-Emperor" in author_section


def test_readme_reports_measured_focal_target_and_data_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.151939 pooled Pearson IC" in readme
    assert "| Pooled margin Rank IC | 0.154008 |" in readme
    assert "not a measured result for the runs below" not in readme
    assert "These are engineering checks on synthetic data, not market results." in readme
    assert "does not prove trading profitability" in readme


def test_readme_has_reproducible_protocol_and_distinguishes_controls() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Project goals",
        "## Pipeline and design",
        "## Dataset, license, and boundaries",
        "## Experiments and controls",
        "## Validation and test coverage",
        "## Reproduce the FI-2010 experiment",
        "## Code and output map",
        "## Limits and interpretation",
    ):
        assert heading in readme
    assert "different objective" in readme
    assert "test score would turn the holdout into training data" in readme
    assert "The held-out files are scored only after selection and refitting." in readme
