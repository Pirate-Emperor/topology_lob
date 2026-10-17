from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "assets" / "figures" / "a100_fi2010_validation_dashboard.svg"
GALLERY = {
    "readme_daily_ic.svg": (
        "Fixed Focal Loss: daily held-out IC", "Pearson IC", "Rank IC",
        "Target = 0.064", "Day 8", "Day 9", "Day 10",
    ),
    "readme_daily_auc.svg": (
        "Fixed Focal Loss: daily ROC-AUC", "ROC-AUC",
        "Chance reference = 0.50", "Day 8", "Day 9", "Day 10",
    ),
    "readme_ffd_throughput.svg": (
        "Fractional-differentiation throughput", "CPU", "A100 CUDA",
        "Million events per second", "147.8", 
    ),
    "readme_feature_ablation.svg": (
        "Feature ablation", "Microstructure", "Micro + FFD",
        "Micro + topology", "Full stack", "Pearson IC", "Rank IC",
    ),
    "readme_robustness.svg": (
        "Moving-block bootstrap interval", "Pooled Pearson IC",
        "Target = 0.064", "95% CI",
    ),
    "readme_return_regression.svg": (
        "Walk-forward selected return regressor", "Day 8", "Day 9",
        "Day 10", "Pearson IC", "Rank IC",
    ),
    "readme_stationarity.svg": (
        "Median ADF p-value", "Correlation with original price",
        "d = 0.2", "d = 0.3",
    ),
}


def text_from_svg(path: Path) -> tuple[ET.Element, str]:
    root = ET.parse(path).getroot()
    text_nodes = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("text")
    ]
    return root, " ".join(text_nodes)


def test_readme_plot_gallery_contains_seven_valid_data_charts() -> None:
    assert len(GALLERY) == 7
    for filename, labels in GALLERY.items():
        path = ROOT / "assets" / "figures" / filename
        assert path.is_file(), f"missing generated README chart: {filename}"
        assert path.stat().st_size > 1000, f"chart output looks incomplete: {filename}"
        root, joined = text_from_svg(path)
        assert root.tag.endswith("svg"), filename
        assert root.get("width"), filename
        assert root.get("height"), filename
        for label in labels:
            assert label in joined, f"{filename} is missing expected label {label!r}"


def test_a100_dashboard_is_valid_and_keeps_key_metrics_readable() -> None:
    root, joined = text_from_svg(DASHBOARD)
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "1600"
    assert root.attrib["height"] == "1120"
    assert root.attrib["viewBox"] == "0 0 1600 1120"
    assert len([node for node in root.iter() if node.tag.endswith("text")]) >= 35
    assert any(node.tag.endswith("title") and "".join(node.itertext()).strip() for node in root.iter())
    assert any(node.tag.endswith("desc") and "".join(node.itertext()).strip() for node in root.iter())
    for label in ("A100", "0.064", "numba-cuda", "147.8× faster", "0.15194", "0.21495"):
        assert label in joined


def test_readme_local_image_links_resolve_and_gallery_is_embedded() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    markdown_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    html_targets = re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', readme, flags=re.IGNORECASE)
    targets = markdown_targets + html_targets
    assert len(html_targets) >= 7, "README should embed the multi-plot gallery"
    assert all(name in readme for name in GALLERY)
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        clean = target.split("#", 1)[0].split("?", 1)[0]
        assert (ROOT / clean).is_file(), f"missing README image: {target}"


def test_obsolete_main_png_and_single_plot_are_removed() -> None:
    assert not (ROOT / "main.png").exists()
    assert not (ROOT / "assets" / "main.png").exists()
    assert not (ROOT / "assets" / "figures" / "benchmark_summary.svg").exists()
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
