from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
EXPECTED = {
    "figures/experimental_validation_dashboard.png": None,
    "figures/robustness_dashboard.png": None,
    "figures/data_diagnostics_dashboard.png": None,
    "figures/fi2010_validation_dashboard.png": None,
    "screenshots/report_preview.png": (2000, 1250),
}

SVG_EXPECTED = {
    "figures/a100_fi2010_validation_dashboard.svg": {
        "width": "1600",
        "height": "1120",
        "viewBox": "0 0 1600 1120",
        "required_text": (
            "A100",
            "0.064",
            "numba-cuda",
            "147.8× faster",
            "0.15194",
            "0.21495",
            "does not prove profitability",
        ),
    },
}


def check_svg(path: Path, expected: dict) -> dict:
    """Check that an SVG report is valid XML, has the expected canvas, and keeps its key labels."""
    root = ET.parse(path).getroot()
    if not root.tag.endswith("svg"):
        raise ValueError(f"Expected an SVG root element, got {root.tag!r}")
    width = root.get("width")
    height = root.get("height")
    view_box = root.get("viewBox")
    text_nodes = [
        "".join(node.itertext()).strip()
        for node in root.iter()
        if node.tag.endswith("text")
    ]
    report_text = " ".join(text_nodes)
    title_or_desc = any(
        node.tag.endswith("title") and "".join(node.itertext()).strip()
        for node in root.iter()
    ) and any(
        node.tag.endswith("desc") and "".join(node.itertext()).strip()
        for node in root.iter()
    )
    required_ok = all(label in report_text for label in expected["required_text"])
    return {
        "width": width,
        "height": height,
        "viewBox": view_box,
        "text_nodes": len(text_nodes),
        "title_and_description": title_or_desc,
        "required_labels_present": required_ok,
        "size_ok": (
            width == expected["width"]
            and height == expected["height"]
            and view_box == expected["viewBox"]
        ),
    }


def check(path: Path, expected_size: tuple[int, int] | None) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = image.size
        mode = image.mode
        extrema = image.convert("RGB").getextrema()
        ranges = [hi - lo for lo, hi in extrema]
    return {
        "size": size,
        "mode": mode,
        "channel_ranges": ranges,
        "nontrivial": max(ranges) > 20,
        "size_ok": expected_size is None or size == expected_size,
    }


def main() -> None:
    failures = []
    for relative, expected in EXPECTED.items():
        path = ASSETS / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        result = check(path, expected)
        print(relative, result)
        if not result["nontrivial"] or not result["size_ok"]:
            failures.append(f"bad image: {relative}: {result}")

    for relative, expected in SVG_EXPECTED.items():
        path = ASSETS / relative
        if not path.is_file():
            failures.append(f"missing: assets/{relative}")
            continue
        try:
            result = check_svg(path, expected)
        except (ET.ParseError, OSError, ValueError) as exc:
            failures.append(f"bad SVG: assets/{relative}: {exc}")
            continue
        print(f"assets/{relative}", result)
        if not result["size_ok"] or not result["title_and_description"] or not result["required_labels_present"] or result["text_nodes"] < 35:
            failures.append(f"bad SVG dashboard: assets/{relative}: {result}")

    for path in (ROOT / "main.png", ASSETS / "main.png"):
        if path.exists():
            failures.append(f"remove obsolete image: {path.relative_to(ROOT)}")

    if failures:
        raise SystemExit("Asset validation failed:\n" + "\n".join(failures))
    print("Asset validation: PASS")


if __name__ == "__main__":
    main()
