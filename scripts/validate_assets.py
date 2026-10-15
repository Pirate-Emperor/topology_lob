from __future__ import annotations

from pathlib import Path

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

    for path in (ROOT / "main.png", ASSETS / "main.png"):
        if path.exists():
            failures.append(f"remove obsolete image: {path.relative_to(ROOT)}")

    if failures:
        raise SystemExit("Asset validation failed:\n" + "\n".join(failures))
    print("Asset validation: PASS")


if __name__ == "__main__":
    main()
