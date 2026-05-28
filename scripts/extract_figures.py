"""Render publication-ready project-page figures from the supplied PDFs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOOLS = ROOT / ".tools"
if LOCAL_TOOLS.exists():
    sys.path.insert(0, str(LOCAL_TOOLS))

import pymupdf


PREVIEW_PAGES = {
    "main": (2, 6, 9, 11, 12, 14),
    "supp": (8, 14, 15),
}

# Coordinates are normalized fractions of the source page rectangle:
# left, top, right, bottom. Each crop begins below the manuscript header.
CROPS = (
    {"filename": "teaser.png", "source": "main", "page": 2, "box": (0.08, 0.055, 0.92, 0.50)},
    {"filename": "framework.png", "source": "main", "page": 6, "box": (0.08, 0.055, 0.92, 0.545)},
    {"filename": "synthetic-comparison.png", "source": "main", "page": 11, "box": (0.08, 0.055, 0.92, 0.66)},
    {"filename": "real-comparison.png", "source": "main", "page": 12, "box": (0.08, 0.055, 0.92, 0.60)},
    {"filename": "aspi-features.png", "source": "main", "page": 14, "box": (0.08, 0.055, 0.92, 0.48)},
    {"filename": "backbone-synthetic-comparison.png", "source": "supp", "page": 10, "box": (0.075, 0.055, 0.925, 0.84)},
    {"filename": "backbone-real-comparison.png", "source": "supp", "page": 11, "box": (0.075, 0.055, 0.925, 0.90)},
    {"filename": "depth-mapping-comparison.png", "source": "supp", "page": 8, "box": (0.08, 0.292, 0.92, 0.95)},
)

EXTRA_CROPS = (
    {"filename": "turbtext-ocr-qualitative.png", "source": "extra", "page": 1, "box": (0.095, 0.808, 0.482, 0.884), "zoom": 4.0},
    {"filename": "flow-unwrapping-qualitative.png", "source": "extra", "page": 1, "box": (0.517, 0.718, 0.905, 0.884), "zoom": 4.0},
)


def render_clip(pdf_path: Path, page_number: int, destination: Path, box=None, zoom: float = 2.8) -> None:
    document = pymupdf.open(pdf_path)
    page = document[page_number - 1]
    clip = None
    if box:
        left, top, right, bottom = box
        bounds = page.rect
        clip = pymupdf.Rect(
            bounds.x0 + bounds.width * left,
            bounds.y0 + bounds.height * top,
            bounds.x0 + bounds.width * right,
            bounds.y0 + bounds.height * bottom,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    pixmap.save(destination)
    document.close()


def render_previews(main_pdf: Path, supplement_pdf: Path) -> None:
    sources = {"main": main_pdf, "supp": supplement_pdf}
    preview_dir = ROOT / ".work" / "previews"
    for source, pages in PREVIEW_PAGES.items():
        for page_number in pages:
            output = preview_dir / f"{source}-p{page_number:02}.png"
            render_clip(sources[source], page_number, output, zoom=1.25)
            print(f"preview: {output}")


def extract_figures(main_pdf: Path, supplement_pdf: Path, extra_pdf: Path | None = None) -> None:
    sources = {"main": main_pdf, "supp": supplement_pdf}
    crop_items = list(CROPS)
    if extra_pdf is not None:
        sources["extra"] = extra_pdf
        crop_items.extend(EXTRA_CROPS)
    image_dir = ROOT / "assets" / "img"
    for item in crop_items:
        output = image_dir / item["filename"]
        render_clip(sources[item["source"]], item["page"], output, item["box"], zoom=item.get("zoom", 2.8))
        print(f"figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_pdf", type=Path)
    parser.add_argument("supplement_pdf", type=Path)
    parser.add_argument("--extra-pdf", type=Path, default=None, help="Optional extra material PDF for additional crops.")
    parser.add_argument("--preview", action="store_true", help="Render full reference pages for crop selection.")
    args = parser.parse_args()

    if args.preview:
        render_previews(args.main_pdf, args.supplement_pdf)
        return
    extract_figures(args.main_pdf, args.supplement_pdf, args.extra_pdf)


if __name__ == "__main__":
    main()
