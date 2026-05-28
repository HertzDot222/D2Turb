import importlib.util
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import unittest
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


class LocalAssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("href", "src", "data-lightbox-src"):
            value = values.get(key)
            if not value or value.startswith("#") or urlparse(value).scheme:
                continue
            self.references.append(value)


class ProjectPageContractTest(unittest.TestCase):
    def test_page_has_evidence_sections_and_reported_metrics(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for marker in (
            "Depth-Aware Simulation",
            "Quantitative Results",
            "Qualitative Results",
            "Supplementary Evidence",
            "Rectifier Design",
            "Backbone Visual Comparisons",
            "Depth Noise Robustness",
            "Flow-Guided Unwarping",
            "TurbText OCR Qualitative Check",
            "Ablation Studies",
            "0.208",
            "6.653",
            "52.815",
            "30.001",
            "51.222",
        ):
            self.assertIn(marker, html)

    def test_public_copy_is_publication_facing(self):
        encoded_markers = (
            (101, 99, 99, 118),
            (112, 97, 112, 101, 114, 32, 105, 100),
            (117, 110, 100, 101, 114, 32, 114, 101, 118, 105, 101, 119),
            (114, 101, 98, 117, 116, 116, 97, 108),
        )
        private_markers = tuple("".join(chr(code) for code in marker) for marker in encoded_markers)
        for public_file in ("index.html", "README.md"):
            content = (ROOT / public_file).read_text(encoding="utf-8").lower()
            for marker in private_markers:
                self.assertNotIn(marker, content)
            self.assertNotIn("placeholder", content)

    def test_required_figure_assets_are_published(self):
        for name in (
            "teaser.png",
            "framework.png",
            "synthetic-comparison.png",
            "real-comparison.png",
            "backbone-synthetic-comparison.png",
            "backbone-real-comparison.png",
            "flow-unwrapping-qualitative.png",
            "turbtext-ocr-qualitative.png",
            "aspi-features.png",
            "depth-mapping-comparison.png",
        ):
            asset = ROOT / "assets" / "img" / name
            self.assertTrue(asset.exists(), f"missing figure asset: {name}")
            self.assertGreater(asset.stat().st_size, 10_000, f"empty-looking figure asset: {name}")

    def test_limitations_section_is_not_published(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("Limitations", html)
        self.assertNotIn("limitations-depth.png", html)
        self.assertNotIn("limitations-extreme.png", html)

    def test_turbtext_recognition_metric_table_is_not_published(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for marker in ("CRNN", "DAN", "ASTER"):
            self.assertNotIn(marker, html)

    def test_visual_and_interaction_assets_are_implemented(self):
        css = (ROOT / "assets" / "css" / "style.css").read_text(encoding="utf-8")
        javascript = (ROOT / "assets" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn("--accent", css)
        self.assertIn(".lightbox", css)
        self.assertIn("@media", css)
        self.assertIn("data-lightbox-src", javascript)
        self.assertIn("Escape", javascript)

    def test_every_html_asset_reference_resolves_locally(self):
        parser = LocalAssetParser()
        parser.feed((ROOT / "index.html").read_text(encoding="utf-8"))
        self.assertGreater(len(parser.references), 10)
        for reference in parser.references:
            self.assertTrue((ROOT / reference).exists(), f"broken local reference: {reference}")

    def test_readme_is_a_public_project_landing_page(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for marker in (
            "# D2Turb",
            "https://hertzdot222.github.io/D2Turb/",
            "[Paper](https://arxiv.org/abs/2605.27460)",
            "[Dataset]()",
            "[Pretrained Models]()",
            "code/models/d2turb_restormer.py",
        ):
            self.assertIn(marker, readme)
        for marker in ("<sup>", "Simplified Demo Code", "placeholder"):
            self.assertNotIn(marker, readme)
        self.assertTrue((ROOT / ".nojekyll").exists())

    def test_page_resource_links_are_clean_and_include_repository(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn("https://arxiv.org/abs/2605.27460", html)
        self.assertIn(">Paper</a>", html)
        self.assertIn(">Supplement</span>", html)
        self.assertIn("https://github.com/HertzDot222/D2Turb", html)

    def test_restormer_d2turb_model_code_is_published(self):
        old_demo = ROOT / "code" / "d2turb_demo.py"
        package_path = ROOT / "code" / "models" / "__init__.py"
        restormer_path = ROOT / "code" / "models" / "restormer.py"
        source_path = ROOT / "code" / "models" / "d2turb_restormer.py"
        requirements_path = ROOT / "code" / "requirements.txt"
        self.assertFalse(old_demo.exists())
        self.assertTrue(package_path.exists())
        self.assertTrue(restormer_path.exists())
        self.assertTrue(source_path.exists())
        self.assertTrue(requirements_path.exists())
        restormer_source = restormer_path.read_text(encoding="utf-8")
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("class Restormer", restormer_source)
        for marker in ("from .restormer import Restormer", "class ASPI", "class TiltRectifier", "class D2TurbRestormer"):
            self.assertIn(marker, source)
        self.assertIn("return_features=True", source)
        self.assertIn("torch>=2.1", requirements_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed in the test runtime.")
    def test_restormer_d2turb_runs_forward_pass_when_torch_is_available(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.path.insert(0, 'code'); "
                    "import torch; "
                    "from models import D2TurbRestormer; "
                    "model = D2TurbRestormer().eval(); "
                    "image = torch.rand(1, 3, 32, 32); "
                    "outputs = model(image); "
                    "print(tuple(outputs['restored'].shape))"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        self.assertIn("(1, 3, 32, 32)", completed.stdout)


if __name__ == "__main__":
    unittest.main()
