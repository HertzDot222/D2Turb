# D²Turb

**Depth-Aware Simulation and Decoupled Learning for Single-Frame Atmospheric Turbulence Mitigation**

[Project Page](https://hertzdot222.github.io/D2Turb/) |
[Paper](https://arxiv.org/abs/2605.27460) |
[Dataset]() |
[Pretrained Models]() |
[Code](code/models/d2turb_restormer.py)

D²Turb is a single-frame atmospheric turbulence mitigation framework that
separates two coupled restoration problems: texture degradation and non-rigid
geometric distortion. It uses depth-aware simulation to synthesize spatially
varying turbulence and to provide intermediate tilt supervision, then applies a
decoupled restoration pipeline with Adaptive Structural Prior Injection (ASPI)
to guide geometric rectification from texture-restoration features.

The [project page](https://hertzdot222.github.io/D2Turb/) is the best entry
point for visual comparisons, expanded benchmark tables, and mechanism studies.

![D²Turb motivation and qualitative result](assets/img/teaser.png)

## Highlights

- **Depth-aware simulation:** models spatially varying blur and deformation using scene-depth cues.
- **Decoupled restoration:** separates texture recovery from geometric rectification.
- **ASPI guidance:** injects Restormer structural features into the rectifier for sharper geometry.
- **Restormer-D²Turb code path:** provides a compact PyTorch model definition under [`code/models/`](code/models/).
- **Expanded evidence:** includes synthetic, RLR-AT, TMT-Static, TurbText, depth-noise, flow-unwrapping, and cross-backbone studies.

## Results Snapshot

| Setting | Metric | Reported D²Turb result |
| --- | --- | ---: |
| Synthetic average | PSNR / SSIM / LPIPS | 25.724 / 0.736 / **0.208** |
| Real-world RLR-AT | NIQE / MUSIQ | **6.653** / **52.815** |
| Depth-aware simulation study | NIQE / MUSIQ | 6.980 / 51.996 |
| TMT-Static | PSNR / SSIM / LPIPS | 24.321 / 0.862 / 0.223 |
| TurbText | AWDR / AD-LCS | 0.782 / 8.708 |

LPIPS and NIQE are lower-better metrics. PSNR, SSIM, MUSIQ, AWDR, and AD-LCS
are higher-better metrics. Full comparisons and visual examples are available
on the [project page](https://hertzdot222.github.io/D2Turb/).

## Repository Structure

```text
.
|-- assets/img/                  # Figures used by the project page
|-- code/
|   |-- requirements.txt         # Minimal PyTorch dependencies
|   `-- models/
|       |-- restormer.py         # Restormer backbone
|       `-- d2turb_restormer.py  # ASPI + rectifier + D²Turb wrapper
|-- index.html                   # GitHub Pages project page
|-- tests/                       # Content and smoke tests
`-- README.md
```

## Quick Start

Install the minimal dependencies:

```bash
pip install -r code/requirements.txt
```

Run a forward-pass smoke test:

```bash
python -c "import sys; sys.path.insert(0, 'code'); import torch; from models import D2TurbRestormer; model = D2TurbRestormer().eval(); image = torch.rand(1, 3, 64, 64); outputs = model(image); print(outputs['restored'].shape)"
```

The public code currently focuses on the Restormer-based D²Turb inference
structure. Dataset links and pretrained model links are reserved above for the
release artifacts.

## Resources

| Resource | Link |
| --- | --- |
| Project page | [https://hertzdot222.github.io/D2Turb/](https://hertzdot222.github.io/D2Turb/) |
| Paper | [https://arxiv.org/abs/2605.27460](https://arxiv.org/abs/2605.27460) |
| Dataset | [Dataset]() |
| Pretrained models | [Pretrained Models]() |
| Model definition | [`code/models/d2turb_restormer.py`](code/models/d2turb_restormer.py) |

## Citation

```bibtex
@article{d2turb,
  title = {D²Turb: Depth-Aware Simulation and Decoupled Learning for
           Single-Frame Atmospheric Turbulence Mitigation},
  journal = {arXiv preprint arXiv:2605.27460},
  year = {2026}
}
```
