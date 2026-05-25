# D<sup>2</sup>Turb: Depth-Aware Simulation and Decoupled Learning

**Single-Frame Atmospheric Turbulence Mitigation**

[Project Page](https://hertzdot222.github.io/D2Turb/) |
[Paper (placeholder)](https://arxiv.org/abs/XXXX.XXXXX) |
[Dataset (placeholder)](https://example.com/dataset-placeholder) |
[Pretrained Models (placeholder)](https://example.com/model-placeholder)

D<sup>2</sup>Turb addresses single-frame atmospheric turbulence by separating
texture recovery from geometric rectification. A depth-aware simulation model
provides spatially varying turbulence and intermediate tilt supervision, while
Adaptive Structural Prior Injection (ASPI) transfers sharp structural cues into
the geometric rectifier.

![D2Turb motivation and qualitative result](assets/img/teaser.png)

## Highlights

- **Depth-aware simulation:** models spatially varying blur and deformation using scene-depth cues.
- **Decoupled restoration:** separates deblurring from non-rigid geometric correction.
- **Structural guidance:** ASPI injects texture-stage priors into the rectification stage.
- **Reported real-world performance:** `6.653` NIQE and `52.815` MUSIQ on RLR-AT.

## Reported Results

| Evaluation | Metric | D<sup>2</sup>Turb |
| --- | ---: | ---: |
| Synthetic average | PSNR / SSIM / LPIPS | 25.724 / 0.736 / **0.208** |
| Real-world RLR-AT | NIQE / MUSIQ | **6.653** / **52.815** |
| Depth-aware simulation study | NIQE / MUSIQ | 6.980 / 51.996 |

More quantitative comparisons, qualitative figures, ablations, and documented
limitations are available on the [project page](https://hertzdot222.github.io/D2Turb/).

## Resources

| Resource | Link | Status |
| --- | --- | --- |
| Project Page | [Website](https://hertzdot222.github.io/D2Turb/) | Available |
| Paper | [arXiv](https://arxiv.org/abs/XXXX.XXXXX) | Placeholder |
| Supplement | [PDF](https://arxiv.org/pdf/XXXX.XXXXX) | Placeholder |
| Dataset | [Download](https://example.com/dataset-placeholder) | Placeholder |
| Pretrained Models | [Download](https://example.com/model-placeholder) | Placeholder |

Placeholder resource links will be replaced when the corresponding public
artifacts are released.

## Simplified Demo Code

[`code/d2turb_demo.py`](code/d2turb_demo.py) is a compact PyTorch architecture
demonstration for understanding the two-stage workflow:

```text
Turbulent Input -> TextureDeblur -> Deblurred Image + Guide Features
                                     |
                                     v
                              ASPI + TiltRectifier -> Restored Image
```

The demo is intentionally simplified for readability. It is not the complete
training implementation and does not contain pretrained weights or benchmark
evaluation code.

```bash
pip install -r code/requirements.txt
python code/d2turb_demo.py
```

Expected output shapes:

```text
input:     (1, 3, 128, 128)
deblurred: (1, 3, 128, 128)
restored:  (1, 3, 128, 128)
```

## Citation

```bibtex
@article{d2turb,
  title = {D2Turb: Depth-Aware Simulation and Decoupled Learning for
           Single-Frame Atmospheric Turbulence Mitigation},
  note = {arXiv preprint, citation to be updated}
}
```
