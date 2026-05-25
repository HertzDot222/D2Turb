# D2Turb Project Page

Static project page for **D2Turb: Depth-Aware Simulation and Decoupled Learning for Single-Frame Atmospheric Turbulence Mitigation**. The page presents the method overview, reported tables, visual comparisons, ablations, and documented limitations as a public companion to an arXiv preprint.

## Publish With GitHub Pages

Target repository: [HertzDot222/D2Turb](https://github.com/HertzDot222/D2Turb).

1. Upload the contents of this folder to `HertzDot222/D2Turb`. Do not upload the local `.tools/` or `.work/` folders.
2. After the arXiv record is available, replace the placeholder URLs in `index.html` with links to the arXiv abstract page and supplementary PDF.
3. In the repository, open **Settings > Pages**, choose **Deploy from a branch**, select `main` and `/(root)`, then save.
4. Visit `https://hertzdot222.github.io/D2Turb/` and confirm that figures, tables, the expanded-image viewer, and the arXiv link load correctly.

The public copy is intentionally publication-facing: keep it focused solely on the paper and its evidence.

## Replace The Resource Buttons

Find the placeholder URLs in `index.html` and replace them with the released preprint links:

```html
<a class="button primary" href="https://arxiv.org/abs/XXXX.XXXXX">arXiv</a>
<a class="button" href="https://arxiv.org/pdf/XXXX.XXXXX">Supplement</a>
```

Update the citation block in the Resources section with the final author list and arXiv identifier at the same time.

## Rebuild Figure Crops

The published PNG assets were cropped from the supplied manuscript PDFs using fixed, header-free regions. To regenerate them locally:

```powershell
python -m pip install PyMuPDF==1.24.14 --target .tools
python scripts/extract_figures.py "<path-to-main-pdf>" "<path-to-supplement-pdf>"
```

For a manual crop audit, render source-page previews into the ignored `.work/` directory:

```powershell
python scripts/extract_figures.py "<path-to-main-pdf>" "<path-to-supplement-pdf>" --preview
```

## Structure

```text
index.html                  Single-page paper website
assets/css/style.css        Responsive visual system
assets/js/main.js           Figure lightbox and navigation state
assets/img/*.png            Cropped paper figures used on the page
scripts/extract_figures.py  Repeatable PDF crop generator
tests/test_site_content.py  Public-page contract checks
```

Run the contract checks before publishing:

```powershell
python -m unittest discover -s tests -v
```
