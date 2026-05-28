# CieloStitch

Core stitching engine for CieloStitch.

CieloStitch Core is the reusable Python package behind the CieloStitch desktop app. It provides the stitching pipeline, subject-aware defaults, image I/O, session parameter resolution, and the command-line entry point for building mosaics from overlapping panels.

It is a panel stitcher built around translation, affine, homography, and APAP based local warping, rather than a WCS-aware reprojection or plate-solving engine. So it works best on overlapping panels with moderate field-of-view change rather than very wide-field or all-sky projection workflows.

It is designed for astrophotography and landscape workflows such as solar, solar H-alpha, lunar, milky way, nightscape, landscape, and general multi-panel mosaics.

## Features

- Subject-aware profiles for `solar`, `solar-h-alpha`, `lunar`, `landscape`, `nightscape`, `milky-way`, `panorama`, and `general`.
- Multiple stitching modes: `auto`, `freeform`, `grid-guided`, `fixed-overlap`, and `zero-overlap`.
- Resolver-driven parameter selection based on panel size, overlap, bit depth, subject type, mount precision, and speed preset.
- Import and export support for TIFF, PNG, JPEG, BMP, WebP, FITS, and XISF.
- Bit-depth-aware processing for integer and floating-point image data.
- Blend modes including multiband, adaptive multiband, feather, adaptive feather, seamless, and none.
- Gain compensation modes including none, simple, uniform, and local.
- Grid and weak-texture fallback logic for subjects where feature matching is less reliable.
- Reusable library modules for CLI workflows, desktop integration, and automated tests.

## Installation

### Requirements

- Python 3.11 or newer
- A virtual environment is strongly recommended

### Install From A Local Checkout

```bash
pip install -e .
pip install PySide6
```

For development:

```bash
pip install -e ".[dev]"
pip install PySide6
```

`PySide6` is currently required at runtime because the shared state and preferences layer uses Qt types. At the moment it is not declared in `pyproject.toml`, so install it explicitly.

Once installed, the CLI entry point is available as:

```bash
cielostitch-cli
```

## Quick Start

Stitch a folder of panels with automatic parameter resolution:

```bash
cielostitch-cli ./panels \
  --profile lunar \
  --stitch-mode auto \
  --resolve \
  --output ./output/lunar_mosaic.tif
```

Useful flags:

- `--profile` selects the subject profile.
- `--stitch-mode` chooses the stitching strategy.
- `--cols`, `--overlap-x-pct`, and `--overlap-y-pct` are useful for grid-based sessions.
- `--blend-type` and `--gain-compensation` override profile defaults.
- `--load-settings` loads a saved settings JSON snapshot.
- `--alpha-policy` controls how alpha channels are handled.

Supported input extensions:

`*.tif`, `*.tiff`, `*.png`, `*.jpg`, `*.jpeg`, `*.bmp`, `*.webp`, `*.fit`, `*.fits`, `*.xisf`

## Examples

### CLI: Grid-Guided Mosaic

```bash
cielostitch-cli ./solar_panels \
  --profile solar \
  --stitch-mode grid-guided \
  --cols 4 \
  --overlap-x-pct 20 \
  --overlap-y-pct 20 \
  --output ./output/solar_grid.tif
```

### CLI: Reuse A Saved Settings Snapshot

```bash
cielostitch-cli ./panels \
  --load-settings ./session.json \
  --stitch-mode freeform \
  --output ./output/result.tif
```

### Python: Run The Freeform Engine Directly

```python
from pathlib import Path

from cielostitch_core.stitching.free_mode import FreeMode
from cielostitch_core.state.profiles import GeneralProfile
from cielostitch_core.utils.image_io import load_panel_images_from_paths, save_image
from cielostitch_core.utils.image_manipulation import from_internal_float32

panel_paths = sorted(str(path) for path in Path("panels").glob("*.tif"))
items, bit_depth, _ = load_panel_images_from_paths(panel_paths)

engine = FreeMode(GeneralProfile())
mosaic = engine.stitch(items)

save_image(from_internal_float32(mosaic, bit_depth), "output.tif")
```

### Python: Inspect Resolver Output Before Stitching

```python
from cielostitch_core.core.session_resolver import ResolverInputs, compute_resolved_params

params = compute_resolved_params(
    ResolverInputs(
        subject="lunar",
        panel_w=4000,
        panel_h=3000,
        stitch_mode="grid-guided",
        mount_precision="tight",
        bit_depth=16,
        overlap_x_pct=25.0,
        overlap_y_pct=25.0,
        speed_preset="balanced",
    )
)

print(params)
```

## Project Structure

```text
cielostitch-core/
├── pyproject.toml
├── README.md
├── src/
│   └── cielostitch_core/
│       ├── cli.py
│       ├── config/
│       ├── core/
│       │   ├── detector.py
│       │   ├── matcher.py
│       │   ├── mosaic_canvas.py
│       │   ├── session_resolver.py
│       │   ├── warper.py
│       │   └── stitching/
│       ├── state/
│       │   ├── preferences.py
│       │   ├── profile_model.py
│       │   ├── session_model.py
│       │   └── profiles/
│       └── utils/
│           ├── image_io.py
│           ├── image_manipulation.py
│           ├── image_meta.py
│           ├── image_resampling.py
│           └── sorting.py
└── tests/
```

High-level responsibilities:

- `cli.py`: command-line interface and argument resolution.
- `core/`: feature detection, matching, warping, blending, canvas handling, and run orchestration.
- `core/stitching/`: stitching engines for freeform and grid-oriented workflows.
- `state/`: profiles, session state, preferences, and runtime configuration.
- `utils/`: image loading, saving, metadata handling, dtype conversions, and helper utilities.
- `tests/`: regression coverage for resolver logic, I/O behavior, metadata transfer, overlap handling, and session settings.

## Roadmap

- Improve packaging metadata so all runtime dependencies are declared directly in `pyproject.toml`.
- Publish clearer API documentation for direct library use outside the desktop app.
- Add more end-to-end examples for common astrophotography capture patterns.
- Expand benchmarks and regression fixtures for large mosaics and weak-texture panels.
- Continue improving deterministic placement, diagnostics, and export workflows.

## License

`cielostitch-core` is released under the MIT License. See the `LICENSE` file for details.
