# satsearch

Desktop tools for **Lucy LORRI** (and similar) FITS data: a scrollable thumbnail browser, a **Stack** view that runs the Python port of the `fullhill.pro` pipeline, and a **Stars** analysis window (refcat overlays, SPICE geometry, optional fake-satellite model, interactive alignment). The GUI uses **Tkinter**; science code uses **Astropy**, **NumPy/SciPy**, **Photutils**, **Pillow**, and **SpiceyPy** where SPICE is enabled.

This repository is intended to be cloned and run on a **local machine** with your own FITS trees, SPICE kernels, and (for Stars) an **atlas-refcat** installation.

---

## Requirements

### Python

- **Python 3.11+** is recommended (stdlib `tomllib` loads `satsearch.toml`).
- **Python 3.10** works if you install a TOML parser: `pip install tomli` (the config loader falls back to `tomli` when `tomllib` is unavailable).

### Python packages

Install from the repo root:

```bash
pip install -r requirements.txt
```

Declared dependencies:

| Package   | Role |
|-----------|------|
| astropy   | FITS I/O, WCS, coordinates |
| numpy     | Arrays |
| scipy     | Shifts, filters (`fullhill`, `lucy_getpsf`) |
| photutils | Source detection (PSF construction) |
| Pillow    | Thumbnails and on-screen images |
| spiceypy  | SPICE ephemeris / geometry (optional at import time; required for SPICE-backed features) |

### System / GUI

- A working **Tk** (Tkinter). On many Linux distributions the package is `python3-tk` (name varies by distro). Windows/macOS Python installers usually include Tk.
- An **X server** (or equivalent) if you run the GUI over SSH.
- **Font files** (optional): magnitude labels in Stars try DejaVu or Liberation Sans under `/usr/share/fonts/...`; if missing, a default font is used.

### Data and external programs (site-specific)

1. **`satsearch.toml`** — Required for the intended configuration path (see [Configuration](#configuration)). It points to **SPICE meta-kernels**, **refcat** paths, and the **default FITS directory**.

2. **CSPICE / SPICE kernels** — Needed for predicted target position, range/phase, and related geometry (`lucy_spice`, `fullhill`). Kernel files are not shipped in this repo; you must obtain NAIF-compatible kernels and list their **meta-kernel** (`.tm`) in config.

3. **atlas-refcat** — Stars mode runs a **`refcat`** executable against a local catalog directory. Paths are set in `satsearch.toml` or via environment variables (see below).

4. **FITS content** — Features assume Lucy-style headers where applicable (e.g. `MIDUTCJD` for SPICE time, WCS for astrometry, `EXPTIME` for exposure). Unsupported or minimal headers will limit or break some tools; fixups are data-specific.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-fork-or-upstream-url> satsearch
cd satsearch
```

### 2. Virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows cmd
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Python 3.10 only:
# pip install tomli
```

### 4. Configuration file

Copy the example and edit paths for **your** machine:

```bash
cp satsearch.toml.example satsearch.toml
```

Edit **`[spice].meta_kernels`**: list one or more NAIF meta-kernel paths (order matters; include leapseconds as required by your kernel set). Set **`target_body`**, **`observer_body`**, **`sun_body`**, **`frame`**, and **`abcorr`** to match the names in your kernels (see kernel documentation for NAIF IDs).

Edit **`[paths]`**:

- **`refcat_exe`** — Path to the `refcat` binary.
- **`refcat_dir`** — Path to the refcat data root (e.g. magnitude-sliced tiles).
- **`default_fits_directory`** — Initial folder when opening the browser (you can change it in the UI).

Alternatively, point to a TOML file elsewhere:

```bash
export SATSEARCH_CONFIG=/path/to/my-satsearch.toml
```

Search order is: `SATSEARCH_CONFIG` (if set), then `satsearch.toml` next to `satsearch_config.py`, then `satsearch.toml` in the current working directory.

### 5. Refcat without `satsearch.toml` (fallback)

If `satsearch.toml` cannot be loaded, Stars falls back to **`SATSEARCH_REFCAT_EXE`** and **`SATSEARCH_REFCAT_DIR`**, or to compile-time defaults inside `stars_analysis.py` (which may not exist on your system). For a normal setup, **use `satsearch.toml`** and keep refcat paths there.

---

## Usage

Run the application from the **repository root** (or ensure the repo is on `PYTHONPATH`) so imports resolve:

```bash
python fits_thumb_viewer.py
```

### Command-line options

| Option | Meaning |
|--------|---------|
| `--minexp SEC` | Default **minimum EXPTIME** (seconds) for the Min EXPTIME filter in the main window (same as the UI default). |
| `--stars FITS [FITS ...]` | Skip the thumbnail browser and open **only** the **Stars** window for the given FITS paths (one group). The root window is minimized; closing Stars exits. Example: `python fits_thumb_viewer.py --stars /data/a.fit /data/b.fit` |

### Main window (thumbnail browser)

1. Set **Directory** to a folder containing `.fit` / `.fits` files (or use **Browse…**).
2. **Load** scans the tree (subject to **Filter out** / **Only include** glob patterns and **Min/Max EXPTIME**, **1×1 only**, **Groups** mode).
3. Select files or groups with checkboxes.
4. **Stack** — Runs the full-hill prepare/stack pipeline on the selection (requires a valid short-mate layout in the sorted file list as documented in `fullhill.py`).
5. **Stars** — Opens the Stars analysis window for the selection.
6. **Define PSF** / **Clear PSF** — Select exactly one FITS to build a session PSF for Stars (fake-satellite and related UI stay disabled until a PSF is defined, per current behavior).

The status line under the buttons shows **Stars PSF** state. Right-click on Stack or Stars images posts **RMS** and **`r_lim`** context to that line when the main browser is open.

### Stars window

- Parallel per-plane cache, plane slider, refcat stars (r band), SPICE geometry when headers/kernels allow.
- **Define center** — Per-plane target position for astrometry alignment.
- **Align** — Two-click refinement on a refcat star.
- **Refresh** — Recompute with current parameters.
- Session **PSF** from the main window affects fake-satellite modeling when enabled.

### Stack window

- Chooses stack type (`imb`, `imbs`, `imz`, `imzs`, medians), plane slider, **Run stack** after changing satellite parameters (diameter, albedo, sat distance/angle).

### Tests

```bash
python -m unittest discover -s tests -v
```

---

## Project layout (high level)

| File / area | Purpose |
|-------------|---------|
| `fits_thumb_viewer.py` | Tk GUI, thumbnails, entry point `main()` |
| `fullhill.py` | Full-hill stack pipeline (Python port of `fullhill.pro`) |
| `stars_analysis.py` | Stars plane computation, refcat, WCS overlays |
| `lucy_spice.py` | SPICE + WCS target prediction and geometry |
| `lucy_getpsf.py` | PSF estimation (DAO-style pipeline) |
| `satsearch_config.py` | Loads `satsearch.toml` |
| `satsearch.toml.example` | Example configuration |
| `g1g2tab.dat` | Phase table for fake-satellite brightness (`fullhill`) |
| `tests/` | Unit tests |

IDL `.pro` files in the repo are reference ports; the running code is Python.

---

## Troubleshooting

- **`FileNotFoundError: No satsearch.toml found`** — Create `satsearch.toml` from the example or set `SATSEARCH_CONFIG`.
- **SPICE errors** — Verify meta-kernel paths, kernel coverage for your observation times, and that FITS **MIDUTCJD** (or equivalent) is present and valid.
- **Stars / refcat errors** — Confirm `refcat_exe` and `refcat_dir` exist and match the atlas-refcat layout expected by your binary.
- **Empty or wrong thumbnails** — Check filters (EXPTIME, 1×1, include/exclude globs) and that paths point at science extensions with data.
