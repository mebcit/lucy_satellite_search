# satsearch

Desktop tools for **Lucy LORRI** (and similar) FITS data: a scrollable thumbnail browser, a **Stack** view that runs the Python port of the `fullhill.pro` pipeline, and a **Stars** analysis window (refcat overlays, SPICE geometry, optional fake-satellite model, interactive alignment). The GUI uses **Tkinter**; science code uses **Astropy**, **NumPy/SciPy**, **Photutils**, **Pillow**, and **SpiceyPy** where SPICE is enabled.

This repository is intended to be cloned and run on a **local machine** with your own FITS trees, SPICE kernels, and (for Stars) an **atlas-refcat** installation.

---

## Configuration (required)

**The application will not start without valid `satsearch.toml` and `encounters.toml`.** There are no built-in default paths for site-specific files: missing files, missing tables, missing keys, empty strings, or paths that do not exist on disk all cause an immediate **`FileNotFoundError`** or **`ValueError`** when you run the program. Environment-variable overrides for refcat paths or “discover `satsearch.toml` from the current working directory” are not supported.

**What you must do after cloning:**

1. **Create** `satsearch.toml` — copy the template and edit every placeholder path:
   ```bash
   cp satsearch.toml.example satsearch.toml
   ```
2. **Create** `encounters.toml` — copy the encounter template and edit paths for each target (at least one `[[encounter]]` block is required):
   ```bash
   cp encounters.toml.example encounters.toml
   ```
3. **Place** those files next to `satsearch_config.py` in the repository root (same directory as `satsearch.py`), **or** set `SATSEARCH_CONFIG` / `SATSEARCH_ENCOUNTERS` to absolute paths before launching.
4. **Fill in** all three tables in `satsearch.toml` — **`[spice]`**, **`[refcat]`**, **`[paths]`** — using the [required tables and keys](#required-tables-and-keys) below. Example values in `satsearch.toml.example` are placeholders for another machine; they will not work on yours until you replace them.
5. **Fill in** every `[[encounter]]` in `encounters.toml` using the [encounter keys](#encounter-blocks-encounterstoml) section below. The **Encounter** control lists each encounter’s `id`; switching encounters updates NAIF `target_body`, which bare kernel files are loaded (from `meta_kernel_dir` in `satsearch.toml`), the default FITS directory, and the **default** Hill sphere radius (you can still edit the Hill field afterward).

Until those steps are done correctly, `python satsearch.py` (and `--stars`) will exit on startup when configuration is loaded.

### Required tables and keys

Every row is **required**; the loader rejects the file if any key is missing, not a string, blank, or points at a non-existent path (where “must exist” applies).

| Table | Key | Rule |
|-------|-----|------|
| `[spice]` | `meta_kernel_dir` | Non-empty string; path must be an **existing directory** containing NAIF meta-kernel (`.tm`) files. |
| `[spice]` | `kernel_data_root` | **Optional.** If set, non-empty string; path must be an **existing directory** — the tree that contains `spk/`, `fk/`, `ik/`, etc. Meta-kernel files are **not** edited on disk; at load time, text lines that reference the snapshot prefix (see below) are rewritten to this root in a temporary `.tm` before CSPICE `furnsh`. Omit this key to furnish meta-kernels exactly as written. |
| `[spice]` | `path_values_snapshot_prefix` | **Optional.** Non-empty string; the absolute path prefix baked into meta-kernels from the source environment (Lucy SOC kernels typically use `/mnt/lucy/soc/spice`). Meaningful only when `kernel_data_root` is set; if omitted, the default prefix is `/mnt/lucy/soc/spice`. Must not be set without `kernel_data_root`. |
| `[spice]` | `observer_body` | Non-empty string. |
| `[spice]` | `sun_body` | Non-empty string. |
| `[spice]` | `frame` | Non-empty string (e.g. `J2000`). |
| `[spice]` | `abcorr` | Non-empty string (e.g. `NONE`). |
| `[refcat]` | `executable` | Non-empty string; path must be an **existing file** (the `refcat` binary). |
| `[refcat]` | `catalog_dir` | Non-empty string; path must be an **existing directory** (atlas-refcat data root). |
| `[paths]` | `default_fits_directory` | Non-empty string; path must be an **existing directory** (used for validation and as a fallback if encounters are not loaded; the thumbnail browser’s **initial** directory comes from the **first** `[[encounter]]` in `encounters.toml` after startup). |

### Encounter blocks (`encounters.toml`)

Each `[[encounter]]` row is **required** to include every key below; `id` values must be unique. Kernel file names are combined with ``[spice].meta_kernel_dir`` from ``satsearch.toml``. The FITS default for each encounter is a **bare subdirectory name** resolved as ``parent([paths].default_fits_directory) / <name>`` from ``satsearch.toml`` (so `[paths].default_fits_directory` should normally be ``…/llori/<first-encounter-subdir>`` and each encounter names a sibling folder under the same parent).

| Key | Rule |
|-----|------|
| `id` | Non-empty string; short name shown in the **Encounter** combobox (e.g. `DJ`). |
| `target_body` | Non-empty NAIF body name for SPICE (e.g. `DONALDJOHANSON`); must match the loaded kernels for that encounter. |
| `meta_kernels` | Non-empty list of strings (or one string); each entry must be a **bare file name** only (no `/` or `\`). The file ``[spice].meta_kernel_dir / <name>`` must exist. |
| `default_fits_directory` | Non-empty string; a **single directory name** only (no `/` or `\`). The directory ``parent([paths].default_fits_directory) / <name>`` must exist (default folder when that encounter is selected). |
| `hill_sphere_km` | Number **> 0**; default Hill radius in km for the thumbnail overlay when that encounter is selected (still editable in the UI). |
| `closest_approach_utc` | **Optional** on first edit; ISO-8601-like UTC timestamp string. If omitted, the **first** application load computes it from SPICE (1-minute grid over 24 hours from the first sorted FITS ``MIDUTCJD`` in that encounter’s directory), writes it into ``encounters.toml``, and later loads read it from disk. Saving uses ``tomli_w`` (comments/formatting in that file may change). Thumbnail **Δt CA** and Stack timing use this JD for the active encounter (legacy :data:`satsearch.CA_REF_JD` only when this key is absent). |

### Where the config files are loaded from

**`satsearch.toml`** search order (first match wins):

1. Path given by **`SATSEARCH_CONFIG`** (if set; relative paths are resolved against the process current working directory).
2. **`satsearch.toml`** in the same directory as **`satsearch_config.py`** (the usual layout after `cp satsearch.toml.example satsearch.toml` in the repo root).

**`encounters.toml`** search order:

1. Path given by **`SATSEARCH_ENCOUNTERS`** (if set; relative paths are resolved against the current working directory).
2. **`encounters.toml`** next to **`satsearch_config.py`**.

There is no other fallback (for example, no automatic `satsearch.toml` lookup from an arbitrary working directory).

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
| tomli_w   | Writing ``encounters.toml`` when auto-filling ``closest_approach_utc`` |

### System / GUI

- A working **Tk** (Tkinter). On many Linux distributions the package is `python3-tk` (name varies by distro). Windows/macOS Python installers usually include Tk.
- An **X server** (or equivalent) if you run the GUI over SSH.
- **Font files** (optional): magnitude labels in Stars try DejaVu or Liberation Sans under `/usr/share/fonts/...`; if missing, a default font is used.

### Data and external programs (site-specific)

1. **`satsearch.toml`** and **`encounters.toml`** — Mandatory; see **[Configuration (required)](#configuration-required)**. `satsearch.toml` holds refcat, `[paths].default_fits_directory`, and **`[spice]`** including the **`meta_kernel_dir`** where `.tm` meta-kernels live. **`target_body`** is **not** in `satsearch.toml`; it is set per encounter in `encounters.toml`. Each encounter lists **bare kernel file names** resolved under `meta_kernel_dir`, and a **bare FITS subdirectory name** resolved under the parent of `[paths].default_fits_directory`. The browser’s **initial** FITS directory comes from the **first** `[[encounter]]` at startup.

2. **CSPICE / SPICE kernels** — Not included in the repository. Put meta-kernels under `[spice].meta_kernel_dir`, and list the file names in each `[[encounter]].meta_kernels`. If those `.tm` files still point at another machine’s `PATH_VALUES` (for example `/mnt/lucy/soc/spice`), set optional `[spice].kernel_data_root` to your local kernel tree so satsearch can rewrite paths when loading; see the [required tables](#required-tables-and-keys) row for `kernel_data_root`.

3. **atlas-refcat** — Not included. Install the `refcat` binary and catalog tiles locally; set `[refcat].executable` and `[refcat].catalog_dir` in `satsearch.toml`.

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

### 4. Create `encounters.toml` (mandatory)

Copy `encounters.toml.example` to `encounters.toml` and add one `[[encounter]]` block per flyby / target. See [Encounter blocks](#encounter-blocks-encounterstoml) in **[Configuration (required)](#configuration-required)**.

### 5. Create `satsearch.toml` (mandatory)

Follow **[Configuration (required)](#configuration-required)** above: copy `satsearch.toml.example` to `satsearch.toml`, fill every table using the [checklist](#required-tables-and-keys), and ensure all paths exist. To use config files outside the repo root:

```bash
export SATSEARCH_CONFIG=/path/to/my-satsearch.toml
export SATSEARCH_ENCOUNTERS=/path/to/my-encounters.toml
```

---

## Usage

**Prerequisite:** valid `satsearch.toml` and `encounters.toml` (see [Configuration (required)](#configuration-required)); otherwise the process exits when loading config.

Run the application from the **repository root** (or ensure the repo is on `PYTHONPATH`) so imports resolve:

```bash
python satsearch.py
```

### Command-line options

| Option | Meaning |
|--------|---------|
| `--minexp SEC` | Default **minimum EXPTIME** (seconds) for the Min EXPTIME filter in the main window (same as the UI default). |
| `--stars FITS [FITS ...]` | Skip the thumbnail browser and open **only** the **Stars** window for the given FITS paths (one group). The root window is minimized; closing Stars exits. Example: `python satsearch.py --stars /data/a.fit /data/b.fit` |

### Main window (thumbnail browser)

1. Choose **Encounter** (top left), then set **Directory** to a folder containing `.fit` / `.fits` files (or use **Browse…** — the encounter selection fills a default directory from `encounters.toml`).
2. **Load** — Parses the **Kernel** field (bare meta-kernel **file names** separated by `;`, resolved under `[spice].meta_kernel_dir`). If the resolved set differs from what CSPICE has loaded, kernels are reloaded (`kclear` / `furnsh`). Then scans the tree (subject to **Filter out** / **Only include** glob patterns and **Min/Max EXPTIME**, **1×1 only**, **Groups** mode). Names that match the active encounter’s defaults clear any session-only kernel override.
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
| `satsearch.py` | Tk GUI, thumbnails, entry point `main()` |
| `fullhill.py` | Full-hill stack pipeline (Python port of `fullhill.pro`) |
| `stars_analysis.py` | Stars plane computation, refcat, WCS overlays |
| `lucy_spice.py` | SPICE + WCS target prediction and geometry |
| `lucy_getpsf.py` | PSF estimation (DAO-style pipeline) |
| `satsearch_config.py` | Loads `satsearch.toml`, `encounters.toml`, and runtime SPICE selection (`get_spice_runtime`) |
| `encounters.toml.example` | **Mandatory** template for per-target SPICE + FITS + Hill defaults: copy to `encounters.toml` |
| `satsearch.toml.example` | **Mandatory** template: copy to `satsearch.toml` and edit all paths (see [Configuration (required)](#configuration-required)) |
| `g1g2tab.dat` | Phase table for fake-satellite brightness (`fullhill`) |
| `tests/` | Unit tests |

IDL `.pro` files in the repo are reference ports; the running code is Python.

---

## Troubleshooting

- **`FileNotFoundError` / `ValueError` from configuration** — See [Configuration (required)](#configuration-required). For `satsearch.toml`, see [Required tables and keys](#required-tables-and-keys). For `encounters.toml`, see [Encounter blocks](#encounter-blocks-encounterstoml). Create both files from the examples (next to `satsearch_config.py`) or set `SATSEARCH_CONFIG` / `SATSEARCH_ENCOUNTERS`. Ensure every path exists on disk.
- **SPICE errors** — Verify `[spice].meta_kernel_dir`, that each bare name in `encounters.toml` / the Kernel field names a real `.tm` file in that directory, kernel coverage for your observation times, and that FITS **MIDUTCJD** (or equivalent) is present and valid.
- **Stars / refcat errors** — Confirm `[refcat].executable` and `[refcat].catalog_dir` exist and match the atlas-refcat layout expected by your binary.
- **Empty or wrong thumbnails** — Check filters (EXPTIME, 1×1, include/exclude globs) and that paths point at science extensions with data.
