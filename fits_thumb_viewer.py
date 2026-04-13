#!/usr/bin/env python3
"""
Scrollable grid viewer for FITS thumbnails (e.g. Lucy LORRI).
Intensity scale: sky ≈ sigma-clipped median, RMS ≈ std, display range [sky-4*rms, sky+5*rms].

Thumbnails are saved next to each FITS as ``<stem>_thumb_h<km>.png`` (Hill radius in km) and reused when
newer than the FITS. Legacy ``*_thumb.png`` files are removed on Load so caches regenerate with the overlay.

Only visible rows (plus a small overscan) are materialized so the X server is not exhausted with
many files (avoids BadAlloc / X_CreatePixmap errors).

Arcseconds per pixel for the Hill line comes from the filename (``…_1x1_…`` → 1″/px, ``…_4x4_…`` → 4″/px), else 1.
Small-angle: ``θ≈H/R`` rad → arcsec → native pixels via ``θ_arcsec / (arcsec/pixel)``.
When the FITS header has no ``SPCTRANG``/``SPCTRANGE``, Lucy–target distance for the Hill line
can come from SPICE (see :mod:`lucy_spice`) if ``satsearch.toml`` and kernels are configured.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image, ImageDraw, ImageTk


FITS_EXTENSIONS = (".fits", ".fit", ".fts")
DEFAULT_SUBDIR = Path("..") / "llori" / "2025110"
CA_REF_JD = 2460786.2439811
DEFAULT_HILL_KM = 711.0
# Filename pattern e.g. ``lor_…_4x4_sci_01.fit`` → 4 arcsec / pixel.
_BINNING_RE = re.compile(r"_(\d+)x(\d+)_", re.IGNORECASE)
DEFAULT_ARCSEC_PER_PIXEL = 1.0
# Old cache files before Hill radius was encoded in the filename.
LEGACY_THUMB_GLOB = "*_thumb.png"
THUMB_MAX = 160
FULL_HILL_SIZE = 1024
# Local sky statistics on left-click (Stars / Stack): median, RMS from median, 5σ limits.
SKY_REPORT_BOX = 30
COLS = 4
WORKERS = 2
# Fixed row height for virtual scrolling (avoids one X pixmap per file — BadAlloc).
ROW_HEIGHT = 300
ROW_OVERSCAN = 1
# Groups mode: colored vertical bar at left (cycles by group index).
GROUP_ACCENT_WIDTH = 12
GROUP_ACCENT_COLORS = ("#1d4ed8", "#047857", "#b45309", "#b91c1c", "#6d28d9", "#0e7490")
# Ignore absurd canvas heights before the window is mapped (prevents materializing every row).
MAX_VIEWPORT_PX = 4096
# Hard cap on rows materialized per sync (guards bogus canvas geometry before map).
MAX_ROWS_PER_SYNC = 10


def _clipped_square_box(
    nr: float, nc: float, h: int, w: int, box: int = SKY_REPORT_BOX
) -> tuple[int, int, int, int]:
    """``r0, r1, c0, c1`` for a ``box``×``box`` region centered at ``(nr, nc)``, clipped to the image."""
    if h < 1 or w < 1:
        return 0, 0, 0, 0
    box = int(box)
    bh = min(box, h)
    bw = min(box, w)
    half_r = bh // 2
    half_c = bw // 2
    ir = int(round(nr))
    ic = int(round(nc))
    r0 = max(0, min(ir - half_r, h - bh))
    c0 = max(0, min(ic - half_c, w - bw))
    return r0, r0 + bh, c0, c0 + bw


def _patch_median_rms_npix(patch: np.ndarray) -> tuple[float, float, int]:
    p = np.asarray(patch, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    n = int(p.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(p))
    rms = float(np.sqrt(np.mean((p - med) ** 2)))
    return med, rms, n


def sky_box_five_sigma_integrated(rms: float, n_pix: int) -> float:
    """Background-limited total counts at 5σ in an aperture of ``n_pix`` independent pixels."""
    if n_pix <= 0 or not math.isfinite(rms):
        return float("nan")
    return float(5.0 * float(rms) * math.sqrt(float(n_pix)))


def format_sky_box_report(
    med: float,
    rms: float,
    five_pix: float,
    flux_5: float,
    n_pix: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    diam_line: str | None,
    *,
    title_note: str = "",
    error: str | None = None,
) -> str:
    """Multi-line text for the Stars / Stack 30×30 sky-box dialog."""
    head = "30×30 sky box"
    if title_note:
        head += f" ({title_note})"
    lines = [
        head,
        f"Box rows [{r0}:{r1}), cols [{c0}:{c1})  ({n_pix} px)",
        f"median (sky) = {med:.8g}",
        f"RMS from median = {rms:.8g}",
        f"5×RMS (per-pixel) = {five_pix:.8g}",
        f"5σ integrated limit = {flux_5:.8g} counts (5·RMS·√N, i.i.d. pixel noise)",
        "",
        "Equivalent sphere diameter (fakesat model, albedo from window):",
    ]
    if error:
        lines.append(f"  —  ({error})")
    elif diam_line:
        lines.append(f"  {diam_line}")
    else:
        lines.append("  —")
    lines.extend(
        [
            "",
            "Diameter inverts fakesat_flux(D)×EXPTIME to match that integrated limit; treat as",
            "order-of-magnitude (local box vs PSF-integrated point source).",
        ]
    )
    return "\n".join(lines)


def _display_to_native_xy(
    ix: int, iy: int, iw: int, ih: int, nw: int, nh: int
) -> tuple[float, float]:
    fx = ix * (nw - 1) / max(iw - 1, 1)
    fy = iy * (nh - 1) / max(ih - 1, 1)
    return float(fx), float(fy)


def default_root_dir() -> Path:
    try:
        from satsearch_config import get_config

        return get_config().paths.default_fits_directory
    except Exception:
        base = Path(__file__).resolve().parent
        return (base / DEFAULT_SUBDIR).resolve()


def list_fits_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in FITS_EXTENSIONS:
            continue
        out.append(p)
    return out


def arcsec_per_pixel_from_filename(path: Path) -> float:
    """Parse ``…_NxN_…`` in the basename (e.g. ``_1x1_``, ``_4x4_``). Square binning N gives N″/pixel."""
    m = _BINNING_RE.search(path.name)
    if not m:
        return DEFAULT_ARCSEC_PER_PIXEL
    a, b = int(m.group(1)), int(m.group(2))
    if a <= 0 or b <= 0:
        return DEFAULT_ARCSEC_PER_PIXEL
    if a == b:
        return float(a)
    return math.sqrt(float(a) * float(b))


def is_1x1_binned_filename(path: Path) -> bool:
    """True if basename contains ``…_1x1_…`` (LORRI-style binning tag)."""
    m = _BINNING_RE.search(path.name)
    if not m:
        return False
    return int(m.group(1)) == 1 and int(m.group(2)) == 1


def thumb_cache_path(fits_path: Path, hill_km: float, arcsec_per_pixel: float) -> Path:
    """PNG next to the FITS; includes Hill radius and plate scale so caches stay valid."""
    h = float(hill_km)
    hk = int(round(h)) if abs(h - round(h)) < 1e-9 else round(h, 6)
    ap = float(arcsec_per_pixel)
    ap_s = int(round(ap)) if abs(ap - round(ap)) < 1e-9 else round(ap, 3)
    return fits_path.with_name(f"{fits_path.stem}_thumb_h{hk}_b{ap_s}.png")


def delete_legacy_thumb_pngs(directory: Path) -> None:
    """Remove old cache PNGs: plain ``*_thumb.png`` and ``*_thumb_h*`` without ``_b`` (binning) in the name."""
    for p in directory.glob(LEGACY_THUMB_GLOB):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass
    for p in directory.glob("*_thumb_h*.png"):
        if not p.is_file():
            continue
        if "_b" not in p.stem:
            try:
                p.unlink()
            except OSError:
                pass


def display_fits_name(path: Path) -> str:
    """Strip leading ``lor_`` from LORRI-style names for display only."""
    name = path.name
    if name.startswith("lor_"):
        return name[4:]
    return name


def headers_for_metadata(path: Path) -> list[fits.Header]:
    """Primary HDU header only.

    LORRI science FITS used here are single-HDU; observation keywords (``SPCTRANG``,
    ``MIDUTCJD``, …) are read from the primary header.
    """
    with fits.open(path, memmap=True) as hdul:
        return [hdul[0].header]


def _range_km_from_headers(headers: list[fits.Header]) -> float | None:
    """Lucy–target distance in km: ``SPCTRANG``, then ``SPCTRANGE`` if present."""
    for key in ("SPCTRANG", "SPCTRANGE"):
        v = _header_float_chain(headers, key)
        if v is not None and math.isfinite(v) and v > 0:
            return float(v)
    return None


def _header_float_chain(headers: list[fits.Header], key: str) -> float | None:
    for h in headers:
        v = header_float(h, key)
        if v is not None:
            return v
    return None


def _header_str_chain(headers: list[fits.Header], key: str) -> str:
    for h in headers:
        s = header_str(h, key)
        if s:
            return s
    return ""


def _header_int_chain(headers: list[fits.Header], key: str) -> int | None:
    for h in headers:
        for k in (key, key.upper(), key.lower()):
            if k in h:
                try:
                    return int(round(float(h[k])))
                except (TypeError, ValueError):
                    pass
    return None


def _parse_julian_day_scalar(value: object) -> float | None:
    """FITS often stores JD as a string like ``'JD 2460785.6612216'`` which ``float()`` cannot parse."""
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    if s.upper().startswith("JD"):
        s = s[2:].strip()
    for tok in s.replace(",", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def _midutcjd_from_headers(headers: list[fits.Header]) -> float | None:
    for h in headers:
        for k in ("MIDUTCJD", "midutcjd"):
            if k not in h:
                continue
            jd = _parse_julian_day_scalar(h[k])
            if jd is not None:
                return jd
    return None


def get_image_data_and_header(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        if data is None:
            for hdu in hdul[1:]:
                if hdu.data is not None:
                    data = np.asarray(hdu.data, dtype=np.float64)
                    header = hdu.header
                    break
            else:
                raise ValueError("No image data in FITS")
        else:
            data = np.asarray(data, dtype=np.float64)
    return data, header


def _float_from_header_value(val: object) -> float | None:
    """Parse a FITS header value as float (handles strings with units or extra text)."""
    if val is None:
        return None
    if isinstance(val, (float, int, np.floating, np.integer)):
        f = float(val)
        return f if math.isfinite(f) else None
    s = str(val).strip()
    if not s:
        return None
    for tok in s.replace(",", " ").split():
        try:
            f = float(tok)
            if math.isfinite(f):
                return f
        except ValueError:
            continue
    return None


def header_float(h: fits.Header, key: str) -> float | None:
    """Read a numeric keyword; match ``HIERARCH …`` / long names and tokenized strings."""
    base = key.upper()
    for k in (key, key.upper(), key.lower()):
        if k in h:
            v = _float_from_header_value(h[k])
            if v is not None:
                return v
    for k in h.keys():
        kn = k.upper().replace("HIERARCH", "").strip()
        if kn == base or kn.endswith(base):
            v = _float_from_header_value(h[k])
            if v is not None:
                return v
    return None


def header_str(h: fits.Header, key: str) -> str:
    for k in (key, key.upper(), key.lower()):
        if k in h:
            v = h[k]
            if v is not None:
                return str(v).strip()
    return ""


def primary_exptime_seconds(path: Path) -> float | None:
    """EXPTIME from the primary header only (header read, no array load)."""
    try:
        h = fits.getheader(path, memmap=False)
        return header_float(h, "EXPTIME")
    except Exception:
        return None


def primary_midutcjd(path: Path) -> float | None:
    """MIDUTCJD (or equivalent) from headers that also carry EXPTIME metadata."""
    try:
        headers = headers_for_metadata(path)
        return _midutcjd_from_headers(headers)
    except Exception:
        return None


def sap_search_string(path: Path) -> str:
    """Concatenated SAP-related FITS header text (SAPID, SAP) for substring matching."""
    try:
        headers = headers_for_metadata(path)
        parts: list[str] = []
        for key in ("SAPID", "SAP"):
            s = _header_str_chain(headers, key)
            if s:
                parts.append(s)
        return " ".join(parts)
    except Exception:
        return ""


def parse_comma_keywords(s: str) -> list[str]:
    """Split on commas; strip whitespace; drop empty tokens (no quotes required)."""
    return [p.strip() for p in s.split(",") if p.strip()]


def filter_paths_by_sap_keywords(
    paths: list[Path],
    exclude_keywords: list[str],
    include_keywords: list[str],
) -> list[Path]:
    """Drop paths whose SAP text matches any exclude keyword; then keep only paths matching
    at least one include keyword if ``include_keywords`` is non-empty. Matching is
    case-insensitive substring search.
    """
    if not exclude_keywords and not include_keywords:
        return paths
    out: list[Path] = []
    for p in paths:
        hay = sap_search_string(p).lower()
        if exclude_keywords and any(kw.lower() in hay for kw in exclude_keywords):
            continue
        if include_keywords and not any(kw.lower() in hay for kw in include_keywords):
            continue
        out.append(p)
    return out


# Max (latest − earliest) observation time within one burst, in seconds (for exposure groups).
_GROUP_MAX_TIME_SPAN_SEC = 10.0


def compute_exposure_groups(paths: list[Path], max_span_sec: float = _GROUP_MAX_TIME_SPAN_SEC) -> list[list[Path]]:
    """Cluster paths into simultaneous bursts.

    Observations are sorted by time (JD as seconds). The timeline is split into
    consecutive segments: each segment is the longest run starting at the current
    file such that every file in the run falls within ``max_span_sec`` of the
    **earliest** time in that run (so max − min ≤ ``max_span_sec`` with no fixed
    clock-aligned boundaries). Within each segment, paths are partitioned by
    (rounded) EXPTIME; only the longest-EXPTIME subset is kept. Singleton groups
    are omitted. Paths missing MIDUTCJD or EXPTIME are omitted.
    """
    records: list[tuple[Path, float, float]] = []
    for p in paths:
        jd = primary_midutcjd(p)
        et = primary_exptime_seconds(p)
        if jd is None or et is None:
            continue
        t_sec = float(jd) * 86400.0
        records.append((p, t_sec, float(et)))
    if not records:
        return []
    records.sort(key=lambda r: (r[1], r[0].name))

    groups: list[list[Path]] = []
    i = 0
    n = len(records)
    while i < n:
        t0 = records[i][1]
        j = i
        while j + 1 < n and records[j + 1][1] - t0 <= max_span_sec:
            j += 1
        segment = records[i : j + 1]
        by_et: dict[float, list[tuple[Path, float]]] = {}
        for p, t_s, et in segment:
            ek = round(et, 6)
            by_et.setdefault(ek, []).append((p, t_s))
        best_et = max(by_et.keys())
        pairs = by_et[best_et]
        pairs.sort(key=lambda t: (t[1], t[0].name))
        chosen = [t[0] for t in pairs]
        if len(chosen) > 1:
            groups.append(chosen)
        i = j + 1
    return groups


def filter_fits_by_exptime(paths: list[Path], min_exp: float, max_exp: float) -> list[Path]:
    """Keep files whose primary EXPTIME is in [min_exp, max_exp] when those bounds apply."""
    if min_exp <= 0 and max_exp == float("inf"):
        return paths
    out: list[Path] = []
    for p in paths:
        et = primary_exptime_seconds(p)
        if et is None:
            continue
        if min_exp > 0 and et < min_exp:
            continue
        if max_exp < float("inf") and et > max_exp:
            continue
        out.append(p)
    return out


def sky_scale(data: np.ndarray) -> tuple[float, float, float, float]:
    """Returns median (sky), std (rms), vmin, vmax for display."""
    if data.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    flat = data[np.isfinite(data)]
    if flat.size == 0:
        return 0.0, 1.0, 0.0, 1.0
    mean, median, std = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
    sky = float(median)
    rms = float(std) if std > 0 else float(np.std(flat))
    if rms <= 0:
        rms = 1.0
    vmin = sky - 4.0 * rms
    vmax = sky + 5.0 * rms
    if vmax <= vmin:
        vmax = vmin + 1.0
    return sky, rms, vmin, vmax


def fits_to_scaled_pil(path: Path, max_side: int) -> Image.Image:
    """Load FITS and render with the same sky stretch as thumbnails (no Hill overlay)."""
    data, _ = get_image_data_and_header(path)
    _, _, vmin, vmax = sky_scale(data)
    return data_to_thumbnail_u8(data, vmin, vmax, max_side)


def data_to_thumbnail_u8(data: np.ndarray, vmin: float, vmax: float, size: int) -> Image.Image:
    """Resize preserving aspect ratio; max side = size."""
    d = np.clip(data, vmin, vmax)
    d = (d - vmin) / (vmax - vmin)
    d = (d * 255.0).astype(np.uint8)
    h, w = d.shape[:2]
    if h == 0 or w == 0:
        img = Image.new("L", (1, 1), 0)
        return img.resize((size, size), Image.Resampling.LANCZOS)
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(d, mode="L")
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def overlay_hill_sphere_line(
    img: Image.Image,
    native_hw: tuple[int, int],
    hill_km: float,
    range_km: float | None,
    arcsec_per_pixel: float = DEFAULT_ARCSEC_PER_PIXEL,
) -> Image.Image:
    """Draw a red horizontal segment near the top: center dot, ±Hill km at ``range_km``.

    Small-angle: ``θ_rad ≈ hill_km / range_km``, then ``θ_arcsec = θ_rad × (180/π) × 3600``;
    half-length in native pixels = ``θ_arcsec / arcsec_per_pixel`` (from filename ``_1x1_`` / ``_4x4_``).
    """
    nh, nw = native_hw
    tw, th = img.size
    rgb = img.convert("RGB")
    asp = float(arcsec_per_pixel)
    if (
        range_km is None
        or range_km <= 0
        or hill_km <= 0
        or nw < 2
        or nh < 2
        or asp <= 0
    ):
        return rgb
    theta_rad = float(hill_km) / float(range_km)
    theta_arcsec = theta_rad * (180.0 / np.pi) * 3600.0
    half_native_px = theta_arcsec / asp
    scale = tw / float(nw)
    half_thumb = float(half_native_px * scale)
    half_thumb = min(half_thumb, tw / 2.0 - 1.0)
    if half_thumb < 0.5:
        return rgb
    cx = tw / 2.0
    y_line = max(4, int(th * 0.08))
    x0 = cx - half_thumb
    x1 = cx + half_thumb
    dr = ImageDraw.Draw(rgb)
    r_dot = max(2, min(th, tw) // 64)
    dr.ellipse(
        [cx - r_dot, y_line - r_dot, cx + r_dot, y_line + r_dot],
        fill=(255, 0, 0),
        outline=(255, 0, 0),
    )
    w_line = max(1, min(th, tw) // 120)
    dr.line([(x0, y_line), (x1, y_line)], fill=(255, 0, 0), width=w_line)
    return rgb


def _fill_header_meta(meta: dict, headers: list[fits.Header]) -> None:
    meta["sapid"] = _header_str_chain(headers, "SAPID")
    jd = _midutcjd_from_headers(headers)
    if jd is not None:
        meta["hours_to_ca"] = (jd - CA_REF_JD) * 24.0
    et = _header_float_chain(headers, "EXPTIME")
    if et is not None:
        meta["exptime"] = et
    rk = _range_km_from_headers(headers)
    meta["range_km"] = rk
    if rk is not None:
        meta["range_int"] = int(round(rk))
    else:
        meta["range_int"] = None


def load_thumb_job(path: Path, hill_km: float) -> dict:
    meta = {
        "path": path,
        "basename": display_fits_name(path),
        "sapid": "",
        "hours_to_ca": None,
        "exptime": None,
        "range_km": None,
        "range_int": None,
        "arcsec_per_pixel": None,
        "image": None,
        "photo": None,
        "cached": False,
        "cache_error": None,
    }
    try:
        headers = headers_for_metadata(path)
        header_rk = _range_km_from_headers(headers)
        _fill_header_meta(meta, headers)
        if header_rk is None and meta.get("range_km") is None:
            try:
                from lucy_spice import range_km_for_display

                rk_spice = range_km_for_display(path)
                if rk_spice is not None:
                    meta["range_km"] = rk_spice
                    meta["range_int"] = int(round(rk_spice))
            except Exception:
                pass
        spice_filled = header_rk is None and meta.get("range_km") is not None

        asp = arcsec_per_pixel_from_filename(path)
        meta["arcsec_per_pixel"] = asp

        cache = thumb_cache_path(path, hill_km, asp)
        fits_mtime = path.stat().st_mtime
        use_cache = (
            cache.is_file()
            and cache.stat().st_mtime >= fits_mtime
            and not spice_filled
        )

        if use_cache:
            img = Image.open(cache).convert("RGB")
            meta["image"] = img
            meta["cached"] = True
        else:
            data, _hdr_img = get_image_data_and_header(path)
            sky, rms, vmin, vmax = sky_scale(data)
            meta["sky"] = sky
            meta["rms"] = rms
            native_hw = (int(data.shape[0]), int(data.shape[1]))
            img_l = data_to_thumbnail_u8(data, vmin, vmax, THUMB_MAX)
            img = overlay_hill_sphere_line(
                img_l, native_hw, hill_km, meta.get("range_km"), asp
            )
            meta["image"] = img
            try:
                img.save(cache, "PNG")
            except OSError as e:
                meta["cache_error"] = str(e)
    except Exception as e:
        meta["error"] = str(e)
    return meta


def _format_plane_shift_px(v: float) -> str:
    """Sign, two digits before the decimal, one after (``+01.2``). Wider fallback if ``|v|`` ≥ 100."""
    r = round(float(v), 1)
    if r >= 100.0 or r <= -100.0:
        return f"{r:+.1f}"
    sign = "+" if r >= 0 else "-"
    x = abs(r)
    whole = int(math.floor(x + 1e-9))
    frac_digit = int(round((x - whole) * 10))
    if frac_digit >= 10:
        whole += 1
        frac_digit = 0
    if whole >= 100:
        return f"{r:+.1f}"
    return f"{sign}{whole:02d}.{frac_digit}"


class FullHillWindow(tk.Toplevel):
    """Full-hill pipeline (fullhill.pro) with stacks ``imb``, ``imbs``, ``imz``, ``imzs``, medians."""

    _STACK_CHOICES: tuple[tuple[str, str], ...] = (
        ("imb", "imb"),
        ("imbs", "imbs"),
        ("imz", "imz"),
        ("imzs", "imzs"),
        ("median_imz", "median(imz)"),
        ("median_imzs", "median(imzs)"),
    )
    _STACK_MEDIAN = frozenset({"median_imz", "median_imzs"})

    def __init__(
        self,
        master: tk.Tk,
        selected_paths: list[Path],
        all_sorted: list[Path],
    ) -> None:
        super().__init__(master)
        self.title("Stack")
        self._selected_paths = list(selected_paths)
        self._all_sorted = list(all_sorted)
        self.transient(master)
        self._photo: ImageTk.PhotoImage | None = None
        self._result: object | None = None
        self._fullhill_prep: object | None = None
        self._busy = False

        self._diam_var = tk.StringVar(value="0")
        self._alb_var = tk.StringVar(value="0.041")
        self._satdist_var = tk.StringVar(value="200")
        self._satang_var = tk.StringVar(value="0")

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="diam (m):").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._diam_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(top, text="albedo:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._alb_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(top, text="sat dist (km):").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._satdist_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(top, text="sat ang (deg):").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self._satang_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(top, text="Run stack", command=self._run_pipeline).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Button(top, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        self._status = ttk.Label(
            self,
            text=f"{len(selected_paths)} file(s)  |  short mate = index−2 in full folder list  |  running with defaults…",
            wraplength=920,
        )
        self._status.pack(fill=tk.X, padx=8, pady=(0, 4))

        var_row = ttk.Frame(self, padding=(8, 0))
        var_row.pack(fill=tk.X)
        ttk.Label(var_row, text="Stack:").pack(side=tk.LEFT)
        self._stack_var = tk.StringVar(value="imb")
        for value, label in self._STACK_CHOICES:
            ttk.Radiobutton(
                var_row,
                text=label,
                value=value,
                variable=self._stack_var,
                command=self._refresh_view,
            ).pack(side=tk.LEFT, padx=(8, 0))

        # Classic tk.Scale honors ``length``; ttk.Scale + Canvas still stretched on some Gtk themes.
        # Grid: only the trailing column expands — column 1 stays at the trough width.
        slide_row = ttk.Frame(self, padding=8)
        slide_row.pack(fill=tk.X)
        slide_row.columnconfigure(4, weight=1)
        slide_row.columnconfigure(1, weight=0)

        ttk.Label(slide_row, text="Plane:").grid(row=0, column=0, sticky="w")
        self.update_idletasks()
        slider_w = max(120, self.winfo_screenwidth() // 3)
        self._idx_var = tk.DoubleVar(value=0.0)
        self._slider = tk.Scale(
            slide_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            length=slider_w,
            resolution=1,
            showvalue=0,
            variable=self._idx_var,
            command=self._on_slider,
        )
        self._slider.grid(row=0, column=1, padx=(8, 4), sticky="w")
        self._play_active = False
        self._play_after: str | int | None = None
        self._play_btn = ttk.Button(
            slide_row,
            text="Play",
            command=self._toggle_play,
        )
        self._play_btn.grid(row=0, column=2, padx=(4, 8), sticky="w")
        self._plane_lbl = ttk.Label(slide_row, text="0 / 0")
        self._plane_lbl.grid(row=0, column=3, sticky="w")
        ttk.Frame(slide_row).grid(row=0, column=4, sticky="nsew")

        self._detail = ttk.Label(self, text="", wraplength=920, justify=tk.LEFT)
        self._detail.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._probe_lbl = ttk.Label(
            self,
            text="",
            wraplength=920,
            justify=tk.LEFT,
            font=("TkFixedFont",),
        )
        self._probe_lbl.pack(fill=tk.X, padx=8, pady=(0, 2))

        img_frame = ttk.Frame(self, padding=8)
        img_frame.pack(fill=tk.BOTH, expand=True)
        self._img_label = ttk.Label(
            img_frame,
            text="Computing stack…",
            anchor=tk.CENTER,
        )
        self._img_label.pack()
        self._img_label.bind("<Motion>", self._on_probe_motion)
        self._img_label.bind("<Leave>", self._on_probe_leave)
        self._img_label.bind("<Button-1>", self._on_stack_sky_box_click)

        self.after_idle(self._run_pipeline)

    def _stop_play(self) -> None:
        if self._play_after is not None:
            try:
                self.after_cancel(self._play_after)
            except tk.TclError:
                pass
            self._play_after = None
        self._play_active = False
        if self._play_btn.winfo_exists():
            self._play_btn.configure(text="Play")

    def _on_stack_sky_box_click(self, event: object) -> None:
        """Left-click: 30×30 local median / RMS, 5σ, equivalent fakesat diameter (this plane)."""
        if self._busy:
            return
        if self._result is None or self._photo is None or self._fullhill_prep is None:
            return
        xy = self._probe_image_xy(event)
        if xy is None:
            return
        ix, iy = xy
        try:
            alb = self._parse_float(self._alb_var, "albedo")
        except ValueError:
            messagebox.showerror("Stack", "Invalid albedo.")
            return

        from fullhill import diameter_for_fakesat_total_counts

        name = self._stack_var.get()
        cube = getattr(self._result, name, None)
        if cube is None:
            return
        prep = self._fullhill_prep
        n_pl = int(prep.rr.shape[0])
        if cube.ndim == 2:
            plane = np.asarray(cube, dtype=np.float64)
            k_geo = n_pl - 1
        else:
            k_geo = int(round(float(self._idx_var.get())))
            k_geo = max(0, min(k_geo, cube.shape[2] - 1))
            plane = np.asarray(cube[:, :, k_geo], dtype=np.float64)

        h, w = int(plane.shape[0]), int(plane.shape[1])
        iw = self._photo.width()
        ih = self._photo.height()
        nc, nr = _display_to_native_xy(ix, iy, iw, ih, w, h)
        r0, r1, c0, c1 = _clipped_square_box(nr, nc, h, w, SKY_REPORT_BOX)
        patch = plane[r0:r1, c0:c1]
        med, rms, n_pix = _patch_median_rms_npix(patch)
        five_pix = 5.0 * rms if np.isfinite(rms) else float("nan")
        flux_5 = sky_box_five_sigma_integrated(rms, n_pix)

        path_m = self._selected_paths[k_geo]
        et = primary_exptime_seconds(path_m)
        et_s = float(et) if et is not None and et > 0 else 1.0
        rr = float(prep.rr[k_geo])
        ph = float(prep.ph_deg[k_geo])
        dk = float(prep.delta_km[k_geo])

        diam = diameter_for_fakesat_total_counts(flux_5, alb, rr, dk, ph, et_s)
        d_s = f"{diam:.4g} m" if diam is not None and math.isfinite(diam) else "—"
        messagebox.showinfo(
            "Stack",
            format_sky_box_report(
                med,
                rms,
                five_pix,
                flux_5,
                n_pix,
                r0,
                r1,
                c0,
                c1,
                d_s,
                title_note="Stack",
            ),
        )

    def _toggle_play(self) -> None:
        if self._play_active:
            self._stop_play()
            return
        if self._result is None:
            return
        n = int(self._result.imb.shape[2])
        if n <= 1:
            return
        if self._stack_var.get() in self._STACK_MEDIAN:
            return
        self._play_active = True
        self._play_btn.configure(text="Stop")
        self._play_after = self.after(200, self._play_step)

    def _play_step(self) -> None:
        self._play_after = None
        if not self._play_active:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._result is None:
            self._stop_play()
            return
        if self._stack_var.get() in self._STACK_MEDIAN:
            self._stop_play()
            return
        n = int(self._result.imb.shape[2])
        if n <= 1:
            self._stop_play()
            return
        k = int(round(float(self._idx_var.get())))
        k = (k + 1) % n
        self._idx_var.set(float(k))
        self._refresh_view()
        if self._play_active:
            self._play_after = self.after(200, self._play_step)

    def destroy(self) -> None:
        self._stop_play()
        super().destroy()

    def _parse_float(self, var: tk.StringVar, name: str) -> float:
        s = var.get().strip()
        v = float(s)
        return v

    def _run_pipeline(self) -> None:
        if self._busy:
            return
        try:
            diam = self._parse_float(self._diam_var, "diam")
            alb = self._parse_float(self._alb_var, "albedo")
            sd = self._parse_float(self._satdist_var, "satdist")
            sa = self._parse_float(self._satang_var, "satang")
        except ValueError as e:
            messagebox.showerror("Stack", f"Invalid parameter: {e}")
            return
        self._busy = True
        use_prep = self._fullhill_prep is not None
        if use_prep:
            self._status.configure(text="Updating satellite model…")
            self._img_label.configure(image="", text="Updating satellite…")
        else:
            self._status.configure(text="Loading FITS, aligning, PSFs…")
            self._img_label.configure(image="", text="Computing stack…")

        paths = list(self._selected_paths)
        all_s = list(self._all_sorted)

        def work() -> None:
            try:
                from fullhill import run_fullhill_from_prep, run_fullhill_prepare

                prep_in = self._fullhill_prep
                if prep_in is None:
                    prep_new = run_fullhill_prepare(paths, all_s)
                    res = run_fullhill_from_prep(prep_new, diam, alb, sd, sa)
                    self.after(0, lambda: self._apply_result(res, prep_new, from_cache=False))
                else:
                    res = run_fullhill_from_prep(prep_in, diam, alb, sd, sa)
                    self.after(0, lambda: self._apply_result(res, from_cache=True))
            except Exception as e:
                self.after(0, lambda err=e: self._fail(err))
            finally:
                self.after(0, self._clear_busy)

        threading.Thread(target=work, daemon=True).start()

    def _clear_busy(self) -> None:
        self._busy = False

    def _fail(self, err: Exception) -> None:
        messagebox.showerror("Stack", str(err))
        self._status.configure(text="Error (see dialog).")
        self._img_label.configure(image="", text=f"Error: {err}")
        self._probe_clear()

    def _apply_result(
        self,
        res: object,
        prep: object | None = None,
        *,
        from_cache: bool = False,
    ) -> None:
        self._stop_play()
        if prep is not None:
            self._fullhill_prep = prep
        self._result = res
        n = int(res.imb.shape[2])
        self._slider.configure(from_=0, to=max(0, n - 1))
        self._idx_var.set(0.0)
        et = getattr(res, "exptime_main", [])
        sc_txt = ""
        if et:
            sc_txt = "  |  EXPTIME: " + ", ".join(f"{float(x):.4g} s" for x in et[: min(6, len(et))])
            if len(et) > 6:
                sc_txt += "…"
        if from_cache:
            msg = f"Done: {n} plane(s).{sc_txt}"
        else:
            msg = (
                f"Done: {n} plane(s).  xs[0]=ys[0]=0; shifts from short centroids."
                f"  Re-run applies new satellite params only.{sc_txt}"
            )
        self._status.configure(text=msg)
        self._refresh_view()

    def _on_slider(self, _val: str) -> None:
        self._refresh_view()

    def _probe_image_xy(self, event: object) -> tuple[int, int] | None:
        """Map widget coordinates to image column ``ix``, row ``iy`` (``PhotoImage`` pixels)."""
        if self._photo is None:
            return None
        iw = self._photo.width()
        ih = self._photo.height()
        if iw < 1 or ih < 1:
            return None
        try:
            lw = self._img_label.winfo_width()
            lh = self._img_label.winfo_height()
        except tk.TclError:
            return None
        if lw < 2 or lh < 2:
            ox = oy = 0
        else:
            ox = max(0, (lw - iw) // 2)
            oy = max(0, (lh - ih) // 2)
        wx = float(event.x) - ox
        wy = float(event.y) - oy
        ix = int(np.clip(np.floor(wx), 0, iw - 1))
        iy = int(np.clip(np.floor(wy), 0, ih - 1))
        return ix, iy

    def _probe_clear(self) -> None:
        try:
            self._probe_lbl.configure(text="")
        except tk.TclError:
            pass

    def _on_probe_leave(self, _event: object) -> None:
        self._probe_clear()

    def _on_probe_motion(self, event: object) -> None:
        from fullhill import (
            IMZ_CONGRID_INPUT_TARGET_CX,
            IMZ_CONGRID_INPUT_TARGET_CY,
            imz_congrid_out_side,
            km_distance_to_target,
        )

        if self._result is None or self._photo is None or self._fullhill_prep is None:
            return
        xy = self._probe_image_xy(event)
        if xy is None:
            self._probe_clear()
            return
        ix, iy = xy
        name = self._stack_var.get()
        cube = getattr(self._result, name, None)
        if cube is None:
            return
        prep = self._fullhill_prep
        rr = prep.rr
        n = int(rr.shape[0])
        kpp = prep.kpp
        tcx_imb = float(prep.djx)
        tcy_imb = float(prep.djy)

        if cube.ndim == 2:
            val = float(cube[iy, ix])
            k_geo = n - 1
            sz_zoom = imz_congrid_out_side(rr, k_geo)
            dist = km_distance_to_target(
                float(ix),
                float(iy),
                target_cx=IMZ_CONGRID_INPUT_TARGET_CX,
                target_cy=IMZ_CONGRID_INPUT_TARGET_CY,
                kpp_km=float(kpp[k_geo]),
                sz_zoom=sz_zoom,
            )
        else:
            k = int(round(float(self._idx_var.get())))
            k = max(0, min(k, cube.shape[2] - 1))
            val = float(cube[iy, ix, k])
            if name in ("imb", "imbs"):
                dist = km_distance_to_target(
                    float(ix),
                    float(iy),
                    target_cx=tcx_imb,
                    target_cy=tcy_imb,
                    kpp_km=float(kpp[k]),
                    sz_zoom=None,
                )
            elif name in ("imz", "imzs"):
                sz_zoom = imz_congrid_out_side(rr, k)
                dist = km_distance_to_target(
                    float(ix),
                    float(iy),
                    target_cx=IMZ_CONGRID_INPUT_TARGET_CX,
                    target_cy=IMZ_CONGRID_INPUT_TARGET_CY,
                    kpp_km=float(kpp[k]),
                    sz_zoom=sz_zoom,
                )
            else:
                dist = None

        if dist is None:
            dist_s = "—"
        else:
            dist_s = f"{dist:.3f} km"
        txt = f"x={ix}  y={iy}  value={val:.8g}  dist={dist_s}"
        try:
            self._probe_lbl.configure(text=txt)
        except tk.TclError:
            pass

    def _refresh_detail(self, k: int) -> None:
        if self._result is None:
            self._detail.configure(text="")
            return
        r = self._result
        n = int(r.imb.shape[2])
        stack = self._stack_var.get()
        if stack in self._STACK_MEDIAN:
            label = "median(imz)" if stack == "median_imz" else "median(imzs)"
            self._detail.configure(
                text=f"{label}  |  per-pixel median of {n} plane(s)  |  no plane slider"
            )
            self._plane_lbl.configure(text="—")
            return
        k = max(0, min(k, n - 1))
        parts = [
            f"plane {k + 1}/{n}",
            f"xs={_format_plane_shift_px(float(r.xs[k]))} px",
            f"ys={_format_plane_shift_px(float(r.ys[k]))} px",
        ]
        if k < len(r.short_paths):
            parts.append(f"file: {display_fits_name(r.short_paths[k])}")
        if k < len(r.exptime_main):
            parts.append(f"EXPTIME={float(r.exptime_main[k]):.6g} s")
        if hasattr(r, "hours_from_ca") and k < len(r.hours_from_ca):
            parts.append(f"Δt CA (h): {float(r.hours_from_ca[k]):+.2f}")
        self._detail.configure(text="  |  ".join(parts))
        self._plane_lbl.configure(text=f"{k + 1} / {n}")

    def _refresh_view(self) -> None:
        if not self._img_label.winfo_exists():
            return
        if self._result is None:
            return
        k_idx = int(round(float(self._idx_var.get())))
        self._refresh_detail(k_idx)
        name = self._stack_var.get()
        cube = getattr(self._result, name, None)
        if cube is None:
            return
        is_median = name in self._STACK_MEDIAN
        try:
            self._slider.config(state=tk.DISABLED if is_median else tk.NORMAL)
            self._play_btn.config(state=tk.DISABLED if is_median else tk.NORMAL)
        except tk.TclError:
            pass
        if cube.ndim == 2:
            plane = np.asarray(cube, dtype=np.float64)
        else:
            k = int(round(float(self._idx_var.get())))
            k = max(0, min(k, cube.shape[2] - 1))
            plane = np.asarray(cube[:, :, k], dtype=np.float64)
        _, _, vmin, vmax = sky_scale(plane)
        pil = data_to_thumbnail_u8(plane, vmin, vmax, FULL_HILL_SIZE)
        self._photo = ImageTk.PhotoImage(pil)
        self._img_label.configure(image=self._photo, text="")
        self._img_label.image = self._photo


class StarsWindow(tk.Toplevel):
    """Stars analysis: precompute all planes in parallel, cache in memory; slider is instant like Stack."""

    def __init__(
        self,
        master: tk.Tk,
        selected_paths: list[Path],
        *,
        quit_root_when_destroyed: bool = False,
    ) -> None:
        super().__init__(master)
        self.title("Stars")
        self._paths = list(selected_paths)
        n = len(self._paths)
        self._quit_root_when_destroyed = quit_root_when_destroyed
        # Not named ``_root``: that shadows ``Misc._root()`` and breaks Tk event dispatch.
        self._master_ref = master
        # If the root is withdrawn, ``transient`` can prevent the window from mapping (Linux WMs).
        try:
            if int(master.winfo_viewable()) != 0:
                self.transient(master)
        except tk.TclError:
            pass
        self._photo: ImageTk.PhotoImage | None = None
        self._native_data: np.ndarray | None = None
        self._cache_fill_gen = 0
        self._plane_cache: list[object | None] = []
        self._tk_photos: list[ImageTk.PhotoImage | None] = []
        self._align_step: str | None = None
        self._align_pred_native: tuple[float, float] | None = None
        self._define_center_waiting = False
        # Per-plane: user-marked target (native px); None → centroid for that plane in the pipeline.
        self._plane_target_center: list[tuple[float, float] | None] = [None] * n
        # Per-plane manual refcat tweak after **Align** (additive to djx−xpred term).
        self._plane_astro_shift: list[tuple[float, float]] = [(0.0, 0.0) for _ in range(n)]

        self.geometry("1100x1000")
        self.minsize(640, 480)

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Close", command=self.destroy).pack(side=tk.RIGHT)

        ctrl = ttk.Frame(self, padding=(8, 0))
        ctrl.pack(fill=tk.X)
        ttk.Label(ctrl, text="Max r mag:").pack(side=tk.LEFT)
        self._mag_max_var = tk.StringVar(value="15")
        ttk.Entry(ctrl, textvariable=self._mag_max_var, width=6).pack(side=tk.LEFT, padx=(4, 16))
        _lbl_d = ttk.Label(ctrl, text="diam (m):")
        _lbl_d.pack(side=tk.LEFT)
        self._diam_var = tk.StringVar(value="2.5")
        self._diam_entry = ttk.Entry(ctrl, textvariable=self._diam_var, width=6)
        self._diam_entry.pack(side=tk.LEFT, padx=(4, 8))
        _lbl_a = ttk.Label(ctrl, text="albedo:")
        _lbl_a.pack(side=tk.LEFT)
        self._alb_var = tk.StringVar(value="0.41")
        self._alb_entry = ttk.Entry(ctrl, textvariable=self._alb_var, width=6)
        self._alb_entry.pack(side=tk.LEFT, padx=(4, 8))
        _lbl_sd = ttk.Label(ctrl, text="sat dist (km):")
        _lbl_sd.pack(side=tk.LEFT)
        self._satdist_var = tk.StringVar(value="200")
        self._satdist_entry = ttk.Entry(ctrl, textvariable=self._satdist_var, width=8)
        self._satdist_entry.pack(side=tk.LEFT, padx=(4, 8))
        _lbl_sa = ttk.Label(ctrl, text="sat ang (deg):")
        _lbl_sa.pack(side=tk.LEFT)
        self._satang_var = tk.StringVar(value="0")
        self._satang_entry = ttk.Entry(ctrl, textvariable=self._satang_var, width=6)
        self._satang_entry.pack(side=tk.LEFT, padx=(4, 8))
        self._fake_sat_entries = (
            self._diam_entry,
            self._alb_entry,
            self._satdist_entry,
            self._satang_entry,
        )
        self._fake_sat_labels = (_lbl_d, _lbl_a, _lbl_sd, _lbl_sa)
        ttk.Button(ctrl, text="Refresh", command=self._reload_current).pack(side=tk.LEFT, padx=(12, 0))
        self._align_btn = ttk.Button(ctrl, text="Align", command=self._start_align)
        self._align_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(ctrl, text="Define center", command=self._start_define_center).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # Same plane slider layout as ``FullHillWindow`` (classic ``tk.Scale``).
        slide_row = ttk.Frame(self, padding=8)
        slide_row.pack(fill=tk.X)
        slide_row.columnconfigure(4, weight=1)
        slide_row.columnconfigure(1, weight=0)
        ttk.Label(slide_row, text="Plane:").grid(row=0, column=0, sticky="w")
        self.update_idletasks()
        slider_w = max(120, self.winfo_screenwidth() // 3)
        self._idx_var = tk.DoubleVar(value=0.0)
        self._slider = tk.Scale(
            slide_row,
            from_=0,
            to=max(0, n - 1),
            orient=tk.HORIZONTAL,
            length=slider_w,
            resolution=1,
            showvalue=0,
            variable=self._idx_var,
            command=self._on_slider,
        )
        self._slider.grid(row=0, column=1, padx=(8, 4), sticky="w")
        self._play_active = False
        self._play_after: str | int | None = None
        self._play_btn = ttk.Button(slide_row, text="Play", command=self._toggle_play)
        self._play_btn.grid(row=0, column=2, padx=(4, 8), sticky="w")
        self._plane_lbl = ttk.Label(slide_row, text=f"1 / {n}")
        self._plane_lbl.grid(row=0, column=3, sticky="w")
        ttk.Frame(slide_row).grid(row=0, column=4, sticky="nsew")
        if n <= 1:
            self._slider.configure(state=tk.DISABLED)
            self._play_btn.configure(state=tk.DISABLED)

        self._status = ttk.Label(
            self,
            text=f"{n} file(s)  |  loading…",
            wraplength=920,
        )
        self._status.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._detail = ttk.Label(self, text="", wraplength=920, justify=tk.LEFT)
        self._detail.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._probe_lbl = ttk.Label(
            self,
            text="",
            wraplength=920,
            justify=tk.LEFT,
            font=("TkFixedFont",),
        )
        self._probe_lbl.pack(fill=tk.X, padx=8, pady=(0, 2))

        img_frame = ttk.Frame(self, padding=8)
        img_frame.pack(fill=tk.BOTH, expand=True)
        self._img_label = ttk.Label(
            img_frame,
            text="Computing stars planes…",
            anchor=tk.CENTER,
        )
        self._img_label.pack()
        self._img_label.bind("<Motion>", self._on_probe_motion)
        self._img_label.bind("<Leave>", self._on_probe_leave)
        self._img_label.bind("<Button-1>", self._on_stars_image_b1)

        self.bind("<Escape>", self._on_escape_stars)
        self.bind("<FocusIn>", lambda _e: self._sync_fake_sat_controls_state())
        self.bind("<Map>", lambda _e: self._sync_fake_sat_controls_state())
        self.after_idle(self._sync_fake_sat_controls_state)
        self.after_idle(self._reload_current)

    def destroy(self) -> None:
        self._stop_play()
        self._define_center_waiting = False
        self._align_step = None
        self._align_pred_native = None
        try:
            self._img_label.unbind("<Button-1>")
        except tk.TclError:
            pass
        super().destroy()
        if self._quit_root_when_destroyed:
            try:
                self._master_ref.destroy()
            except tk.TclError:
                pass

    def _target_center_for_plane(self, k: int) -> tuple[float, float] | None:
        """Native (x, y) target center for this plane; ``None`` → use brightness centroid."""
        o = self._plane_target_center
        if o is None or k < 0 or k >= len(o):
            return None
        return o[k]

    def _on_escape_stars(self, event: object | None = None) -> None:
        self._cancel_define_center()
        self._cancel_align()

    def _on_stars_image_b1(self, event: object) -> None:
        if self._define_center_waiting:
            self._on_define_center_b1(event)
            return
        if self._align_step is not None:
            self._on_align_b1(event)
            return
        self._on_sky_box_report_stars(event)

    def _on_sky_box_report_stars(self, event: object) -> None:
        """Left-click (idle): 30×30 local median / RMS, 5σ, equivalent fakesat diameter."""
        if self._native_data is None or self._photo is None:
            return
        xy = self._probe_image_xy(event)
        if xy is None:
            return
        try:
            alb = float(self._alb_var.get().strip())
        except ValueError:
            messagebox.showerror("Stars", "Invalid albedo.")
            return
        from fullhill import diameter_for_fakesat_total_counts
        from lucy_spice import stars_ephemeris_bundle

        ix, iy = xy
        plane = np.asarray(self._native_data, dtype=np.float64)
        h, w = int(plane.shape[0]), int(plane.shape[1])
        iw = self._photo.width()
        ih = self._photo.height()
        nc, nr = _display_to_native_xy(ix, iy, iw, ih, w, h)
        r0, r1, c0, c1 = _clipped_square_box(nr, nc, h, w, SKY_REPORT_BOX)
        patch = plane[r0:r1, c0:c1]
        med, rms, n_pix = _patch_median_rms_npix(patch)
        five_pix = 5.0 * rms if np.isfinite(rms) else float("nan")
        flux_5 = sky_box_five_sigma_integrated(rms, n_pix)

        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        k = max(0, min(k, len(self._paths) - 1))
        path = self._paths[k]
        try:
            range_km, phase_deg, delta_km, _xp, _yp = stars_ephemeris_bundle(path)
        except Exception as e:
            messagebox.showinfo(
                "Stars",
                format_sky_box_report(
                    med,
                    rms,
                    five_pix,
                    flux_5,
                    n_pix,
                    r0,
                    r1,
                    c0,
                    c1,
                    None,
                    title_note="Stars",
                    error=str(e),
                ),
            )
            return

        et = primary_exptime_seconds(path)
        et_s = float(et) if et is not None and et > 0 else 1.0
        diam = diameter_for_fakesat_total_counts(
            flux_5, alb, range_km, delta_km, phase_deg, et_s
        )
        d_s = f"{diam:.4g} m" if diam is not None and math.isfinite(diam) else "—"
        messagebox.showinfo(
            "Stars",
            format_sky_box_report(
                med,
                rms,
                five_pix,
                flux_5,
                n_pix,
                r0,
                r1,
                c0,
                c1,
                d_s,
                title_note="Stars",
            ),
        )

    def _start_define_center(self) -> None:
        if self._define_center_waiting:
            self._cancel_define_center()
            return
        self._stop_play()
        self._cancel_align()
        self._define_center_waiting = True
        self._status.configure(
            text="Define center: click the target on **this plane** only — sets target position and "
            "astrometry vs ephemeris for this plane, then redraws. Esc cancels.",
        )

    def _cancel_define_center(self, _event: object | None = None) -> None:
        if not self._define_center_waiting:
            return
        self._define_center_waiting = False
        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        self._show_plane(max(0, min(k, len(self._paths) - 1)))

    def _on_define_center_b1(self, event: object) -> None:
        if not self._define_center_waiting:
            return
        xy = self._probe_image_xy(event)
        if xy is None:
            return
        ix, iy = xy
        xy_native = self._display_xy_to_native_float(ix, iy)
        if xy_native is None:
            return
        nx, ny = xy_native
        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        n = len(self._paths)
        k = max(0, min(k, n - 1))

        if self._plane_target_center is None or len(self._plane_target_center) != n:
            self._plane_target_center = [None] * n
        if len(self._plane_astro_shift) != n:
            self._plane_astro_shift = [(0.0, 0.0) for _ in range(n)]
        self._plane_target_center[k] = (nx, ny)
        self._plane_astro_shift[k] = (0.0, 0.0)

        self._define_center_waiting = False
        self._start_cache_fill()

    def _sync_fake_sat_controls_state(self) -> None:
        """Enable diam/albedo/sat offset fields only when **Define PSF** has set a session PSF."""
        from stars_analysis import stars_psf_source

        ok = stars_psf_source() is not None
        st = tk.NORMAL if ok else tk.DISABLED
        for w in self._fake_sat_entries:
            try:
                w.configure(state=st)
            except tk.TclError:
                pass
        fg = "" if ok else "gray50"
        for lb in self._fake_sat_labels:
            try:
                lb.configure(foreground=fg)
            except tk.TclError:
                pass

    def _parse_params(self) -> tuple[float, float, float, float, float]:
        from stars_analysis import stars_psf_source

        try:
            mag_max = float(self._mag_max_var.get().strip())
        except ValueError:
            mag_max = 15.0
        try:
            diam = float(self._diam_var.get().strip())
        except ValueError:
            diam = 2.5
        try:
            alb = float(self._alb_var.get().strip())
        except ValueError:
            alb = 0.41
        try:
            sd = float(self._satdist_var.get().strip())
        except ValueError:
            sd = 200.0
        try:
            sa = float(self._satang_var.get().strip())
        except ValueError:
            sa = 0.0
        if stars_psf_source() is None:
            diam = 0.0
        return diam, alb, sd, sa, mag_max

    def _display_xy_to_native_float(self, ix: int, iy: int) -> tuple[float, float] | None:
        d = self._native_data
        if d is None or self._photo is None:
            return None
        h, w = int(d.shape[0]), int(d.shape[1])
        iw = self._photo.width()
        ih = self._photo.height()
        if iw < 1 or ih < 1:
            return None
        fx = ix * (w - 1) / max(iw - 1, 1)
        fy = iy * (h - 1) / max(ih - 1, 1)
        return float(fx), float(fy)

    @staticmethod
    def _nearest_star_native(
        nx: float, ny: float, stars: list[tuple[float, float]]
    ) -> tuple[float, float] | None:
        if not stars:
            return None
        best = stars[0]
        best_d = 1e30
        for sx, sy in stars:
            d = (sx - nx) ** 2 + (sy - ny) ** 2
            if d < best_d:
                best_d = d
                best = (float(sx), float(sy))
        return best

    @staticmethod
    def _peak_near_native(
        data: np.ndarray, nx: float, ny: float, *, radius: int = 20
    ) -> tuple[float, float]:
        h, w = int(data.shape[0]), int(data.shape[1])
        ic = int(round(nx))
        jc = int(round(ny))
        x0 = max(0, ic - radius)
        x1 = min(w, ic + radius + 1)
        y0 = max(0, jc - radius)
        y1 = min(h, jc + radius + 1)
        sub = np.asarray(data[y0:y1, x0:x1], dtype=np.float64)
        if sub.size == 0:
            return float(nx), float(ny)
        flat = int(np.argmax(sub))
        py, px = np.unravel_index(flat, sub.shape)
        return float(x0 + px), float(y0 + py)

    def _start_align(self) -> None:
        if self._align_step is not None:
            self._cancel_align()
            return
        self._stop_play()
        self._cancel_define_center()
        from stars_analysis import StarsPlaneResult

        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        k = max(0, min(k, len(self._paths) - 1))
        ent = self._plane_cache[k] if k < len(self._plane_cache) else None
        if not isinstance(ent, StarsPlaneResult) or len(ent.star_xy_native) == 0:
            messagebox.showinfo(
                "Align",
                "Need a loaded plane with refcat stars (wait until the cache is ready), then try again.",
            )
            return
        self._align_step = "pred"
        self._align_pred_native = None
        self._status.configure(
            text="Align: (1) Click inside a red circle (predicted star). "
            "(2) Then click near the true star — Esc cancels.",
        )

    def _cancel_align(self, _event: object | None = None) -> None:
        if self._align_step is None:
            return
        self._align_step = None
        self._align_pred_native = None
        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        self._show_plane(max(0, min(k, len(self._paths) - 1)))

    def _on_align_b1(self, event: object) -> None:
        if self._align_step is None:
            return
        from stars_analysis import StarsPlaneResult

        xy = self._probe_image_xy(event)
        if xy is None:
            return
        ix, iy = xy
        try:
            k = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k = 0
        k = max(0, min(k, len(self._paths) - 1))
        ent = self._plane_cache[k] if k < len(self._plane_cache) else None
        if not isinstance(ent, StarsPlaneResult):
            return
        if self._align_step == "pred":
            xy_native = self._display_xy_to_native_float(ix, iy)
            if xy_native is None:
                return
            nx, ny = xy_native
            nearest = self._nearest_star_native(nx, ny, ent.star_xy_native)
            if nearest is None:
                return
            self._align_pred_native = nearest
            self._align_step = "true"
            self._status.configure(
                text="Align: (2) Click near the true star (max within 20 px). Esc cancels.",
            )
            return
        if self._align_step == "true":
            if self._align_pred_native is None:
                self._cancel_align()
                return
            xy_native = self._display_xy_to_native_float(ix, iy)
            if xy_native is None:
                return
            nx, ny = xy_native
            if self._native_data is None:
                return
            tx, ty = self._peak_near_native(self._native_data, nx, ny, radius=20)
            px, py = self._align_pred_native
            dx = tx - px
            dy = ty - py
            ox, oy = self._plane_astro_shift[k]
            self._plane_astro_shift[k] = (ox + dx, oy + dy)
            self._align_step = None
            self._align_pred_native = None
            self._start_cache_fill()

    def _reload_current(self) -> None:
        self._start_cache_fill()

    def _start_cache_fill(self) -> None:
        self._stop_play()
        self._sync_fake_sat_controls_state()
        self._cache_fill_gen += 1
        fill_gen = self._cache_fill_gen
        n = len(self._paths)
        self._plane_cache = [None] * n
        self._tk_photos = [None] * n
        params = self._parse_params()
        self._status.configure(
            text=f"{n} file(s)  |  computing all {n} plane(s) in parallel (cached for slider)…",
        )
        try:
            self._img_label.configure(image="", text="Computing stars planes…")
        except tk.TclError:
            pass
        paths = list(self._paths)
        astro_shifts = list(self._plane_astro_shift)
        target_centers = tuple(self._target_center_for_plane(i) for i in range(n))
        try:
            cur_idx = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            cur_idx = 0
        cur_idx = max(0, min(cur_idx, n - 1))
        self._show_plane(cur_idx)

        def worker() -> None:
            from stars_analysis import StarsPlaneResult, run_stars_plane

            diam, alb, sd, sa, mag_max = params

            def compute_one(ii: int, p: Path) -> tuple[int, StarsPlaneResult | None, BaseException | None]:
                try:
                    tc = target_centers[ii]
                    adx, ady = astro_shifts[ii]
                    res = run_stars_plane(
                        p,
                        diam_m=diam,
                        albedo=alb,
                        satdist_km=sd,
                        satang_deg=sa,
                        mag_max=mag_max,
                        mag_min=5.0,
                        astrometry_dx=adx,
                        astrometry_dy=ady,
                        target_center_native=tc,
                    )
                    return ii, res, None
                except BaseException as e:
                    return ii, None, e

            nw = min(8, max(1, len(paths)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
                futs = [ex.submit(compute_one, i, p) for i, p in enumerate(paths)]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        ii, res, err = fut.result()
                    except Exception:
                        continue
                    if err is not None:
                        self.after(
                            0,
                            lambda i=ii, er=err, g=fill_gen: self._on_plane_failed(i, er, g),
                        )
                    elif res is not None:
                        self.after(
                            0,
                            lambda i=ii, r=res, g=fill_gen: self._on_plane_ready(i, r, g),
                        )

        threading.Thread(target=worker, daemon=True).start()

    def _on_plane_failed(self, i: int, err: BaseException, fill_gen: int) -> None:
        if fill_gen != self._cache_fill_gen:
            return
        self._plane_cache[i] = err
        self._tk_photos[i] = None
        self._update_status_counts()
        try:
            cur = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            cur = 0
        n = len(self._paths)
        cur = max(0, min(cur, n - 1))
        # Always refresh the visible plane: cur==i can miss updates if the slider var
        # was out of sync, leaving the label stuck on "Loading" with no overlays.
        self._show_plane(cur)

    def _on_plane_ready(self, i: int, res: object, fill_gen: int) -> None:
        if fill_gen != self._cache_fill_gen:
            return
        from stars_analysis import StarsPlaneResult

        if not isinstance(res, StarsPlaneResult):
            return
        self._plane_cache[i] = res
        # master=self keeps PhotoImage tied to this Toplevel; .copy() avoids stale buffers.
        self._tk_photos[i] = ImageTk.PhotoImage(res.pil_image.copy(), master=self)
        self._update_status_counts()
        try:
            cur = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            cur = 0
        n = len(self._paths)
        cur = max(0, min(cur, n - 1))
        self._show_plane(cur)

    def _update_status_counts(self) -> None:
        from stars_analysis import StarsPlaneResult

        n = len(self._paths)
        ok = sum(1 for x in self._plane_cache if isinstance(x, StarsPlaneResult))
        bad = sum(1 for x in self._plane_cache if isinstance(x, BaseException))
        pending = n - ok - bad
        self._status.configure(
            text=(
                f"{n} file(s)  |  cache: {ok} ok, {bad} err, {pending} pending  "
                f"(slider uses cache; instant once ready)"
            ),
        )

    def _stop_play(self) -> None:
        if self._play_after is not None:
            try:
                self.after_cancel(self._play_after)
            except tk.TclError:
                pass
            self._play_after = None
        self._play_active = False
        try:
            if self._play_btn.winfo_exists():
                self._play_btn.configure(text="Play")
        except tk.TclError:
            pass

    def _toggle_play(self) -> None:
        if self._play_active:
            self._stop_play()
            return
        n = len(self._paths)
        if n <= 1:
            return
        self._play_active = True
        try:
            self._play_btn.configure(text="Stop")
        except tk.TclError:
            pass
        self._play_after = self.after(400, self._play_step)

    def _play_step(self) -> None:
        self._play_after = None
        if not self._play_active:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        n = len(self._paths)
        if n <= 1:
            self._stop_play()
            return
        k = int(round(float(self._idx_var.get())))
        k = (k + 1) % n
        self._idx_var.set(float(k))
        self._plane_lbl.configure(text=f"{k + 1} / {n}")
        self._show_plane(k)
        if self._play_active:
            self._play_after = self.after(400, self._play_step)

    def _on_slider(self, val: str) -> None:
        try:
            k = int(round(float(val)))
        except (ValueError, tk.TclError, TypeError):
            return
        k = max(0, min(k, len(self._paths) - 1))
        self._idx_var.set(float(k))
        self._plane_lbl.configure(text=f"{k + 1} / {len(self._paths)}")
        self._show_plane(k)

    def _show_plane(self, k: int) -> None:
        from stars_analysis import StarsPlaneResult

        k = max(0, min(int(k), len(self._paths) - 1))
        self._idx_var.set(float(k))
        self._plane_lbl.configure(text=f"{k + 1} / {len(self._paths)}")
        ent = self._plane_cache[k] if k < len(self._plane_cache) else None
        ph = self._tk_photos[k] if k < len(self._tk_photos) else None

        if isinstance(ent, StarsPlaneResult) and ph is not None:
            self._native_data = ent.native_display
            self._photo = ph
            self._img_label.configure(image=ph, text="")
            self._img_label.image = ph
            # Keep ``_status`` empty here: a long wrapped line shifts the image vertically.
            self._status.configure(text="")
            self._detail.configure(text=display_fits_name(ent.path))
        elif isinstance(ent, BaseException):
            self._native_data = None
            self._photo = None
            self._img_label.configure(image="", text=f"Error (plane {k + 1}): {ent}")
            self._detail.configure(text="")
            self._update_status_counts()
        else:
            self._native_data = None
            self._photo = None
            self._img_label.configure(
                image="",
                text=f"Loading plane {k + 1}… (parallel cache still running)",
            )
            self._detail.configure(text="")

    def _fail(self, err: Exception) -> None:
        messagebox.showerror("Stars", str(err))
        self._status.configure(text="Error (see dialog).")
        self._img_label.configure(image="", text=f"Error: {err}")
        self._probe_clear()

    def _probe_image_xy(self, event: object) -> tuple[int, int] | None:
        """Map widget coordinates to image column ``ix``, row ``iy`` (PhotoImage pixels)."""
        if self._photo is None:
            return None
        iw = self._photo.width()
        ih = self._photo.height()
        if iw < 1 or ih < 1:
            return None
        try:
            lw = self._img_label.winfo_width()
            lh = self._img_label.winfo_height()
        except tk.TclError:
            return None
        if lw < 2 or lh < 2:
            ox = oy = 0
        else:
            ox = max(0, (lw - iw) // 2)
            oy = max(0, (lh - ih) // 2)
        wx = float(event.x) - ox
        wy = float(event.y) - oy
        ix = int(np.clip(np.floor(wx), 0, iw - 1))
        iy = int(np.clip(np.floor(wy), 0, ih - 1))
        return ix, iy

    def _probe_clear(self) -> None:
        try:
            self._probe_lbl.configure(text="")
        except tk.TclError:
            pass

    def _on_probe_leave(self, _event: object) -> None:
        self._probe_clear()

    def _native_indices_from_display(self, ix: int, iy: int) -> tuple[int, int]:
        """Map displayed thumbnail pixel to nearest native array indices."""
        d = self._native_data
        if d is None:
            return 0, 0
        h, w = int(d.shape[0]), int(d.shape[1])
        iw = self._photo.width()
        ih = self._photo.height()
        if iw < 1 or ih < 1:
            return 0, 0
        fx = ix * (w - 1) / max(iw - 1, 1)
        fy = iy * (h - 1) / max(ih - 1, 1)
        nx = int(np.clip(np.round(fx), 0, w - 1))
        ny = int(np.clip(np.round(fy), 0, h - 1))
        return nx, ny

    def _on_probe_motion(self, event: object) -> None:
        if self._native_data is None or self._photo is None:
            return
        from fullhill import km_distance_to_target
        from stars_analysis import StarsPlaneResult

        xy = self._probe_image_xy(event)
        if xy is None:
            self._probe_clear()
            return
        ix, iy = xy
        nx_i, ny_i = self._native_indices_from_display(ix, iy)
        val = float(self._native_data[ny_i, nx_i])
        if not np.isfinite(val):
            val_s = "nan"
        else:
            val_s = f"{val:.8g}"

        dist_s = "—"
        try:
            k_pl = int(round(float(self._idx_var.get())))
        except (ValueError, tk.TclError):
            k_pl = 0
        k_pl = max(0, min(k_pl, len(self._paths) - 1))
        ent = self._plane_cache[k_pl] if k_pl < len(self._plane_cache) else None
        xy_nat = self._display_xy_to_native_float(ix, iy)
        if (
            isinstance(ent, StarsPlaneResult)
            and xy_nat is not None
            and ent.kpp_km > 0
        ):
            nx, ny = xy_nat
            dist = km_distance_to_target(
                nx,
                ny,
                target_cx=ent.target_cx_native,
                target_cy=ent.target_cy_native,
                kpp_km=ent.kpp_km,
                sz_zoom=None,
            )
            if dist is not None:
                dist_s = f"{dist:.3f} km"

        txt = f"x={ix}  y={iy}  value={val_s}  dist={dist_s}"
        try:
            self._probe_lbl.configure(text=txt)
        except tk.TclError:
            pass


class FitsThumbViewer(tk.Tk):
    def __init__(self, *, min_exp_default: str = "0") -> None:
        super().__init__()
        self.title("FITS thumbnails (Lucy LORRI)")
        self.geometry("1100x720")
        self._dir_var = tk.StringVar(value=str(default_root_dir()))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
        self._pending: set[concurrent.futures.Future] = set()
        self._files: list[Path] = []
        self._file_rows: list[list[Path]] = []
        self._row_group_ids: list[int] = []
        self._groups_paths: list[list[Path]] = []
        self._group_selection: dict[int, tk.BooleanVar] = {}
        self._num_rows = 0
        self._materialized_rows: dict[int, ttk.Frame] = {}
        self._row_canvas_ids: dict[int, int] = {}
        self._row_futures: dict[int, list[concurrent.futures.Future]] = {}
        self._placeholder_item: int | None = None
        self._file_selection: dict[Path, tk.BooleanVar] = {}
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Directory:").grid(row=0, column=0, sticky="nw", pady=(2, 0))
        self._entry = ttk.Entry(top, textvariable=self._dir_var, width=72)
        self._entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        browse_col = ttk.Frame(top)
        browse_col.grid(row=0, column=2, sticky="nw")
        btn_row = ttk.Frame(browse_col)
        btn_row.pack(anchor=tk.W)
        ttk.Button(btn_row, text="Browse…", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Load", command=self._load).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(btn_row, text="Clear all selections", command=self._clear_all_selections).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(btn_row, text="Quit", command=self._on_close).pack(side=tk.LEFT, padx=(4, 0))
        browse_actions = ttk.Frame(browse_col)
        browse_actions.pack(anchor=tk.W, pady=(6, 0))
        self._stack_btn = ttk.Button(browse_actions, text="Stack", command=self._open_full_hill)
        self._stack_btn.pack(side=tk.LEFT)
        self._stars_btn = ttk.Button(browse_actions, text="Stars", command=self._open_stars)
        self._stars_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._define_psf_btn = ttk.Button(browse_actions, text="Define PSF", command=self._define_psf)
        self._define_psf_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._clear_psf_btn = ttk.Button(browse_actions, text="Clear PSF", command=self._clear_psf)
        self._clear_psf_btn.pack(side=tk.LEFT, padx=(6, 0))
        psf_status_row = ttk.Frame(browse_col)
        psf_status_row.pack(anchor=tk.W, pady=(4, 0))
        self._psf_status_var = tk.StringVar(value="Stars PSF: per image")
        ttk.Label(psf_status_row, textvariable=self._psf_status_var).pack(side=tk.LEFT)

        hill_bar = ttk.Frame(self, padding=(8, 0, 8, 8))
        hill_bar.pack(fill=tk.X)
        hill_row0 = ttk.Frame(hill_bar)
        hill_row0.pack(fill=tk.X)
        ttk.Label(hill_row0, text="Hill sphere radius (km):").pack(side=tk.LEFT)
        self._hill_km_var = tk.StringVar(value=str(DEFAULT_HILL_KM))
        ttk.Entry(hill_row0, textvariable=self._hill_km_var, width=12).pack(side=tk.LEFT, padx=(6, 0))

        sap_row = ttk.Frame(hill_bar)
        sap_row.pack(fill=tk.X, pady=(6, 0))
        self._sap_exclude_var = tk.StringVar(value="")
        self._sap_include_var = tk.StringVar(value="")
        ttk.Label(sap_row, text="Filter out:").grid(row=0, column=0, sticky="w")
        ex_ent = ttk.Entry(sap_row, textvariable=self._sap_exclude_var)
        ex_ent.grid(row=0, column=1, sticky="ew", padx=(6, 16))
        ttk.Label(sap_row, text="Only include:").grid(row=0, column=2, sticky="w")
        in_ent = ttk.Entry(sap_row, textvariable=self._sap_include_var)
        in_ent.grid(row=0, column=3, sticky="ew", padx=(6, 0))
        sap_row.columnconfigure(1, weight=1)
        sap_row.columnconfigure(3, weight=1)
        for w in (ex_ent, in_ent):
            w.bind("<Return>", lambda _e: self._load())

        hill_row1 = ttk.Frame(hill_bar)
        hill_row1.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(hill_row1, text="Min EXPTIME (s):").pack(side=tk.LEFT)
        self._min_exp_var = tk.StringVar(value=min_exp_default)
        ttk.Entry(hill_row1, textvariable=self._min_exp_var, width=8).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(hill_row1, text="Max EXPTIME (s):").pack(side=tk.LEFT, padx=(16, 0))
        self._max_exp_var = tk.StringVar(value="60")
        ttk.Entry(hill_row1, textvariable=self._max_exp_var, width=8).pack(side=tk.LEFT, padx=(6, 0))
        self._one_x_one_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(hill_row1, text="1x1 only", variable=self._one_x_one_var).pack(side=tk.LEFT, padx=(16, 0))
        self._groups_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            hill_row1,
            text="Groups",
            variable=self._groups_var,
            command=self._load,
        ).pack(side=tk.LEFT, padx=(12, 0))

        # Scrollable grid
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._canvas = tk.Canvas(outer, highlightthickness=0)
        self._vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self._scrollbar_cmd)
        self._canvas.configure(yscrollcommand=self._on_yscroll_set)
        self._vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Rows are separate canvas.create_window items (tiny windows). One huge embedded frame caused X11 BadAlloc.
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.bind_all("<Button-5>", self._on_mousewheel_linux)

        self._load()

    def _on_close(self) -> None:
        for f in self._pending:
            f.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        w = max(event.width, 1)
        for wid in self._row_canvas_ids.values():
            self._canvas.itemconfigure(wid, width=w)
        if self._placeholder_item is not None:
            self._canvas.itemconfigure(self._placeholder_item, width=w)
        self._update_scrollregion()
        self.after_idle(self._sync_visible_rows)

    def _on_yscroll_set(self, first: str, last: str) -> None:
        self._vsb.set(first, last)
        self.after_idle(self._sync_visible_rows)

    def _scrollbar_cmd(self, *args: object) -> None:
        self._canvas.yview(*args)
        self.after_idle(self._sync_visible_rows)

    def _update_scrollregion(self) -> None:
        if not self._files:
            return
        w = max(self._canvas.winfo_width(), 1)
        h = self._num_rows * ROW_HEIGHT
        self._canvas.configure(scrollregion=(0, 0, w, h))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if sys.platform == "darwin":
            self._canvas.yview_scroll(int(-1 * (event.delta)), "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.after_idle(self._sync_visible_rows)

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self._canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(3, "units")
        self.after_idle(self._sync_visible_rows)

    def _browse(self) -> None:
        initial = self._dir_var.get().strip() or str(default_root_dir())
        p = filedialog.askdirectory(initialdir=initial, title="FITS directory")
        if p:
            self._dir_var.set(p)

    def _parse_hill_km(self) -> float:
        try:
            v = float(self._hill_km_var.get().strip())
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
        return DEFAULT_HILL_KM

    def _parse_min_exptime(self) -> float:
        try:
            v = float(self._min_exp_var.get().strip())
            return max(0.0, v)
        except (TypeError, ValueError):
            return 0.0

    def _parse_max_exptime(self) -> float:
        """Upper bound in seconds; empty field means no maximum."""
        s = self._max_exp_var.get().strip()
        if not s:
            return float("inf")
        try:
            v = float(s)
            return v if v >= 0 else float("inf")
        except (TypeError, ValueError):
            return 60.0

    def _clear_grid(self) -> None:
        for f in self._pending:
            f.cancel()
        self._pending.clear()
        for futs in self._row_futures.values():
            for fu in futs:
                fu.cancel()
        self._row_futures.clear()
        for wid in self._row_canvas_ids.values():
            self._canvas.delete(wid)
        self._row_canvas_ids.clear()
        self._materialized_rows.clear()
        if self._placeholder_item is not None:
            self._canvas.delete(self._placeholder_item)
            self._placeholder_item = None
        self._files = []
        self._file_rows = []
        self._row_group_ids = []
        self._groups_paths = []
        self._group_selection.clear()
        self._num_rows = 0
        self._file_selection.clear()

    def _clear_all_selections(self) -> None:
        if self._group_selection:
            for gi in self._group_selection:
                self._group_selection[gi].set(False)
            for p in self._files:
                self._file_selection[p].set(False)
        else:
            for v in self._file_selection.values():
                v.set(False)

    def _sync_group_to_files(self, gid: int) -> None:
        """Sync group checkbox to per-file flags (for future batch actions; Stack is off in Groups view)."""
        val = self._group_selection[gid].get()
        for p in self._groups_paths[gid]:
            self._file_selection[p].set(val)

    def _update_action_buttons_state(self) -> None:
        """Stack needs well-separated images (disabled in Groups). Stars works in both modes."""
        try:
            self._stack_btn.configure(state=tk.DISABLED if self._groups_var.get() else tk.NORMAL)
        except tk.TclError:
            pass

    def _open_full_hill(self) -> None:
        if self._groups_var.get():
            messagebox.showinfo(
                "Stack",
                "Stack is not available in Groups view — it needs well-separated images.",
            )
            return
        if not self._files:
            messagebox.showinfo("Stack", "No FITS files loaded.")
            return
        selected = [p for p in self._files if self._file_selection[p].get()]
        if not selected:
            messagebox.showinfo(
                "Stack",
                "Select at least one FITS file (checkbox) to open the stack.",
            )
            return
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            messagebox.showerror("Stack", "Browse to a valid FITS directory first.")
            return
        all_sorted = list_fits_files(root)
        FullHillWindow(self, selected, all_sorted)

    def _open_stars(self) -> None:
        """Stars analysis (works in normal grid or Groups view)."""
        if not self._files:
            messagebox.showinfo("Stars", "No FITS files loaded.")
            return
        selected = [p for p in self._files if self._file_selection[p].get()]
        if not selected:
            if self._group_selection:
                msg = "Select at least one group (checkbox) to run Stars analysis."
            else:
                msg = "Select at least one FITS file (checkbox) to run Stars analysis."
            messagebox.showinfo("Stars", msg)
            return
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            messagebox.showerror("Stars", "Browse to a valid FITS directory first.")
            return
        StarsWindow(self, selected)

    def _refresh_psf_status(self) -> None:
        from stars_analysis import stars_psf_source

        src = stars_psf_source()
        if src is not None:
            self._psf_status_var.set(f"Stars PSF: {display_fits_name(src)}")
        else:
            self._psf_status_var.set("Stars PSF: not defined (fake satellite off)")

    def _sync_open_stars_windows_fake_sat(self) -> None:
        """Re-enable or grey Stars fake-satellite fields when Define PSF / Clear PSF runs."""

        def walk(w: tk.Misc) -> None:
            for ch in w.winfo_children():
                if isinstance(ch, StarsWindow):
                    try:
                        ch.after_idle(ch._sync_fake_sat_controls_state)
                    except tk.TclError:
                        pass
                walk(ch)

        try:
            walk(self.winfo_toplevel())
        except tk.TclError:
            pass

    def _define_psf(self) -> None:
        """Build PSF from one selected FITS; reuse for all Stars planes until cleared."""
        if not self._files:
            messagebox.showinfo("Define PSF", "No FITS files loaded.")
            return
        selected = [p for p in self._files if self._file_selection[p].get()]
        if len(selected) != 1:
            messagebox.showinfo("Define PSF", "Select exactly one FITS file (checkbox).")
            return
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            messagebox.showerror("Define PSF", "Browse to a valid FITS directory first.")
            return
        path = selected[0]

        def work() -> None:
            try:
                from stars_analysis import set_stars_psf_from_image

                set_stars_psf_from_image(path)
                self.after(0, lambda p=path: self._define_psf_done(p, None))
            except Exception as e:
                self.after(0, lambda err=e: self._define_psf_done(None, err))

        threading.Thread(target=work, daemon=True).start()

    def _define_psf_done(self, path_ok: Path | None, err: Exception | None) -> None:
        if err is not None:
            messagebox.showerror("Define PSF", str(err))
            return
        assert path_ok is not None
        self._refresh_psf_status()
        self._sync_open_stars_windows_fake_sat()
        messagebox.showinfo("Define PSF", f"PSF defined from:\n{display_fits_name(path_ok)}")

    def _clear_psf(self) -> None:
        from stars_analysis import clear_stars_psf_override

        clear_stars_psf_override()
        self._refresh_psf_status()
        self._sync_open_stars_windows_fake_sat()

    def _apply_thumb_result(self, ph: ttk.Label, meta_lbl: ttk.Label, m: dict) -> None:
        if not ph.winfo_exists():
            return
        if m.get("error") or m.get("image") is None:
            ph.configure(text="Failed")
            meta_lbl.configure(text=m.get("error", "Unknown error"))
            return
        photo = ImageTk.PhotoImage(m["image"])
        ph.photo_ref = photo  # keep pixmap ref while label exists
        ph.configure(image=photo)
        lines = [m["sapid"] if m.get("sapid") else "—"]
        h = m.get("hours_to_ca")
        if h is not None:
            lines.append(f"Δt CA (h): {h:+.4f}")
        else:
            lines.append("Δt CA (h): —")
        et = m.get("exptime")
        if et is not None:
            lines.append(f"EXPTIME: {et:g}")
        else:
            lines.append("EXPTIME: —")
        ri = m.get("range_int")
        if ri is not None:
            lines.append(f"range: {ri}")
        else:
            lines.append("range: —")
        if m.get("cache_error"):
            lines.append(f"cache write: {m['cache_error']}")
        meta_lbl.configure(text="\n".join(lines))

    def _dematerialize_row(self, r: int) -> None:
        if r not in self._materialized_rows:
            return
        for fu in self._row_futures.pop(r, []):
            fu.cancel()
            self._pending.discard(fu)
        wid = self._row_canvas_ids.pop(r, None)
        if wid is not None:
            self._canvas.delete(wid)
        self._materialized_rows.pop(r, None)

    def _materialize_row(self, r: int) -> None:
        if r in self._materialized_rows:
            return
        row_paths = self._file_rows[r]
        row_frame = ttk.Frame(self._canvas)
        content = ttk.Frame(row_frame)
        # Groups mode: thick left accent color (per group) + "Group N" on first row of each group.
        # Canvas-embedded bottom strips were unreliable; this stays inside the row window.
        if self._row_group_ids and r < len(self._row_group_ids):
            gid = self._row_group_ids[r]
            color = GROUP_ACCENT_COLORS[gid % len(GROUP_ACCENT_COLORS)]
            bar = tk.Frame(
                row_frame,
                width=GROUP_ACCENT_WIDTH,
                bg=color,
                highlightthickness=0,
                borderwidth=0,
            )
            bar.grid(row=0, column=0, sticky=tk.NS)
            content.grid(row=0, column=1, sticky=tk.NSEW)
            row_frame.columnconfigure(1, weight=1)
            row_frame.rowconfigure(0, weight=1)
        else:
            content.grid(row=0, column=0, sticky=tk.NSEW)
            row_frame.columnconfigure(0, weight=1)
            row_frame.rowconfigure(0, weight=1)
        cw = max(self._canvas.winfo_width(), 1)
        win_id = self._canvas.create_window(
            0,
            r * ROW_HEIGHT,
            window=row_frame,
            anchor=tk.NW,
            width=cw,
            height=ROW_HEIGHT,
        )
        self._row_canvas_ids[r] = win_id
        for c in range(len(row_paths)):
            content.columnconfigure(c, weight=1)

        use_group_checks = bool(self._group_selection)
        thumb_row = 0
        if self._row_group_ids and r < len(self._row_group_ids):
            gid = self._row_group_ids[r]
            if r == 0 or self._row_group_ids[r - 1] != gid:
                hdr = ttk.Frame(content)
                hdr.grid(row=0, column=0, columnspan=len(row_paths), sticky=tk.EW, pady=(0, 6))
                if use_group_checks:
                    ttk.Checkbutton(
                        hdr,
                        variable=self._group_selection[gid],
                        command=lambda g=gid: self._sync_group_to_files(g),
                    ).pack(side=tk.LEFT, padx=(0, 8))
                ttk.Label(
                    hdr,
                    text=f"Group {gid + 1}",
                    font=("TkDefaultFont", 10, "bold"),
                ).pack(side=tk.LEFT)
                thumb_row = 1

        futs: list[concurrent.futures.Future] = []
        for c, fp in enumerate(row_paths):
            cell = ttk.Frame(content, padding=3, relief=tk.GROOVE)
            cell.grid(row=thumb_row, column=c, padx=2, pady=2, sticky=tk.NSEW)
            if not use_group_checks:
                chk_row = ttk.Frame(cell)
                chk_row.pack(anchor=tk.W, fill=tk.X)
                ttk.Checkbutton(chk_row, variable=self._file_selection[fp]).pack(side=tk.LEFT)
            ph = ttk.Label(cell, text="Loading…", anchor=tk.CENTER)
            ph.pack()
            ttk.Label(cell, text=display_fits_name(fp), wraplength=240, justify=tk.CENTER).pack()
            meta_lbl = ttk.Label(cell, text="", wraplength=260, justify=tk.LEFT)
            meta_lbl.pack()
            future = self._executor.submit(load_thumb_job, fp, self._parse_hill_km())
            self._pending.add(future)
            futs.append(future)

            def done(fut: concurrent.futures.Future, _ph: ttk.Label = ph, _meta: ttk.Label = meta_lbl) -> None:
                self._pending.discard(fut)
                try:
                    m = fut.result()
                except Exception as e:
                    if _ph.winfo_exists():
                        _ph.configure(text="Error")
                        _meta.configure(text=str(e))
                    return
                self._apply_thumb_result(_ph, _meta, m)

            future.add_done_callback(lambda fut, cb=done: self.after(0, lambda: cb(fut)))

        self._materialized_rows[r] = row_frame
        self._row_futures[r] = futs

    def _sync_visible_rows(self) -> None:
        if not self._files or self._num_rows <= 0:
            return
        self.update_idletasks()
        try:
            top_px = float(self._canvas.canvasy(0))
        except tk.TclError:
            return
        win_h = max(1, min(self._canvas.winfo_height(), MAX_VIEWPORT_PX))
        total_h = float(max(1, self._num_rows * ROW_HEIGHT))
        top_px = max(0.0, min(top_px, max(0.0, total_h - win_h)))
        first = int(top_px // ROW_HEIGHT)
        last = int((top_px + win_h) // ROW_HEIGHT)
        first = max(0, first - ROW_OVERSCAN)
        last = min(self._num_rows - 1, last + ROW_OVERSCAN)
        if last - first + 1 > MAX_ROWS_PER_SYNC:
            last = min(self._num_rows - 1, first + MAX_ROWS_PER_SYNC - 1)
        needed = set(range(first, last + 1))

        for r in list(self._materialized_rows.keys()):
            if r not in needed:
                self._dematerialize_row(r)

        for r in range(first, last + 1):
            self._materialize_row(r)

    def _load(self) -> None:
        try:
            self._load_body()
        finally:
            self._update_action_buttons_state()

    def _load_body(self) -> None:
        self._clear_grid()
        root = Path(self._dir_var.get().strip())
        if not root.is_dir():
            box = ttk.Frame(self._canvas)
            ttk.Label(
                box,
                text=f"Not a directory (pick another folder with Browse…):\n{root}",
                justify=tk.CENTER,
            ).pack(pady=24)
            cw = max(self._canvas.winfo_width(), 400)
            self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
            self._canvas.configure(scrollregion=(0, 0, cw, 120))
            return
        delete_legacy_thumb_pngs(root)
        files = list_fits_files(root)
        if not files:
            box = ttk.Frame(self._canvas)
            ttk.Label(
                box,
                text=f"No FITS files found in:\n{root}",
                justify=tk.CENTER,
            ).pack(pady=24)
            cw = max(self._canvas.winfo_width(), 400)
            self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
            self._canvas.configure(scrollregion=(0, 0, cw, 120))
            return

        min_exp = self._parse_min_exptime()
        max_exp = self._parse_max_exptime()
        files = filter_fits_by_exptime(files, min_exp, max_exp)
        if not files:
            box = ttk.Frame(self._canvas)
            parts: list[str] = []
            if min_exp > 0:
                parts.append(f"EXPTIME ≥ {min_exp:g} s")
            if max_exp < float("inf"):
                parts.append(f"EXPTIME ≤ {max_exp:g} s")
            cond = " and ".join(parts) if parts else "the exposure filter"
            msg = f"No FITS files in:\n{root}\nmatching {cond}" if parts else f"No FITS files found in:\n{root}"
            ttk.Label(box, text=msg, justify=tk.CENTER).pack(pady=24)
            cw = max(self._canvas.winfo_width(), 400)
            self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
            self._canvas.configure(scrollregion=(0, 0, cw, 160))
            return

        one_x_one = self._one_x_one_var.get()
        if one_x_one:
            files = [p for p in files if is_1x1_binned_filename(p)]
        if not files:
            box = ttk.Frame(self._canvas)
            parts: list[str] = []
            if min_exp > 0:
                parts.append(f"EXPTIME ≥ {min_exp:g} s")
            if max_exp < float("inf"):
                parts.append(f"EXPTIME ≤ {max_exp:g} s")
            if one_x_one:
                parts.append("1×1 binning in filename")
            cond = " and ".join(parts) if parts else "the current filters"
            msg = f"No FITS files in:\n{root}\nmatching {cond}"
            ttk.Label(box, text=msg, justify=tk.CENTER).pack(pady=24)
            cw = max(self._canvas.winfo_width(), 400)
            self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
            self._canvas.configure(scrollregion=(0, 0, cw, 180))
            return

        excl = parse_comma_keywords(self._sap_exclude_var.get())
        incl = parse_comma_keywords(self._sap_include_var.get())
        files = filter_paths_by_sap_keywords(files, excl, incl)
        if not files:
            box = ttk.Frame(self._canvas)
            bits: list[str] = []
            if excl:
                bits.append("filter out: " + ", ".join(excl))
            if incl:
                bits.append("only include: " + ", ".join(incl))
            sap_txt = "; ".join(bits) if bits else "SAP filters"
            msg = (
                f"No FITS files in:\n{root}\nmatching {sap_txt} (SAPID/SAP text), "
                "after other filters."
            )
            ttk.Label(box, text=msg, justify=tk.CENTER).pack(pady=24)
            cw = max(self._canvas.winfo_width(), 400)
            self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
            self._canvas.configure(scrollregion=(0, 0, cw, 180))
            return

        groups_active = self._groups_var.get()
        if groups_active:
            groups = compute_exposure_groups(files)
            if not groups:
                box = ttk.Frame(self._canvas)
                ttk.Label(
                    box,
                    text=f"No multi-image groups in Groups view for:\n{root}\n"
                    "Need MIDUTCJD and EXPTIME, and at least two images whose times all fall "
                    "within 10 s of each other in a segment (after the longest-EXPTIME rule), "
                    "after current filters.",
                    justify=tk.CENTER,
                ).pack(pady=24)
                cw = max(self._canvas.winfo_width(), 400)
                self._placeholder_item = self._canvas.create_window(0, 0, window=box, anchor=tk.NW, width=cw)
                self._canvas.configure(scrollregion=(0, 0, cw, 180))
                return
            self._file_rows = []
            self._row_group_ids = []
            self._groups_paths = list(groups)
            self._group_selection = {gi: tk.BooleanVar(value=False) for gi in range(len(groups))}
            for gi, group in enumerate(groups):
                for start in range(0, len(group), COLS):
                    self._file_rows.append(group[start : start + COLS])
                    self._row_group_ids.append(gi)
            self._files = [p for row in self._file_rows for p in row]
        else:
            self._file_rows = [files[i : i + COLS] for i in range(0, len(files), COLS)]
            self._row_group_ids = []
            self._groups_paths = []
            self._group_selection = {}
            self._files = list(files)

        self._file_selection = {p: tk.BooleanVar(value=False) for p in self._files}
        self._num_rows = len(self._file_rows)
        self._update_scrollregion()
        self._canvas.yview_moveto(0)
        # Defer until after the window is mapped so canvas/window geometry is real.
        self.after(150, self._sync_visible_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="FITS thumbnail viewer (Lucy LORRI).")
    parser.add_argument(
        "--minexp",
        type=float,
        default=None,
        metavar="SEC",
        help="Default minimum EXPTIME in seconds for the Min EXPTIME filter (same as the UI field).",
    )
    parser.add_argument(
        "--stars",
        nargs="+",
        metavar="FITS",
        help=(
            "Open only the Stars analysis window for these FITS paths (a group), "
            "without the thumbnail browser. Example: %(prog)s --stars a.fit b.fit"
        ),
    )
    args = parser.parse_args()
    if args.stars:
        root = tk.Tk()
        root.title("")
        root.resizable(False, False)
        # Do not ``withdraw()`` the root: many Linux WMs then never map Toplevel children.
        # Park a 1×1 root off-screen (avoid ``-alpha`` on root — it can hide child windows too).
        root.geometry("1x1+10000+10000")
        paths = [Path(p).expanduser().resolve() for p in args.stars]
        win = StarsWindow(root, paths, quit_root_when_destroyed=True)
        win.update_idletasks()
        win.lift()
        win.focus_force()
        root.mainloop()
        return
    min_exp_default = "0" if args.minexp is None else str(args.minexp)
    app = FitsThumbViewer(min_exp_default=min_exp_default)
    app.mainloop()


if __name__ == "__main__":
    main()
