"""
Python port of fullhill.pro (no fullhillshifts.dat).

- Short-exposure mates: file index ``idx - 2`` in the full sorted directory listing.
- Exposure scaling: ``EXPTIME_main / EXPTIME_short`` from FITS headers (replaces fixed *100).
- Alignment: central ROI → 3×3 median (cosmic rays) → peak → 30×30 patch centroid on each
  short image; offsets ``xs[i]=cx_i-cx_0``, ``ys[i]=cy_i-cy_0``; same shifts on mains/shorts
  via ``scipy.ndimage.shift`` (order 3). ``xs[0]=ys[0]=0``.
- Closest-approach timing: ``MIDUTCJD`` differences (same reference as thumbnails), not MIDSCLK.
- Geometry (range, phase, Sun–target distance): SPICE via :mod:`lucy_spice` (IDL ``getgeometry.pro``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage as ndi

from fits_thumb_viewer import (
    CA_REF_JD,
    arcsec_per_pixel_from_filename,
    get_image_data_and_header,
    headers_for_metadata,
    primary_exptime_seconds,
    sky_scale,
    _midutcjd_from_headers,
)
from lucy_getpsf import lucy_getpsf

G = 6.64e-11
ZPT = 18.933
VOLUME_KM3 = 58.3
# IDL: mass=volume*1.2/1000.*1.e15
MASS_KG = VOLUME_KM3 * 1.2 / 1000.0 * 1.0e15
G1 = 0.63
G2 = 0.18
AU_KM = 1.496e8


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_g1g2_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = _package_dir() / "g1g2tab.dat"
    raw = np.loadtxt(p)
    ang = raw[:, 0]
    f1 = raw[:, 1]
    f2 = raw[:, 2]
    f3 = raw[:, 3]
    return ang, f1, f2, f3


_ANG_F1_F2_F3 = _load_g1g2_table()


def _interp_phase(phase_deg: float, col: np.ndarray) -> float:
    ang = _ANG_F1_F2_F3[0]
    p = float(np.clip(phase_deg, float(ang[0]), float(ang[-1])))
    return float(np.interp(p, ang, col))


def fakesat_flux(
    diam_m: float,
    albedo: float,
    range_km: float,
    delta_km: float,
    phase_deg: float,
) -> float:
    """Return model count-rate scale factor (IDL fakesat). Diameter in meters."""
    ang = _ANG_F1_F2_F3[0]
    f1i = _interp_phase(phase_deg, _ANG_F1_F2_F3[1])
    f2i = _interp_phase(phase_deg, _ANG_F1_F2_F3[2])
    f3i = _interp_phase(phase_deg, _ANG_F1_F2_F3[3])
    phase_term = G1 * f1i + G2 * f2i + (1.0 - G1 * G2) * f3i
    H = -5.0 * math.log10(max(diam_m / 1000.0 * math.sqrt(albedo) / 1369.0, 1e-30))
    v0 = H - 2.5 * math.log10(max(phase_term, 1e-30))
    v_mag = (
        v0
        + 5.0 * math.log10(max(delta_km / AU_KM, 1e-30))
        + 5.0 * math.log10(max(range_km / AU_KM, 1e-30))
    )
    return float(10.0 ** (-(v_mag - ZPT) / 2.5))


def diameter_for_fakesat_total_counts(
    total_counts: float,
    albedo: float,
    range_km: float,
    delta_km: float,
    phase_deg: float,
    exptime_s: float,
) -> float | None:
    """Invert :func:`fakesat_flux` × exposure time to a sphere diameter (meters).

    ``total_counts`` is the modeled **integrated** source counts in one exposure (same units as
    ``fakesat_flux(...) * exptime`` in stack/stars fake injection). Uses bisection on diameter
    between ``1e-6`` m and ``1e8`` m; returns ``None`` if inputs are non-finite or non-positive.
    """
    if (
        not math.isfinite(total_counts)
        or total_counts <= 0
        or not math.isfinite(exptime_s)
        or exptime_s <= 0
    ):
        return None
    target_rate = float(total_counts) / float(exptime_s)
    if not math.isfinite(target_rate) or target_rate <= 0:
        return None

    lo_d, hi_d = 1e-6, 1e8
    f_lo = fakesat_flux(lo_d, albedo, range_km, delta_km, phase_deg)
    f_hi = fakesat_flux(hi_d, albedo, range_km, delta_km, phase_deg)
    if target_rate <= f_lo:
        return lo_d
    if target_rate >= f_hi:
        return hi_d

    lo, hi = lo_d, hi_d
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        fm = fakesat_flux(mid, albedo, range_km, delta_km, phase_deg)
        if fm < target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < max(lo, 1e-30) * 1e-10:
            break
    return float(0.5 * (lo + hi))


def sky_subtract_median(im: np.ndarray) -> tuple[np.ndarray, float]:
    m = float(np.nanmedian(im))
    return im - m, m


CENTROID_SMALL_BOX = 30  # final centroid box (pixels), centered on smoothed maximum


def centroid_bright_near_center(im: np.ndarray) -> tuple[float, float]:
    """Find centroid of the bright target near the field center (x, y = column, row).

    1. Take a square ROI centered on the image (half-width ``min(128, h/4, w/4)``).
    2. Apply 3×3 median filtering to the ROI (cosmic-ray rejection).
    3. Locate the maximum on the smoothed ROI (object center).
    4. Take a ``CENTROID_SMALL_BOX``×``CENTROID_SMALL_BOX`` region centered on that peak
       (clipped at ROI edges), subtract patch median, weighted centroid in that patch.
    5. Map back to full-image coordinates.

    The source should lie inside the initial ROI (see ``tests/test_fullhill_centroid.py``).
    """
    sub = np.asarray(im, dtype=np.float64)
    h, w = sub.shape[:2]
    hb = min(128, h // 4, w // 4)
    r0, r1 = h // 2 - hb, h // 2 + hb
    c0, c1 = w // 2 - hb, w // 2 + hb
    roi = sub[r0:r1, c0:c1].copy()
    rh, rw = roi.shape[:2]

    smoothed = ndi.median_filter(roi, size=3, mode="nearest")

    flat_idx = int(np.nanargmax(smoothed))
    my, mx = np.unravel_index(flat_idx, smoothed.shape)

    half = CENTROID_SMALL_BOX // 2
    y0 = max(0, my - half)
    y1 = min(rh, my + half)
    x0 = max(0, mx - half)
    x1 = min(rw, mx + half)
    patch = smoothed[y0:y1, x0:x1]

    pmed = float(np.nanmedian(patch))
    wt = np.maximum(patch - pmed, 0.0)
    if np.sum(wt) <= 0:
        cx_roi = float(mx)
        cy_roi = float(my)
    else:
        yy, xx = np.indices(patch.shape)
        cx_roi = float(np.sum(xx * wt) / np.sum(wt) + x0)
        cy_roi = float(np.sum(yy * wt) / np.sum(wt) + y0)

    cx_full = cx_roi + c0
    cy_full = cy_roi + r0

    return cx_full, cy_full


def xyshift_cubic(im: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    return ndi.shift(
        np.asarray(im, dtype=np.float64),
        (shift_y, shift_x),
        order=3,
        mode="constant",
        cval=0.0,
    )


def ensure_square_1024(data: np.ndarray) -> np.ndarray:
    """Center-crop to 1024×1024 when larger; smaller images are rejected."""
    a = np.asarray(data, dtype=np.float64)
    h, w = a.shape[:2]
    if h < 1024 or w < 1024:
        raise ValueError(f"Full hill expects at least 1024×1024 data; got {h}×{w}")
    if h == w == 1024:
        return a
    y0 = (h - 1024) // 2
    x0 = (w - 1024) // 2
    return a[y0 : y0 + 1024, x0 : x0 + 1024].copy()


def geometry_from_spice(path: Path) -> tuple[float, float, float]:
    """(range_km Lucy–target, phase_deg, delta_km Sun–target) from SPICE (IDL ``getgeometry.pro``)."""
    from lucy_spice import range_phase_delta_km

    return range_phase_delta_km(path)


# Backward-compatible name
geometry_from_headers = geometry_from_spice


def jd_from_path(path: Path) -> float:
    hlist = headers_for_metadata(path)
    jd = _midutcjd_from_headers(hlist)
    if jd is None:
        raise ValueError(f"Missing MIDUTCJD in {path.name}")
    return float(jd)


def omega_rad_per_s(satdist_km: float) -> float:
    """Circular-motion angular rate: ``sqrt(G*M/R)/R`` with ``R`` in meters."""
    r_m = float(satdist_km) * 1000.0
    if r_m <= 0:
        return 0.0
    return math.sqrt(G * MASS_KG / r_m) / r_m


def _idl_congrid_linear_square(im: np.ndarray, nout: int) -> np.ndarray:
    """Linear resample a square image to ``nout×nout`` (IDL ``congrid`` edge mapping).

    ``scipy.ndimage.zoom`` scales the index grid from the origin and rounds output
    dimensions; that does **not** match IDL ``congrid``, so the mapping from input
    pixel coordinates to output drifts with zoom factor. Different ``sz`` per plane then
    misalign ``imz`` / ``imzs`` in the stack when the resampled field extends past edges
    (zeros from ``xyshift``) or when cropping the central 1024².

    IDL maps the first input pixel to the first output and the last to the last via
    linear interpolation — equivalent to sampling at ``linspace(0, hin-1, nout)``.
    """
    im = np.asarray(im, dtype=np.float64)
    hin, win = im.shape[:2]
    if hin != win:
        raise ValueError("_idl_congrid_linear_square expects a square image")
    if nout <= 0:
        nout = 1
    y = np.linspace(0.0, float(hin - 1), nout, dtype=np.float64)
    x = np.linspace(0.0, float(win - 1), nout, dtype=np.float64)
    yi, xi = np.meshgrid(y, x, indexing="ij")
    coords = np.empty((2, nout, nout), dtype=np.float64)
    coords[0] = yi
    coords[1] = xi
    return ndi.map_coordinates(im, coords, order=1, mode="constant", cval=0.0)


def congrid_zoom_center(im: np.ndarray, out_side: int, target_side: int = 1024) -> np.ndarray:
    """IDL-style ``congrid`` to ``out_side²`` then central crop/pad to ``target_side²``."""
    im = np.asarray(im, dtype=np.float64)
    h, w = im.shape[:2]
    if h != w:
        raise ValueError("congrid_zoom_center expects a square image")
    if out_side <= 0:
        out_side = 1
    zim = _idl_congrid_linear_square(im, out_side)
    sz = int(zim.shape[0])
    out = np.zeros((target_side, target_side), dtype=np.float64)
    if sz >= target_side:
        c0 = sz // 2 - target_side // 2
        out[:, :] = zim[c0 : c0 + target_side, c0 : c0 + target_side]
    else:
        p0 = (target_side - sz) // 2
        out[p0 : p0 + sz, p0 : p0 + sz] = zim
    return out


def congrid_display_to_native_xy(
    xd: float,
    yd: float,
    sz: int,
    target_side: int = 1024,
) -> tuple[float, float] | None:
    """Invert :func:`congrid_zoom_center`: display pixel ``(xd,yd)`` → native ``[0, target_side-1]``.

    ``sz`` is the ``out_side`` passed to ``_idl_congrid_linear_square`` (same as ``congrid_zoom_center``).
    Returns ``None`` if ``(xd,yd)`` lies in zero-padding (only when ``sz < target_side``).
    """
    T = int(target_side)
    sz = int(sz)
    if sz <= 0:
        return None
    hin = float(T - 1)
    if sz >= T:
        c0 = sz // 2 - T // 2
        jx = xd + float(c0)
        jy = yd + float(c0)
    else:
        p0 = (T - sz) // 2
        jx = xd - float(p0)
        jy = yd - float(p0)
        if jx < -0.5 or jx > sz - 0.5 or jy < -0.5 or jy > sz - 0.5:
            return None
    if sz <= 1:
        return 0.0, 0.0
    scale = hin / float(sz - 1)
    return jx * scale, jy * scale


def imz_congrid_out_side(rr: np.ndarray, k: int) -> int:
    """``out_side`` for plane ``k`` (same rule as :func:`run_fullhill_from_prep`)."""
    rr = np.asarray(rr, dtype=np.float64)
    rr_last = float(rr[-1])
    if rr_last <= 0:
        return 1024
    return max(2, int(round(1024.0 * float(rr[k]) / rr_last)))


# Target position in the **shifted** 1024² image that is passed to :func:`congrid_zoom_center`
# (``xyshift_cubic`` uses ``511.5 - djx`` / ``511.5 - djy`` so the body sits at the field center).
IMZ_CONGRID_INPUT_TARGET_CX = (1024 - 1) / 2.0
IMZ_CONGRID_INPUT_TARGET_CY = (1024 - 1) / 2.0


def km_distance_to_target(
    xd: float,
    yd: float,
    *,
    target_cx: float,
    target_cy: float,
    kpp_km: float,
    sz_zoom: int | None,
    target_side: int = 1024,
) -> float | None:
    """Distance in km from the science target to ``(xd,yd)`` in display pixels.

    ``target_cx``, ``target_cy`` must match the coordinate system of the inverted native
    position:

    - **``imb`` / ``imbs``:** use :attr:`FullHillPrep.djx` / ``djy`` (target in aligned mains).
    - **``imz`` / ``imzs``:** use :data:`IMZ_CONGRID_INPUT_TARGET_CX` / ``CY`` (``511.5``) —
      the stack shifts the target to center *before* ``congrid``; :func:`congrid_display_to_native_xy`
      returns coordinates in that shifted frame.

    ``kpp_km`` is km per native pixel at the target (:attr:`FullHillPrep.kpp`).

    If ``sz_zoom`` is ``None``, ``(xd,yd)`` are native stack coordinates (``imb`` / ``imbs``).

    If ``sz_zoom`` is set, display coordinates are inverted through
    :func:`congrid_display_to_native_xy` (``imz`` / ``imzs`` zoom) before applying ``kpp_km``.
    """
    if sz_zoom is None:
        nx, ny = float(xd), float(yd)
    else:
        mapped = congrid_display_to_native_xy(xd, yd, sz_zoom, target_side)
        if mapped is None:
            return None
        nx, ny = mapped
    dx_km = (nx - float(target_cx)) * float(kpp_km)
    dy_km = (ny - float(target_cy)) * float(kpp_km)
    return float(math.hypot(dx_km, dy_km))


StackName = Literal["imb", "imbs", "imz", "imzs", "median_imz", "median_imzs"]


@dataclass
class FullHillResult:
    imb: np.ndarray
    imbs: np.ndarray
    imz: np.ndarray
    imzs: np.ndarray
    #: Per-pixel median over planes (single 1024² frame each).
    median_imz: np.ndarray
    median_imzs: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    short_paths: list[Path]
    exposure_scale: list[float]
    #: Primary ``EXPTIME`` from each main FITS (seconds).
    exptime_main: list[float]
    hours_from_ca: np.ndarray


@dataclass
class FullHillPrep:
    """Cached load, alignment, sky-subtracted mains, and per-plane PSFs.

    Satellite parameters (diameter, albedo, distance, angle) only affect the fake source
    and downstream stacks; use :func:`run_fullhill_from_prep` to recompute those quickly.
    """

    imb: np.ndarray
    psf: list[np.ndarray]
    #: Per-plane primary ``EXPTIME`` from each **main** FITS (seconds), via
    #: :func:`fits_thumb_viewer.primary_exptime_seconds`. Used for ``fakesat_flux * exptime``;
    #: not the short mate's ``EXPTIME``, and not :attr:`exposure_scale` (``et_m / et_short``).
    et_m: np.ndarray
    rr: np.ndarray
    ph_deg: np.ndarray
    delta_km: np.ndarray
    kpp: np.ndarray
    jd_arr: np.ndarray
    dj: np.ndarray
    dj_den: np.ndarray
    djx: float
    djy: float
    short_paths: list[Path]
    exposure_scale: list[float]
    hours_ca: np.ndarray
    xs: np.ndarray
    ys: np.ndarray


def _resolve_short_path(
    main_path: Path, all_sorted: list[Path]
) -> tuple[int, Path]:
    key = main_path.resolve()
    idx_map = {p.resolve(): i for i, p in enumerate(all_sorted)}
    if key not in idx_map:
        raise ValueError(f"{main_path.name} not found in directory listing")
    idx = idx_map[key]
    if idx < 2:
        raise ValueError(
            f"Need at least two earlier files in the folder for short mate (index {idx} < 2): {main_path.name}"
        )
    return idx, all_sorted[idx - 2]


def _dj_flux_denominators(dj: np.ndarray) -> np.ndarray:
    """Positive divisors for ``imb/dj``-style flux normalization (``fullhill.pro``).

    ``dj[i]`` is the sum of sky-subtracted short pixels in the photometry box. That sum
    is meant as a **brightness scale** for dividing out exposure/source strength. Using
    the **signed** sum as the divisor flips the entire plane when the box total is
    negative (common after sky subtraction), so:

    - normalized cubes mix opposite signs across planes and ``median(..., axis=2)`` is a
      poor shared PSF model (often making plane 0 look “unsubtracted”);
    - ``/dj[i]*dj[0]`` in the zoom views negates planes whenever ``sign(dj[i])`` differs
      from ``sign(dj[0])``, which inverts grayscale in ``sky_scale`` display.

    We therefore normalize by ``|dj|`` with a small relative floor so every plane uses a
    positive flux scale and the same normalization path.
    """
    abs_d = np.abs(np.asarray(dj, dtype=np.float64))
    m = float(np.max(abs_d)) if abs_d.size else 0.0
    floor = max(m, 1.0) * 1e-15
    return np.maximum(abs_d, floor)


def run_fullhill_prepare(
    main_paths: list[Path],
    all_sorted: list[Path],
) -> FullHillPrep:
    """
    Load FITS, align, sky-subtract mains, ``lucy_getpsf`` per plane, and photometry scalars.

    Call once per selection; reuse with :func:`run_fullhill_from_prep` when only satellite
    parameters change.
    """
    n = len(main_paths)
    if n == 0:
        raise ValueError("No main FITS files selected.")

    imb = np.zeros((1024, 1024, n), dtype=np.float64)
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    short_paths: list[Path] = []
    exposure_scale: list[float] = []

    rr = np.zeros(n, dtype=np.float64)
    ph_deg = np.zeros(n, dtype=np.float64)
    delta_km = np.zeros(n, dtype=np.float64)
    dj = np.zeros(n, dtype=np.float64)
    kpp = np.zeros(n, dtype=np.float64)
    jd_arr = np.zeros(n, dtype=np.float64)
    hours_ca = np.zeros(n, dtype=np.float64)

    ims_raw: list[np.ndarray] = []
    for i, main_p in enumerate(main_paths):
        _, short_p = _resolve_short_path(main_p, all_sorted)
        short_paths.append(short_p)
        data_s, _ = get_image_data_and_header(short_p)
        data_m, _ = get_image_data_and_header(main_p)
        et_s = primary_exptime_seconds(short_p)
        et_m = primary_exptime_seconds(main_p)
        if et_s is None or et_s <= 0 or et_m is None or et_m <= 0:
            raise ValueError(
                f"Need EXPTIME on both mates: {short_p.name} ({et_s}), {main_p.name} ({et_m})"
            )
        scale = float(et_m / et_s)
        exposure_scale.append(scale)
        ds = ensure_square_1024(np.asarray(data_s, dtype=np.float64) * scale)
        ims_raw.append(sky_subtract_median(ds)[0])

    cx0: float | None = None
    cy0: float | None = None
    ims_aligned = []
    for i in range(n):
        cxi, cyi = centroid_bright_near_center(ims_raw[i])
        if i == 0:
            cx0, cy0 = float(cxi), float(cyi)
            djx, djy = cx0, cy0
            xs[i] = 0.0
            ys[i] = 0.0
            ims_aligned.append(ims_raw[0].copy())
        else:
            assert cx0 is not None and cy0 is not None
            xs[i] = cxi - cx0
            ys[i] = cyi - cy0
            ims_aligned.append(xyshift_cubic(ims_raw[i], -ys[i], -xs[i]))

    raw_shifted: list[np.ndarray] = []
    for i, main_p in enumerate(main_paths):
        data_m, _hdr_m = get_image_data_and_header(main_p)
        im = ensure_square_1024(np.asarray(data_m, dtype=np.float64))
        if i > 0:
            im = xyshift_cubic(im, -ys[i], -xs[i])
        raw_shifted.append(im.copy())
        sk, _ = sky_subtract_median(im)
        imb[:, :, i] = sk

        rr[i], ph_deg[i], delta_km[i] = geometry_from_spice(main_p)
        asp = arcsec_per_pixel_from_filename(main_p)
        kpp[i] = asp / 3600.0 * (math.pi / 180.0) * rr[i]

        jd_arr[i] = jd_from_path(main_p)
        hours_ca[i] = (jd_arr[i] - CA_REF_JD) * 24.0

        dj[i] = float(np.sum(ims_aligned[i][430:538, 479:618]))

    dj_den = _dj_flux_denominators(dj)

    psf: list[np.ndarray] = []
    et_m = np.zeros(n, dtype=np.float64)
    for i, main_p in enumerate(main_paths):
        psf.append(lucy_getpsf(raw_shifted[i]))
        et = primary_exptime_seconds(main_p)
        et_m[i] = float(et) if et is not None else 1.0

    return FullHillPrep(
        imb=imb,
        psf=psf,
        et_m=et_m,
        rr=rr,
        ph_deg=ph_deg,
        delta_km=delta_km,
        kpp=kpp,
        jd_arr=jd_arr,
        dj=dj,
        dj_den=dj_den,
        djx=djx,
        djy=djy,
        short_paths=short_paths,
        exposure_scale=exposure_scale,
        hours_ca=hours_ca,
        xs=xs,
        ys=ys,
    )


def run_fullhill_from_prep(
    prep: FullHillPrep,
    diam_m: float,
    albedo: float,
    satdist_km: float,
    satang_deg: float,
) -> FullHillResult:
    """Build fake satellite on a **copy** of ``prep.imb`` (prep stays pristine for re-run)."""
    imb0 = prep.imb
    n = int(imb0.shape[2])
    dj_den = prep.dj_den
    djx = prep.djx
    djy = prep.djy
    rr = prep.rr
    jd_arr = prep.jd_arr

    # Sky-subtracted mains + fake satellite (never mutate ``prep.imb``).
    imb_sat = imb0.copy()
    imsat = imb0.copy()
    satang_rad = math.radians(satang_deg)
    om = omega_rad_per_s(satdist_km)

    for i in range(n):
        if diam_m <= 0.0:
            plane = imb0[:, :, i]
        else:
            dt_sec = (jd_arr[i] - jd_arr[0]) * 86400.0

            satx_km = satdist_km * math.cos(satang_rad + dt_sec * om)
            saty_km = satdist_km * math.sin(satang_rad + dt_sec * om)

            # Total counts: IDL fakesat × main-frame EXPTIME (same file as this stack plane).
            flux = (
                fakesat_flux(diam_m, albedo, rr[i], prep.delta_km[i], prep.ph_deg[i])
                * prep.et_m[i]
            )
            fake = np.zeros((1024, 1024), dtype=np.float64)
            fake[504:519, 504:519] = flux * prep.psf[i]

            sx_pix = satx_km / prep.kpp[i]
            sy_pix = saty_km / prep.kpp[i]
            shifted = xyshift_cubic(
                fake,
                (djy - 512.5 + sy_pix),
                (djx - 512.5 + sx_pix),
            )
            plane = imb0[:, :, i] + shifted
        imb_sat[:, :, i] = plane
        imsat[:, :, i] = plane

    imbs = imb_sat.copy()
    for i in range(n):
        imbs[:, :, i] /= dj_den[i]

    psf_med = np.median(imbs, axis=2)
    for i in range(n):
        imbs[:, :, i] -= psf_med
    for i in range(n):
        imbs[:, :, i] *= dj_den[i]
    # Same normalization / median PSF as ``imbs`` (input was identical ``imb_sat``).
    imsats = imbs.copy()

    imz = np.array(imsat, copy=True)
    imzs = np.array(imsats, copy=True)

    rr_last = float(rr[-1])
    for i in range(n):
        if rr_last <= 0:
            sz = 1024
        else:
            sz = max(2, int(round(1024.0 * rr[i] / rr_last)))

        sx_c = 511.5 - djx
        sy_c = 511.5 - djy
        a = xyshift_cubic(imsat[:, :, i], sy_c, sx_c)
        a = a / dj_den[i] * dj_den[0]
        imz[:, :, i] = congrid_zoom_center(a, sz, 1024)

        b = xyshift_cubic(imsats[:, :, i], sy_c, sx_c)
        b = b / dj_den[i] * dj_den[0]
        imzs[:, :, i] = congrid_zoom_center(b, sz, 1024)

    median_imz = np.median(imz, axis=2)
    median_imzs = np.median(imzs, axis=2)

    return FullHillResult(
        imb=imb_sat,
        imbs=imbs,
        imz=imz,
        imzs=imzs,
        median_imz=median_imz,
        median_imzs=median_imzs,
        xs=prep.xs,
        ys=prep.ys,
        short_paths=prep.short_paths,
        exposure_scale=prep.exposure_scale,
        exptime_main=[float(x) for x in prep.et_m],
        hours_from_ca=prep.hours_ca,
    )


def run_fullhill(
    main_paths: list[Path],
    all_sorted: list[Path],
    diam_m: float,
    albedo: float,
    satdist_km: float,
    satang_deg: float,
) -> FullHillResult:
    """
    Full pipeline: prepare then satellite-dependent stacks.

    For interactive satellite tuning, call :func:`run_fullhill_prepare` once and then
    :func:`run_fullhill_from_prep` repeatedly.
    """
    prep = run_fullhill_prepare(main_paths, all_sorted)
    return run_fullhill_from_prep(prep, diam_m, albedo, satdist_km, satang_deg)
