"""SPICE geometry and predicted target position (IDL ``getgeometry.pro``, ``lucy_finddj.pro``).

Requires ``spiceypy``, NAIF kernels from :mod:`satsearch_config`, and ``MIDUTCJD`` in the FITS header.
All CSPICE calls are serialized with a lock (kernels loaded once per process).
"""

from __future__ import annotations

import atexit
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from satsearch_config import get_config

if TYPE_CHECKING:
    pass

_lock = threading.Lock()
_kernels_loaded = False


def _ensure_spiceypy():
    try:
        import spiceypy as spice  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "spiceypy is required for SPICE geometry. Install with: pip install spiceypy"
        ) from e


def _furnsh_once() -> None:
    global _kernels_loaded
    import spiceypy as spice

    if _kernels_loaded:
        return
    cfg = get_config()
    for k in cfg.spice.meta_kernels:
        if not k.is_file():
            raise FileNotFoundError(f"SPICE kernel not found: {k}")
        spice.furnsh(str(k))
    _kernels_loaded = True


def _unload_kernels() -> None:
    global _kernels_loaded
    import spiceypy as spice

    if _kernels_loaded:
        try:
            spice.kclear()
        except Exception:
            pass
        _kernels_loaded = False


atexit.register(_unload_kernels)


def et_from_midutcjd(path: Path) -> float:
    """Ephemeris seconds from ``MIDUTCJD`` in the FITS header (matches IDL ``lucy_finddj`` time)."""
    from fits_thumb_viewer import _midutcjd_from_headers, headers_for_metadata

    _ensure_spiceypy()
    import spiceypy as spice

    hlist = headers_for_metadata(path)
    jd = _midutcjd_from_headers(hlist)
    if jd is None or not math.isfinite(float(jd)):
        raise ValueError(f"MIDUTCJD missing or invalid in {path.name}")
    jd = float(jd)
    # IDL: cspice_utc2et, string(time,format='(f25.7)')+' JD', et — not UNITIM(...,'JDT',...),
    # which is not portable (JDT is not a valid UNITIM input on many CSPICE builds).
    with _lock:
        _furnsh_once()
        et = spice.str2et(f"{jd:.7f} JD")
    return float(et)


def wcs_from_path(path: Path) -> WCS:
    """Same as ``stars_analysis.wcs_from_path`` (duplicated to avoid import cycles)."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            hdr = hdu.header
            try:
                w = WCS(hdr, naxis=2)
                if w.naxis >= 2 and ("CRVAL1" in hdr or "CTYPE1" in hdr):
                    return w
            except Exception:
                continue
        return WCS(hdul[0].header, naxis=2, relax=True)


def _lucy_observer_pos_km(path: Path) -> np.ndarray:
    """Target position (km) relative to Lucy only; one ET + one ``spkpos``."""
    _ensure_spiceypy()
    import spiceypy as spice

    cfg = get_config()
    et = et_from_midutcjd(path)
    with _lock:
        _furnsh_once()
        pos, _lt = spice.spkpos(
            cfg.spice.target_body,
            et,
            cfg.spice.frame,
            cfg.spice.abcorr,
            cfg.spice.observer_body,
        )
    return np.asarray(pos, dtype=np.float64).ravel()


def _lucy_sun_pos_km(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Target position (km) from Lucy and from Sun; one ET + two ``spkpos`` under the same lock."""
    _ensure_spiceypy()
    import spiceypy as spice

    cfg = get_config()
    et = et_from_midutcjd(path)
    with _lock:
        _furnsh_once()
        pos, _lt = spice.spkpos(
            cfg.spice.target_body,
            et,
            cfg.spice.frame,
            cfg.spice.abcorr,
            cfg.spice.observer_body,
        )
        spos, _lt2 = spice.spkpos(
            cfg.spice.target_body,
            et,
            cfg.spice.frame,
            cfg.spice.abcorr,
            cfg.spice.sun_body,
        )
    pos = np.asarray(pos, dtype=np.float64).ravel()
    spos = np.asarray(spos, dtype=np.float64).ravel()
    return pos, spos


def _predicted_pixels_from_lucy_pos(path: Path, pos: np.ndarray) -> tuple[float, float]:
    """Map J2000 Lucy→target vector (km) to pixel (x, y) using WCS."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    r = float(np.sqrt(np.sum(pos * pos)))
    if r <= 0:
        raise ValueError(f"SPICE returned zero range for {path.name}")
    uy, ux = float(pos[1] / r), float(pos[0] / r)
    uz = max(-1.0, min(1.0, float(pos[2] / r)))
    ra_rad = math.atan2(uy, ux)
    dec_rad = math.asin(uz)
    ra_deg = math.degrees(ra_rad) % 360.0
    dec_deg = math.degrees(dec_rad)

    wcs = wcs_from_path(path)
    sky_j2000 = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5", equinox="J2000.0")
    sky_icrs = sky_j2000.icrs

    try:
        px, py = skycoord_to_pixel(sky_icrs, wcs, origin=0)
        out_x = float(np.asarray(px).ravel()[0])
        out_y = float(np.asarray(py).ravel()[0])
        if math.isfinite(out_x) and math.isfinite(out_y):
            return out_x, out_y
    except Exception:
        pass

    ra_icrs = float(sky_icrs.ra.deg)
    dec_icrs = float(sky_icrs.dec.deg)
    out = wcs.all_world2pix(
        np.array([[ra_icrs]]),
        np.array([[dec_icrs]]),
        0,
        tolerance=1e-2,
        maxiter=50,
        adaptive=False,
        quiet=True,
    )
    px2, py2 = out[0].flat[0], out[1].flat[0]
    if not (math.isfinite(px2) and math.isfinite(py2)):
        raise ValueError(
            f"WCS could not map SPICE sky position to pixels for {path.name} "
            f"(ICRS ra={ra_icrs:.5f}°, dec={dec_icrs:.5f}°). Check astrometry in the FITS header."
        )
    return float(px2), float(py2)


def range_phase_delta_km(path: Path) -> tuple[float, float, float]:
    """Lucy–target range (km), phase angle (deg), Sun–target distance (km).

    Matches IDL ``getgeometry.pro``, including ``phase = acos(pos·spos)/(range*delta))`` in degrees.
    """
    pos, spos = _lucy_sun_pos_km(path)
    range_km = float(np.sqrt(np.sum(pos * pos)))
    delta_km = float(np.sqrt(np.sum(spos * spos)))
    if range_km <= 0 or delta_km <= 0:
        raise ValueError(f"SPICE returned non-positive range for {path.name}")
    ph_rad = float(np.dot(pos, spos) / (range_km * delta_km))
    ph_rad = max(-1.0, min(1.0, ph_rad))
    phase_deg = math.degrees(math.acos(ph_rad))
    return range_km, phase_deg, delta_km


def stars_ephemeris_bundle(path: Path) -> tuple[float, float, float, float, float]:
    """Single SPICE pass for Stars: geometry + predicted pixels (avoids duplicate ``et``/``spkpos``).

    Returns ``(range_km, phase_deg, delta_km, xpred, ypred)``.
    """
    pos, spos = _lucy_sun_pos_km(path)
    range_km = float(np.sqrt(np.sum(pos * pos)))
    delta_km = float(np.sqrt(np.sum(spos * spos)))
    if range_km <= 0 or delta_km <= 0:
        raise ValueError(f"SPICE returned non-positive range for {path.name}")
    ph_rad = float(np.dot(pos, spos) / (range_km * delta_km))
    ph_rad = max(-1.0, min(1.0, ph_rad))
    phase_deg = math.degrees(math.acos(ph_rad))
    xpred, ypred = _predicted_pixels_from_lucy_pos(path, pos)
    return range_km, phase_deg, delta_km, xpred, ypred


def predicted_target_pixel_xy(path: Path) -> tuple[float, float]:
    """Predicted target (x, y) native pixels from SPICE + WCS (IDL ``lucy_finddj`` + ``ad2xy``).

    Uses one Lucy ``spkpos`` only. For Stars analysis, prefer :func:`stars_ephemeris_bundle` to avoid
    duplicating geometry work.
    """
    pos = _lucy_observer_pos_km(path)
    return _predicted_pixels_from_lucy_pos(path, pos)


def range_km_for_display(path: Path) -> float | None:
    """Lucy–target distance in km for thumbnail Hill overlay; ``None`` if SPICE cannot run."""
    try:
        r, _p, _d = range_phase_delta_km(path)
        return r
    except Exception:
        return None
