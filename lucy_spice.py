"""SPICE geometry and predicted target position (IDL ``getgeometry.pro``, ``lucy_finddj.pro``).

Requires ``spiceypy``, NAIF kernels from :func:`satsearch_config.get_spice_runtime` (kernel
directory from ``satsearch.toml``, filenames from ``encounters.toml`` / UI, ``target_body`` from
the active encounter), and ``MIDUTCJD`` in the FITS header.
When ``[spice].kernel_data_root`` is set in ``satsearch.toml``, each meta-kernel (``.tm``) is read
from disk, ``PATH_VALUES`` text pointing at ``path_values_snapshot_prefix`` is rewritten to that
root, and the result is furnished from a short-lived temp file so shipped Lucy SOC paths need not
be edited by hand.
All CSPICE calls are serialized with a lock (kernels loaded once per process, or reloaded after
an encounter switch).
"""

from __future__ import annotations

import atexit
import math
import os
import tempfile
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from satsearch_config import SpiceSiteConfig, get_config, get_spice_runtime

if TYPE_CHECKING:
    pass

_lock = threading.Lock()
_kernels_loaded = False
_shadow_meta_kernel_paths: list[Path] = []


def _purge_shadow_meta_kernels() -> None:
    global _shadow_meta_kernel_paths
    for p in _shadow_meta_kernel_paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    _shadow_meta_kernel_paths.clear()


def _rewrite_meta_kernel_path_values(body: str, snapshot_prefix: str, dest_root: Path) -> str:
    """Replace SOC ``PATH_VALUES`` root embedded in a ``.tm`` with ``dest_root`` (POSIX paths)."""
    snap = snapshot_prefix.strip().rstrip("/")
    to = dest_root.expanduser().resolve().as_posix().rstrip("/")
    out = body.replace(snap + "/", to + "/")
    out = out.replace(snap, to)
    return out


def _shadowed_metakernel_for_furnsh(path: Path, site: SpiceSiteConfig) -> tuple[Path, Path | None]:
    """Return ``(path_for_furnsh, temp_copy_or_none)`` for ``furnsh``.

    When a temp file is returned, the caller must either register it in
    ``_shadow_meta_kernel_paths`` after all kernels load, or unlink it on failure.
    """
    if site.kernel_data_root is None or site.path_values_snapshot_prefix is None:
        return path, None
    if path.suffix.lower() != ".tm":
        return path, None
    body = path.read_text(encoding="utf-8", errors="replace")
    prefix = site.path_values_snapshot_prefix
    new_body = _rewrite_meta_kernel_path_values(body, prefix, site.kernel_data_root)
    if new_body == body:
        warnings.warn(
            f"No snapshot path prefix {prefix!r} found in meta-kernel {path.name!r}; "
            "furnishing the original file (PATH_VALUES unchanged).",
            stacklevel=2,
        )
        return path, None
    fd, name = tempfile.mkstemp(prefix="satsearch_mk_", suffix=".tm", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_body)
    except Exception:
        try:
            Path(name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return Path(name), Path(name)


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
    cfg = get_spice_runtime()
    site = get_config().spice
    staged: list[Path] = []
    try:
        for k in cfg.meta_kernels:
            if not k.is_file():
                raise FileNotFoundError(f"SPICE kernel not found: {k}")
            furnish_path, tmp = _shadowed_metakernel_for_furnsh(k, site)
            spice.furnsh(str(furnish_path))
            if tmp is not None:
                staged.append(tmp)
        _shadow_meta_kernel_paths.extend(staged)
        _kernels_loaded = True
    except Exception:
        for p in staged:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def reload_spice_kernels() -> None:
    """Clear CSPICE kernel pool so the next ``_furnsh_once`` loads :func:`satsearch_config.get_spice_runtime` kernels."""
    global _kernels_loaded
    _ensure_spiceypy()
    import spiceypy as spice

    with _lock:
        _purge_shadow_meta_kernels()
        if _kernels_loaded:
            try:
                spice.kclear()
            except Exception:
                pass
            _kernels_loaded = False


def _unload_kernels() -> None:
    global _kernels_loaded
    import spiceypy as spice

    _purge_shadow_meta_kernels()
    if _kernels_loaded:
        try:
            spice.kclear()
        except Exception:
            pass
        _kernels_loaded = False


atexit.register(_unload_kernels)


def et_from_midutcjd(path: Path) -> float:
    """Ephemeris seconds from ``MIDUTCJD`` in the FITS header (matches IDL ``lucy_finddj`` time)."""
    _ensure_spiceypy()
    import spiceypy as spice

    jd = _midutcjd_from_fits_path(path)
    if jd is None or not math.isfinite(float(jd)):
        raise ValueError(f"MIDUTCJD missing or invalid in {path.name}")
    jd = float(jd)
    # IDL: cspice_utc2et, string(time,format='(f25.7)')+' JD', et — not UNITIM(...,'JDT',...),
    # which is not portable (JDT is not a valid UNITIM input on many CSPICE builds).
    with _lock:
        _furnsh_once()
        et = spice.str2et(f"{jd:.7f} JD")
    return float(et)


def _parse_julian_day_scalar(value: object) -> float | None:
    """Parse FITS ``MIDUTCJD``-style values (duplicated from ``satsearch`` to avoid import cycles)."""
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


def _midutcjd_from_fits_path(path: Path) -> float | None:
    with fits.open(path, memmap=True) as hdul:
        h = hdul[0].header
    for k in ("MIDUTCJD", "midutcjd"):
        if k not in h:
            continue
        jd = _parse_julian_day_scalar(h[k])
        if jd is not None:
            return jd
    return None


def compute_closest_approach_et(
    et_start: float,
    *,
    step_s: float = 60.0,
    forward_hours: float = 24.0,
) -> float:
    """ET of minimum Lucy–target range over ``[et_start, et_start + forward_hours]`` (``step_s`` grid).

    Uses the active encounter's ``target_body`` and loaded kernels. Intended for one-day flyby
    windows anchored at the first FITS exposure in the encounter directory.
    """
    _ensure_spiceypy()
    import spiceypy as spice

    if not math.isfinite(et_start) or not math.isfinite(step_s) or step_s <= 0:
        raise ValueError("et_start and step_s must be finite and step_s > 0")
    t_end = et_start + max(0.0, forward_hours) * 3600.0
    best_et = et_start
    best_r = float("inf")
    with _lock:
        _furnsh_once()
        cfg = get_spice_runtime()
        t = et_start
        while t <= t_end + 1e-9:
            pos, _lt = spice.spkpos(
                cfg.target_body,
                t,
                cfg.frame,
                cfg.abcorr,
                cfg.observer_body,
            )
            r = float(np.linalg.norm(np.asarray(pos, dtype=np.float64).ravel()))
            if r < best_r:
                best_r = r
                best_et = t
            t += step_s
    return float(best_et)


def et_to_utc_iso(et: float, *, prec: int = 3) -> str:
    """UTC calendar string from ephemeris seconds (``ISOC`` via CSPICE)."""
    _ensure_spiceypy()
    import spiceypy as spice

    with _lock:
        _furnsh_once()
        s = spice.et2utc(et, "ISOC", prec)
    return str(s).strip()


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

    cfg = get_spice_runtime()
    et = et_from_midutcjd(path)
    with _lock:
        _furnsh_once()
        pos, _lt = spice.spkpos(
            cfg.target_body,
            et,
            cfg.frame,
            cfg.abcorr,
            cfg.observer_body,
        )
    return np.asarray(pos, dtype=np.float64).ravel()


def _lucy_sun_pos_km(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Target position (km) from Lucy and from Sun; one ET + two ``spkpos`` under the same lock."""
    _ensure_spiceypy()
    import spiceypy as spice

    cfg = get_spice_runtime()
    et = et_from_midutcjd(path)
    with _lock:
        _furnsh_once()
        pos, _lt = spice.spkpos(
            cfg.target_body,
            et,
            cfg.frame,
            cfg.abcorr,
            cfg.observer_body,
        )
        spos, _lt2 = spice.spkpos(
            cfg.target_body,
            et,
            cfg.frame,
            cfg.abcorr,
            cfg.sun_body,
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
