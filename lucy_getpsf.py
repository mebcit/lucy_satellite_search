"""
Python port of ``lucy_getpsf.pro`` for the ``psf=`` keyword path (normalized 15×15 PSF only).

**Intentional approximations / differences from IDL** (stated explicitly; not silent substitutes):

1. **``sky``** — IDL ``sky`` is not replicated. We use Astropy ``sigma_clipped_stats`` (median as sky,
   same family as elsewhere in this repo). Subtraction: ``im_work = im - median``.

2. **``find``** — IDL Astronomy ``find`` (``IDLAstro`` ``find.pro``) is approximated by
   ``photutils.DAOStarFinder``. We use ``threshold = 3.5 * std`` and ``fwhm=2.`` like
   ``lucy_getpsf.pro``. The call there is
   ``find,im,…,3.5,2.,[-1.5,-.5],[.4,1.]``; in IDL the next two vectors after ``fwhm`` are
   **``roundlim``** then **``sharplim``** (see ``find.pro``), i.e. roundness
   ``∈[-1.5,-0.5]`` and sharpness ``∈[0.4,1.0]``. Sharpness uses ``DAOStarFinder``’s
   ``sharplo``/``sharphi``. IDL exposes a **single** roundness statistic
   ``around = 2*(dx-dy)/(dx+dy)``, which matches Photutils **``roundness2``**; we apply the
   roundness band in a **post-filter** on ``roundness2`` only (Photutils would otherwise
   require the same interval for ``roundness1`` and ``roundness2``, which is stricter than IDL).

3. **Blend rejection** — Same geometric rule: discard sources that have another source within **10 px**
   (IDL loop marks blends; we keep brightest-first and drop neighbors within 10 px).

4. **``gauss2dfit``, ``/tilt``** — Implemented with ``scipy.optimize.curve_fit`` on the analytic
   tilted elliptical Gaussian below (not IDL’s MPFIT-based ``gauss2dfit``).

5. **Angle handling** — IDL adjusts ``a[6]`` with ``if a[6] lt !pi/2 then a[6]+=!pi`` etc.; we take
   the **median** of fitted ``theta`` values without that wrap, then fix ``θ`` in pass 2.

6. **First-guess amplitude** — IDL ``pim[5,5]`` / ``pim[7,7]`` (1-based vs mixed indexing). We use
   **0-based** ``pim[7, 7]`` for the center of a 15×15 patch.

7. **``psf = pim/total(pim)``** — IDL uses the **last** 15×15 **data** cutout from the second loop over
   the (up to) 20 brightest stars. We match that: last successful second-pass stamp, normalized.

8. **Source detection footprint** — ``DAOStarFinder`` runs on the **full** sky-subtracted frame
   (same idea as IDL ``FIND`` on the full image). No central-region exclusion; the brightest
   sources after blend/border filters drive the PSF (as in ``lucy_getpsf.pro``).

---

Input must match IDL: **sky still present** in ``im``; this routine subtracts sky internally before
FIND/fitting (same as ``lucy_getpsf.pro`` lines 11–12).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.optimize import OptimizeWarning, curve_fit

PATCH = 15
HALF = 7  # 15//2; cutout [yc-7:yc+8] × [xc-7:xc+8]
# ``curve_fit`` often emits OptimizeWarning when the covariance is ill-conditioned; fits are still OK.
_MAXFEV = 6000

# IDL ``find`` after ``fwhm``: ``roundlim`` then ``sharplim`` (``lucy_getpsf.pro``).
_FIND_SHARP_LO = 0.4
_FIND_SHARP_HI = 1.0
_FIND_ROUNDNESS2_LO = -1.5
_FIND_ROUNDNESS2_HI = -0.5


def _curve_fit_fit(*args: object, **kwargs: object):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return curve_fit(*args, **kwargs)


def _sky_subtract_like_idl(im: np.ndarray) -> tuple[np.ndarray, float]:
    _, median, _ = sigma_clipped_stats(
        np.asarray(im, dtype=np.float64),
        sigma=3.0,
        maxiters=5,
    )
    sky = float(median)
    return im.astype(np.float64) - sky, sky


def _filter_blends_brightest_first(
    x: np.ndarray,
    y: np.ndarray,
    flux: np.ndarray,
    min_sep_px: float = 10.0,
) -> np.ndarray:
    """Indices into ``x,y,flux`` to keep; brightest sources win over blends within ``min_sep_px``."""
    order = np.argsort(-flux)
    keep_idx: list[int] = []
    kept: list[tuple[float, float]] = []
    sep2 = min_sep_px * min_sep_px
    for i in order:
        xi, yi = float(x[i]), float(y[i])
        if all((xi - kx) ** 2 + (yi - ky) ** 2 >= sep2 for kx, ky in kept):
            keep_idx.append(int(i))
            kept.append((xi, yi))
    return np.array(keep_idx, dtype=np.int64)


def _filter_idl_find_roundness2(sources: object) -> object:
    """Keep rows whose ``roundness2`` lies in IDL ``find`` ``roundlim`` (see module doc)."""
    r2 = np.asarray(sources["roundness2"], dtype=np.float64)
    m = (r2 >= _FIND_ROUNDNESS2_LO) & (r2 <= _FIND_ROUNDNESS2_HI)
    return sources[m]


def _gauss2d_tilt(
    xy: tuple[np.ndarray, np.ndarray],
    bg: float,
    amp: float,
    sigx: float,
    sigy: float,
    x0: float,
    y0: float,
    theta: float,
) -> np.ndarray:
    xx, yy = xy
    c = np.cos(theta)
    s = np.sin(theta)
    u = (xx - x0) * c + (yy - y0) * s
    v = -(xx - x0) * s + (yy - y0) * c
    sx = max(abs(sigx), 1e-6)
    sy = max(abs(sigy), 1e-6)
    return bg + amp * np.exp(-0.5 * (u * u / (sx * sx) + v * v / (sy * sy)))


def _fit_pass1(pim: np.ndarray) -> np.ndarray | None:
    """Seven-parameter tilted Gaussian; sky-subtracted patch → bg should be ~0."""
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    flat = pim.ravel().astype(np.float64)
    amp0 = float(pim[7, 7]) - float(np.median(pim))
    p0 = [0.0, amp0, 1.6, 1.0, 7.0, 7.0, -0.36]
    lo = [-np.inf, -np.inf, 0.15, 0.15, 2.0, 2.0, -np.pi]
    hi = [np.inf, np.inf, 20.0, 20.0, 12.0, 12.0, np.pi]

    def model(xy, *a):
        return _gauss2d_tilt(xy, *a).ravel()

    try:
        popt, _ = _curve_fit_fit(
            model,
            (xx, yy),
            flat,
            p0=p0,
            bounds=(lo, hi),
            maxfev=_MAXFEV,
        )
    except (RuntimeError, ValueError):
        return None
    return popt


def _fit_pass2(
    pim: np.ndarray,
    sigx: float,
    sigy: float,
    theta: float,
) -> np.ndarray | None:
    """IDL: fix sky, σx, σy, θ; fit amplitude and center (fita [0,1,0,0,1,1,0] style on 7-tuple)."""
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    flat = pim.ravel().astype(np.float64)
    amp0 = float(pim[7, 7]) - float(np.median(pim))
    p0 = [0.0, amp0, 7.0, 7.0]

    def model2(xy, bg, amp, x0, y0):
        return _gauss2d_tilt(xy, bg, amp, sigx, sigy, x0, y0, theta).ravel()

    try:
        popt, _ = _curve_fit_fit(
            model2,
            (xx, yy),
            flat,
            p0=p0,
            maxfev=_MAXFEV,
        )
    except (RuntimeError, ValueError):
        return None
    return np.array(
        [popt[0], popt[1], sigx, sigy, popt[2], popt[3], theta],
        dtype=np.float64,
    )


@dataclass
class LucyGetpsfDebug:
    """PSF stamp and star centroids (full-frame, 0-based pixel coords) used in the pipeline."""

    psf: np.ndarray
    #: Stars with successful pass-1 Gaussian (set median σx, σy, θ).
    pass1_xy: list[tuple[float, float]]
    #: Stars with successful pass-2 fit (last one supplies the normalized PSF stamp).
    pass2_xy: list[tuple[float, float]]


def lucy_getpsf_debug(im: np.ndarray) -> LucyGetpsfDebug:
    """
    Same as :func:`lucy_getpsf`, plus subpixel centroids of stars that contributed to pass 1
    and pass 2 (for visualization).
    """
    im = np.asarray(im, dtype=np.float64)
    h, w = im.shape[:2]
    im_work, _sky = _sky_subtract_like_idl(im)

    _, _, std = sigma_clipped_stats(im_work, sigma=3.0, maxiters=5)
    rms = float(std) if std is not None and float(std) > 0 else 1.0
    thresh = 3.5 * rms

    daofind = DAOStarFinder(
        threshold=thresh,
        fwhm=2.0,
        sharplo=_FIND_SHARP_LO,
        sharphi=_FIND_SHARP_HI,
        roundlo=-2.0,
        roundhi=2.0,
        brightest=80,
        min_separation=10.0,
    )
    try:
        sources = daofind(im_work)
    except Exception as e:
        raise RuntimeError(f"lucy_getpsf: DAOStarFinder failed: {e}") from e

    if sources is None or len(sources) == 0:
        raise RuntimeError(
            "lucy_getpsf: no sources from DAOStarFinder (IDL FIND would also need detections)."
        )

    sources = _filter_idl_find_roundness2(sources)
    if sources is None or len(sources) == 0:
        raise RuntimeError(
            "lucy_getpsf: no sources after IDL roundness2 filter "
            f"[{_FIND_ROUNDNESS2_LO}, {_FIND_ROUNDNESS2_HI}]."
        )

    x = np.asarray(sources["xcentroid"], dtype=np.float64)
    y = np.asarray(sources["ycentroid"], dtype=np.float64)

    flux = np.asarray(sources["flux"], dtype=np.float64)

    blend_idx = _filter_blends_brightest_first(x, y, flux)
    x, y, flux = x[blend_idx], y[blend_idx], flux[blend_idx]

    border = (x > 7.0) & (x < 1016.0) & (y > 7.0) & (y < 1016.0)
    x, y, flux = x[border], y[border], flux[border]

    if len(flux) == 0:
        raise RuntimeError("lucy_getpsf: no sources after border/blend filters.")

    order = np.argsort(-flux)[: min(20, len(flux))]

    g1s: list[float] = []
    g2s: list[float] = []
    angs: list[float] = []
    pass1_xy: list[tuple[float, float]] = []

    for idx in order:
        xc = int(np.floor(float(x[idx]) + 0.5))
        yc = int(np.floor(float(y[idx]) + 0.5))
        if xc < HALF or yc < HALF or xc >= w - HALF or yc >= h - HALF:
            continue
        pim = im_work[yc - HALF : yc + HALF + 1, xc - HALF : xc + HALF + 1].copy()
        if pim.shape != (PATCH, PATCH):
            continue
        popt = _fit_pass1(pim)
        if popt is None:
            continue
        g1s.append(float(popt[2]))
        g2s.append(float(popt[3]))
        angs.append(float(popt[6]))
        pass1_xy.append((float(x[idx]), float(y[idx])))

    if not g1s:
        raise RuntimeError("lucy_getpsf: all first-pass Gaussian fits failed.")

    g1_med = float(np.median(np.array(g1s)))
    g2_med = float(np.median(np.array(g2s)))
    ang_med = float(np.median(np.array(angs)))

    last_pim: np.ndarray | None = None
    pass2_xy: list[tuple[float, float]] = []
    for idx in order:
        xc = int(np.floor(float(x[idx]) + 0.5))
        yc = int(np.floor(float(y[idx]) + 0.5))
        if xc < HALF or yc < HALF or xc >= w - HALF or yc >= h - HALF:
            continue
        pim = im_work[yc - HALF : yc + HALF + 1, xc - HALF : xc + HALF + 1].copy()
        if pim.shape != (PATCH, PATCH):
            continue
        p2 = _fit_pass2(pim, g1_med, g2_med, ang_med)
        if p2 is None:
            continue
        last_pim = pim.copy()
        pass2_xy.append((float(x[idx]), float(y[idx])))

    if last_pim is None:
        raise RuntimeError("lucy_getpsf: all second-pass Gaussian fits failed.")

    s = float(np.sum(last_pim))
    if s == 0.0:
        raise RuntimeError("lucy_getpsf: last PSF stamp sums to zero.")
    psf = (last_pim / s).astype(np.float64)
    return LucyGetpsfDebug(psf=psf, pass1_xy=pass1_xy, pass2_xy=pass2_xy)


def lucy_getpsf(im: np.ndarray) -> np.ndarray:
    """
    Return a **15×15** PSF, **sum-normalized to 1**, matching IDL ``psf=pim/total(pim)`` on the last
    star processed in the second loop.

    **Input:** single band, square science image (e.g. 1024×1024), **including sky** — same as IDL
    before ``lucy_getpsf`` subtracts sky internally.
    """
    return lucy_getpsf_debug(im).psf
