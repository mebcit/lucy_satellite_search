#!/usr/bin/env python3
"""Debug ``lucy_getpsf``: save PSF stamp PNG and full-frame PNG with stars circled.

Uses the same display stretch as the Full hill viewer (:func:`sky_scale` + linear stretch).
Overlays use **matplotlib** in native pixel coordinates (column = x, row = y) so they match
Photutils ``xcentroid`` / ``ycentroid``. No central exclusion — same as the pipeline.

Usage: ``debug_lucy_psf.py [file.fit] [--show]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from satsearch import get_image_data_and_header, sky_scale
from fullhill import ensure_square_1024
from lucy_getpsf import lucy_getpsf_debug

_PSF_UP = 300
_CIRCLE_R_NATIVE = 8.0  # native pixels (same coord system as Photutils)


def _uniq_xy(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for a, b in points:
        key = (round(a, 4), round(b, 4))
        if key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--show"]
    do_show = "--show" in sys.argv[1:]
    if not args:
        print(
            "Usage: debug_lucy_psf.py <file.fit> [--show]\n"
            "Requires a FITS path argument (no default path; use satsearch.toml for app defaults).",
            file=sys.stderr,
        )
        sys.exit(2)
    path = Path(args[0]).resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    out_stars = path.with_name(path.stem + "_lucy_psf_stars.png")
    out_psf = path.with_name(path.stem + "_lucy_psf_stamp.png")

    data, _hdr = get_image_data_and_header(path)
    print(f"{path}: raw shape {data.shape}")
    im = ensure_square_1024(np.asarray(data, dtype=np.float64))
    h, w = im.shape[:2]
    print(f"  → PSF pipeline input: {w}×{h}")

    dbg = lucy_getpsf_debug(im)
    stars = _uniq_xy(dbg.pass1_xy + dbg.pass2_xy)
    print(
        f"  PSF {dbg.psf.shape[0]}×{dbg.psf.shape[1]}  sum={float(np.sum(dbg.psf)):.6f}  "
        f"pass1={len(dbg.pass1_xy)}  pass2={len(dbg.pass2_xy)}  unique circled={len(stars)}"
    )

    lo, hi = float(dbg.psf.min()), float(dbg.psf.max())
    if hi > lo:
        u8 = ((np.clip(dbg.psf, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        u8 = np.zeros(dbg.psf.shape, dtype=np.uint8)
    Image.fromarray(u8, mode="L").resize((_PSF_UP, _PSF_UP), Image.Resampling.NEAREST).save(
        out_psf
    )
    print(f"  → wrote {out_psf}")

    _, _, vmin, vmax = sky_scale(im)
    d = np.clip(im, vmin, vmax)
    d = (d - vmin) / (vmax - vmin)
    rgb = np.stack([d, d, d], axis=2)

    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=100)
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    for xf, yf in stars:
        ax.add_patch(
            Circle(
                (xf, yf),
                radius=_CIRCLE_R_NATIVE,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_stars, dpi=100, pad_inches=0)
    plt.close(fig)
    print(f"  → wrote {out_stars}")
    if do_show:
        Image.open(out_stars).show()


if __name__ == "__main__":
    main()
