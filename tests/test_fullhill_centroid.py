"""Unit tests for ``centroid_bright_near_center`` (ROI → 3×3 median → peak → 30×30 patch centroid)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Import fullhill from package parent
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fullhill import centroid_bright_near_center  # noqa: E402


def gaussian_image(
    h: int,
    w: int,
    cx: float,
    cy: float,
    sigma: float,
    amp: float = 5000.0,
    bg: float = 100.0,
) -> np.ndarray:
    """Sampled 2D Gaussian + flat background (evaluated at pixel centers)."""
    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    return bg + amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))


class CentroidGaussianTests(unittest.TestCase):
    def assert_close_xy(
        self,
        got: tuple[float, float],
        expect: tuple[float, float],
        tol: float,
        msg: str = "",
    ) -> None:
        gx, gy = got
        ex, ey = expect
        self.assertLess(abs(gx - ex), tol, msg or f"x: got {gx} expect {ex}")
        self.assertLess(abs(gy - ey), tol, msg or f"y: got {gy} expect {ey}")

    def test_peak_at_geometric_center_1024(self) -> None:
        """Target at field center — ROI is centered here; should recover ~center."""
        h, w = 1024, 1024
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        im = gaussian_image(h, w, cx, cy, sigma=4.0)
        got = centroid_bright_near_center(im)
        # Weighted centroid of a sampled symmetric bump should be near the true peak.
        self.assert_close_xy(got, (cx, cy), tol=0.15)

    def test_subpixel_offset_near_center(self) -> None:
        """Peak a few pixels off geometric middle but still inside central ROI."""
        h, w = 1024, 1024
        cx, cy = 508.4, 519.75
        im = gaussian_image(h, w, cx, cy, sigma=5.0)
        got = centroid_bright_near_center(im)
        self.assert_close_xy(got, (cx, cy), tol=0.2)

    def test_narrow_peak_high_precision(self) -> None:
        """Tight Gaussian — centroid should still track the peak."""
        h, w = 1024, 1024
        cx, cy = 512.0, 511.0
        im = gaussian_image(h, w, cx, cy, sigma=2.0, amp=8000.0)
        got = centroid_bright_near_center(im)
        self.assert_close_xy(got, (cx, cy), tol=0.12)

    def test_far_from_center_roi_misses_peak(self) -> None:
        """Algorithm only searches a central ROI; a peak far from center is wrong by design."""
        h, w = 1024, 1024
        cx, cy = 120.0, 130.0
        im = gaussian_image(h, w, cx, cy, sigma=3.0)
        got = centroid_bright_near_center(im)
        # Should report something near the ROI center / local clutter, not (120, 130).
        self.assertGreater(math.hypot(got[0] - cx, got[1] - cy), 50.0)


if __name__ == "__main__":
    unittest.main()
