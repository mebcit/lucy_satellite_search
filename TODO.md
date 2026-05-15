# Satsearch — working notes / to-do

Informal list of follow-ups and open questions. Not a release checklist.

---

## Stack — fake satellite geometry (open concern)

**Worry:** During the **Stack** procedure, are fake satellite positions really correct given **viewing geometry**? Are we using a **real (x, y, z) position in space** or always something **relative / simplified**?

**What the code does today (see `fullhill.run_fullhill_from_prep`):**

- **Across-image motion** is modeled in a **2D km plane**, not full 3D ephemeris:
  - `satx_km = satdist_km * cos(satang + dt * omega)`
  - `saty_km = satdist_km * sin(satang + dt * omega)`
  - with `dt` from `MIDUTCJD` differences vs the first frame, and `omega` from `omega_rad_per_s(satdist_km)` (circular-orbit-style angular rate from distance).
- That **(satx_km, saty_km)** pair is turned into **detector pixel offsets** with **`sx_pix = satx_km / prep.kpp[i]`** (and `sy_pix`), where **`kpp[i]`** is **km per native pixel** for that plane (built from SPICE range + plate scale in `run_fullhill_prepare`). So **range / plate scale enter the pixel mapping**, but the satellite path itself is still this **2D parametric disk**, not Lucy/target-centric 3D vectors per time.
- The PSF patch is **shifted in the image** with **`xyshift_cubic`** using offsets tied to **`djx`, `djy`** (alignment centroid vs 512.5) plus **`sx_pix`, `sy_pix`**. That is **image-plane / stack-aligned** geometry, not an independent sky-to-pixel projection of a 3D point each frame.

**To decide / verify:**

- [ ] Compare against the **IDL `fullhill.pro`** intent: confirm whether the original also used this **in-plane km + kpp** model, or something closer to full viewing geometry.
- [ ] Document clearly for users: fake satellite is **“toy orbit in a 2D km plane → pixels via per-frame kpp”**, **not** “integrate a 3D state in J2000 and project through the real camera model each exposure.”
- [ ] If science use needs **true 3D + camera projection**, scope a separate path (heavy): per-frame SPICE/camera ray or at least explicit parallax / roll terms beyond current `kpp` scaling.

---

## Other (placeholder)

- [ ] Add items here as you discover follow-ups (CI, packaging, docs, etc.).
