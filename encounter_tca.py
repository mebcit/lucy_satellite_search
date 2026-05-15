"""Closest-approach (TCA) time for each ``[[encounter]]`` in ``encounters.toml``.

When ``closest_approach_utc`` is absent, the first load computes it from SPICE: the time of
minimum Lucy–target range on a 1-minute grid over the 24 hours beginning at the first sorted
FITS file's ``MIDUTCJD`` in the encounter's default directory. The result is written back to
``encounters.toml`` so later runs do not repeat the search.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from satsearch_config import Encounter

_FITS_SUFFIX = (".fits", ".fit", ".fts")


def first_sorted_fits(directory: Path) -> Path | None:
    """First FITS path in ``directory`` (lexicographic order, same as the thumbnail browser)."""
    if not directory.is_dir():
        return None
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _FITS_SUFFIX:
            return p
    return None


def fill_missing_closest_approach(
    encounters_toml: Path,
    encounters_tuple: tuple[Encounter, ...],
) -> bool:
    """Compute and persist missing ``closest_approach_utc`` entries. Returns True if file changed."""
    import tomllib

    import tomli_w

    import lucy_spice
    import satsearch_config as sc

    if not any(e.closest_approach_utc is None for e in encounters_tuple):
        return False

    data = tomllib.loads(encounters_toml.read_text(encoding="utf-8"))
    items = data.get("encounter")
    if not isinstance(items, list) or len(items) != len(encounters_tuple):
        raise ValueError(
            f"{encounters_toml}: encounter table length mismatch; refusing TCA auto-fill"
        )

    saved_id = sc.get_active_encounter_id()
    updated_any = False
    try:
        for i, enc in enumerate(encounters_tuple):
            if enc.closest_approach_utc is not None:
                continue
            block = items[i]
            if not isinstance(block, dict):
                continue
            sc.activate_encounter_spice_only(enc.id)
            p = first_sorted_fits(enc.default_fits_directory)
            if p is None:
                warnings.warn(
                    f"No FITS files in {enc.default_fits_directory}; skipping closest_approach_utc "
                    f"for encounter {enc.id!r}",
                    stacklevel=2,
                )
                continue
            try:
                et0 = lucy_spice.et_from_midutcjd(p)
                et_ca = lucy_spice.compute_closest_approach_et(et0)
                iso = lucy_spice.et_to_utc_iso(et_ca)
            except Exception as e:
                warnings.warn(
                    f"Could not compute closest_approach_utc for encounter {enc.id!r}: {e}",
                    stacklevel=2,
                )
                continue
            block["closest_approach_utc"] = iso
            updated_any = True
    finally:
        restore = saved_id if saved_id is not None else encounters_tuple[0].id
        sc.activate_encounter_spice_only(restore)

    if updated_any:
        text = tomli_w.dumps(data)
        tmp = encounters_toml.with_suffix(encounters_toml.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(encounters_toml)
    return updated_any
