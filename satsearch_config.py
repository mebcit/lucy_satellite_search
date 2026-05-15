"""Load ``satsearch.toml`` and ``encounters.toml`` (site paths, refcat, SPICE kernel directory, per-encounter presets).

``satsearch.toml`` search order:

1. Path in ``SATSEARCH_CONFIG`` environment variable (absolute or relative to cwd).
2. ``satsearch.toml`` next to this file (package directory).

``encounters.toml`` search order:

1. Path in ``SATSEARCH_ENCOUNTERS`` (if set).
2. ``encounters.toml`` next to this file.

Encounter ``default_fits_directory`` entries are bare subdirectory names resolved under the parent
of ``[paths].default_fits_directory`` in ``satsearch.toml``.

There is no other discovery path and no in-code defaults for site paths: if either file is
missing, incomplete, or paths do not exist, loading raises immediately.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePath


@dataclass(frozen=True)
class SpiceSiteConfig:
    """``[spice]`` from ``satsearch.toml``: kernel directory + observer settings (no ``target_body``).

    Optional ``kernel_data_root``: when set, meta-kernel (``.tm``) bodies are rewritten at load time
    so ``PATH_VALUES`` entries pointing at ``path_values_snapshot_prefix`` aim at this directory
    instead (Lucy meta-kernels often ship with ``/mnt/lucy/soc/spice`` from the mission SOC).
    """

    meta_kernel_dir: Path
    observer_body: str
    sun_body: str
    frame: str
    abcorr: str
    kernel_data_root: Path | None
    path_values_snapshot_prefix: str | None


@dataclass(frozen=True)
class SpiceRuntime:
    """Resolved SPICE settings passed to ``lucy_spice`` (kernels + target from encounter)."""

    meta_kernels: tuple[Path, ...]
    target_body: str
    observer_body: str
    sun_body: str
    frame: str
    abcorr: str


@dataclass(frozen=True)
class RefcatConfig:
    """Atlas-refcat: binary and catalog data root (both required in ``[refcat]``)."""

    executable: Path
    catalog_dir: Path


@dataclass(frozen=True)
class PathsConfig:
    default_fits_directory: Path


@dataclass(frozen=True)
class SatsearchConfig:
    spice: SpiceSiteConfig
    refcat: RefcatConfig
    paths: PathsConfig


@dataclass(frozen=True)
class Encounter:
    """One flyby / target preset (``encounters.toml`` ``[[encounter]]``).

    ``meta_kernels`` stores resolved absolute paths (``meta_kernel_dir / basename`` at load time).
    ``default_fits_directory`` is resolved as the parent of ``[paths].default_fits_directory`` from
    ``satsearch.toml`` joined with the bare subdirectory name from the encounter table.
    ``closest_approach_utc`` is an optional ISO-8601 UTC string (written automatically on first load
    when missing; see :mod:`encounter_tca`).
    """

    id: str
    target_body: str
    meta_kernels: tuple[Path, ...]
    default_fits_directory: Path
    hill_sphere_km: float
    closest_approach_utc: str | None


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef, import-not-found]
    with path.open("rb") as f:
        return tomllib.load(f)


def config_path_candidates() -> list[Path]:
    env = os.environ.get("SATSEARCH_CONFIG", "").strip()
    out: list[Path] = []
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        out.append(p.resolve())
    out.append(_package_dir() / "satsearch.toml")
    return out


def _require_table(raw_path: Path, root: dict, name: str) -> dict:
    v = root.get(name)
    if not isinstance(v, dict):
        raise ValueError(
            f"{raw_path}: required [{name}] table is missing or invalid "
            f"(expected mapping, got {type(v).__name__})"
        )
    return v


def _require_nonempty_str(raw_path: Path, table: str, d: dict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{raw_path}: [{table}].{key!r} must be a non-empty string")
    return v.strip()


def validate_meta_kernel_basename(name: str) -> str:
    """Reject path-like strings; return a single bare file name (e.g. ``lcy.foo.LATEST.tm``)."""
    n = (name or "").strip()
    if not n or n in (".", ".."):
        raise ValueError("meta-kernel name must be non-empty")
    pp = PurePath(n)
    if len(pp.parts) != 1 or pp.name != n:
        raise ValueError(f"meta-kernel must be a bare file name (no directories), got {name!r}")
    return n


def validate_fits_directory_basename(name: str) -> str:
    """Reject path-like strings; return a single directory name (e.g. ``2025110``).

    Resolved as ``[paths].default_fits_directory``'s parent / ``name`` when loading encounters.
    """
    n = (name or "").strip()
    if not n or n in (".", ".."):
        raise ValueError("FITS directory name must be non-empty")
    pp = PurePath(n)
    if len(pp.parts) != 1 or pp.name != n:
        raise ValueError(
            f"default_fits_directory must be a bare directory name (no path separators), got {name!r}"
        )
    return n


def resolve_fits_directory_under_paths_default(paths_default_fits: Path, subdir: str) -> Path:
    """``paths_default_fits.parent / subdir``; result must exist, be a directory, and stay under parent."""
    default_dir = paths_default_fits.expanduser().resolve()
    if not default_dir.is_dir():
        raise FileNotFoundError(
            f"[paths].default_fits_directory is not an existing directory: {default_dir}"
        )
    root = default_dir.parent
    if not root.is_dir():
        raise FileNotFoundError(
            f"Parent of [paths].default_fits_directory is not an existing directory: {root}"
        )
    p = (root / subdir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"Encounter FITS directory not found: {root / subdir!s} -> {p} (resolved)"
        )
    try:
        p.relative_to(root.resolve())
    except ValueError as e:
        raise ValueError(
            f"Encounter FITS path must resolve under {root}, got {p}"
        ) from e
    return p


def resolve_meta_kernel_files(meta_kernel_dir: Path, basenames: tuple[str, ...]) -> tuple[Path, ...]:
    """``meta_kernel_dir / name`` for each basename; each result must be an existing file."""
    out: list[Path] = []
    d = meta_kernel_dir.expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"SPICE meta-kernel directory is not an existing directory: {d}")
    for bn in basenames:
        p = (d / bn).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Meta-kernel file not found under {d}: {bn} -> {p}")
        out.append(p)
    return tuple(out)


def load_satsearch_config() -> SatsearchConfig:
    raw_path: Path | None = None
    for p in config_path_candidates():
        if p.is_file():
            raw_path = p
            break
    if raw_path is None:
        raise FileNotFoundError(
            "No satsearch.toml found. Copy satsearch.toml.example to satsearch.toml "
            "next to satsearch_config.py, or set SATSEARCH_CONFIG to a TOML path. "
            "See satsearch.toml.example."
        )
    d = _load_toml(raw_path)
    sp = _require_table(raw_path, d, "spice")
    rc = _require_table(raw_path, d, "refcat")
    pa = _require_table(raw_path, d, "paths")

    mk_dir = Path(
        _require_nonempty_str(raw_path, "spice", sp, "meta_kernel_dir")
    ).expanduser().resolve()
    if not mk_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_path}: [spice].meta_kernel_dir is not an existing directory: {mk_dir}"
        )

    observer_body = _require_nonempty_str(raw_path, "spice", sp, "observer_body")
    sun_body = _require_nonempty_str(raw_path, "spice", sp, "sun_body")
    frame = _require_nonempty_str(raw_path, "spice", sp, "frame")
    abcorr = _require_nonempty_str(raw_path, "spice", sp, "abcorr")

    kdr_raw = sp.get("kernel_data_root")
    kernel_data_root: Path | None = None
    if kdr_raw is not None:
        if not isinstance(kdr_raw, str) or not kdr_raw.strip():
            raise ValueError(
                f"{raw_path}: [spice].kernel_data_root, if present, must be a non-empty string path"
            )
        kernel_data_root = Path(kdr_raw.strip()).expanduser().resolve()
        if not kernel_data_root.is_dir():
            raise FileNotFoundError(
                f"{raw_path}: [spice].kernel_data_root is not an existing directory: "
                f"{kernel_data_root}"
            )

    pvs_raw = sp.get("path_values_snapshot_prefix")
    path_values_snapshot_prefix: str | None = None
    if pvs_raw is not None:
        if not isinstance(pvs_raw, str) or not pvs_raw.strip():
            raise ValueError(
                f"{raw_path}: [spice].path_values_snapshot_prefix, if present, must be non-empty"
            )
        path_values_snapshot_prefix = pvs_raw.strip()
    if path_values_snapshot_prefix is not None and kernel_data_root is None:
        raise ValueError(
            f"{raw_path}: [spice].path_values_snapshot_prefix requires [spice].kernel_data_root"
        )
    if kernel_data_root is not None and path_values_snapshot_prefix is None:
        path_values_snapshot_prefix = "/mnt/lucy/soc/spice"

    spice = SpiceSiteConfig(
        meta_kernel_dir=mk_dir,
        observer_body=observer_body,
        sun_body=sun_body,
        frame=frame,
        abcorr=abcorr,
        kernel_data_root=kernel_data_root,
        path_values_snapshot_prefix=path_values_snapshot_prefix,
    )

    refcat_executable = Path(
        _require_nonempty_str(raw_path, "refcat", rc, "executable")
    ).expanduser().resolve()
    if not refcat_executable.is_file():
        raise FileNotFoundError(
            f"{raw_path}: [refcat].executable is not an existing file: {refcat_executable}"
        )

    refcat_catalog_dir = Path(
        _require_nonempty_str(raw_path, "refcat", rc, "catalog_dir")
    ).expanduser().resolve()
    if not refcat_catalog_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_path}: [refcat].catalog_dir is not an existing directory: {refcat_catalog_dir}"
        )

    refcat = RefcatConfig(executable=refcat_executable, catalog_dir=refcat_catalog_dir)

    default_fits_directory = Path(
        _require_nonempty_str(raw_path, "paths", pa, "default_fits_directory")
    ).expanduser().resolve()
    if not default_fits_directory.is_dir():
        raise FileNotFoundError(
            f"{raw_path}: [paths].default_fits_directory is not an existing directory: "
            f"{default_fits_directory}"
        )

    paths = PathsConfig(default_fits_directory=default_fits_directory)
    return SatsearchConfig(spice=spice, refcat=refcat, paths=paths)


_cached: SatsearchConfig | None = None


def get_config() -> SatsearchConfig:
    global _cached
    if _cached is None:
        _cached = load_satsearch_config()
    return _cached


def reload_config() -> SatsearchConfig:
    global _cached, _encounters
    _cached = load_satsearch_config()
    _encounters = None
    return _cached


# --- encounters.toml (per-target SPICE + FITS defaults + Hill radius) ---

_encounters: tuple[Encounter, ...] | None = None
_active_encounter_id: str | None = None
_meta_kernel_session_override: tuple[Path, ...] | None = None


def encounters_path_candidates() -> list[Path]:
    env = os.environ.get("SATSEARCH_ENCOUNTERS", "").strip()
    out: list[Path] = []
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        out.append(p.resolve())
    out.append(_package_dir() / "encounters.toml")
    return out


def encounters_toml_path() -> Path:
    """Resolved path to the loaded ``encounters.toml`` (first candidate that exists)."""
    for p in encounters_path_candidates():
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "No encounters.toml found. Copy encounters.toml.example to encounters.toml "
        "next to satsearch_config.py, or set SATSEARCH_ENCOUNTERS."
    )


def _load_encounters_toml(path: Path) -> dict:
    return _load_toml(path)


def _validate_closest_approach_utc(raw_path: Path, i: int, s: str) -> str:
    """Ensure ``closest_approach_utc`` is a parseable ISO-like UTC timestamp string."""
    from datetime import datetime

    t = s.strip()
    if not t:
        raise ValueError(f"{raw_path}: encounter[{i}].closest_approach_utc must be non-empty")
    probe = t.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(probe)
    except ValueError as e:
        raise ValueError(
            f"{raw_path}: encounter[{i}].closest_approach_utc must be ISO-8601-like UTC ({s!r})"
        ) from e
    return t


def load_encounters_from_disk() -> tuple[Encounter, ...]:
    raw_path: Path | None = None
    for p in encounters_path_candidates():
        if p.is_file():
            raw_path = p
            break
    if raw_path is None:
        raise FileNotFoundError(
            "No encounters.toml found. Copy encounters.toml.example to encounters.toml "
            "next to satsearch_config.py, or set SATSEARCH_ENCOUNTERS. "
            "See encounters.toml.example."
        )
    d = _load_encounters_toml(raw_path)
    items = d.get("encounter")
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(
            f"{raw_path}: expected a non-empty array of [[encounter]] tables "
            f"(got {type(items).__name__})"
        )
    cfg = get_config()
    site = cfg.spice
    mk_root = site.meta_kernel_dir
    paths_default_fits = cfg.paths.default_fits_directory
    seen: set[str] = set()
    out: list[Encounter] = []
    for i, block in enumerate(items):
        if not isinstance(block, dict):
            raise ValueError(f"{raw_path}: encounter[{i}] must be a table (mapping)")
        eid = _require_nonempty_str(raw_path, f"encounter[{i}]", block, "id")
        if eid in seen:
            raise ValueError(f"{raw_path}: duplicate encounter id {eid!r}")
        seen.add(eid)
        target_body = _require_nonempty_str(raw_path, f"encounter[{i}]", block, "target_body")

        mk = block.get("meta_kernels")
        if isinstance(mk, str):
            mk_list = [mk]
        elif isinstance(mk, list):
            mk_list = mk
        else:
            raise ValueError(
                f"{raw_path}: encounter[{i}].meta_kernels must be a string or non-empty list "
                f"(got {type(mk).__name__})"
            )
        if len(mk_list) == 0:
            raise ValueError(f"{raw_path}: encounter[{i}].meta_kernels must not be empty")
        basenames: list[str] = []
        for j, item in enumerate(mk_list):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"{raw_path}: encounter[{i}].meta_kernels[{j}] must be a non-empty string"
                )
            try:
                basenames.append(validate_meta_kernel_basename(item))
            except ValueError as e:
                raise ValueError(
                    f"{raw_path}: encounter[{i}].meta_kernels[{j}]: {e}"
                ) from e
        try:
            kernels = resolve_meta_kernel_files(mk_root, tuple(basenames))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{raw_path}: encounter[{i}]: {e}") from e

        fits_raw = _require_nonempty_str(raw_path, f"encounter[{i}]", block, "default_fits_directory")
        try:
            fits_sub = validate_fits_directory_basename(fits_raw)
        except ValueError as e:
            raise ValueError(f"{raw_path}: encounter[{i}].default_fits_directory: {e}") from e
        try:
            fits_dir = resolve_fits_directory_under_paths_default(paths_default_fits, fits_sub)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{raw_path}: encounter[{i}]: {e}") from e
        except ValueError as e:
            raise ValueError(f"{raw_path}: encounter[{i}]: {e}") from e

        hk = block.get("hill_sphere_km")
        if isinstance(hk, bool) or not isinstance(hk, (int, float)):
            raise ValueError(
                f"{raw_path}: encounter[{i}].hill_sphere_km must be a number (got {type(hk).__name__})"
            )
        hill = float(hk)
        if not math.isfinite(hill) or hill <= 0:
            raise ValueError(
                f"{raw_path}: encounter[{i}].hill_sphere_km must be a finite number > 0 (got {hill!r})"
            )

        ca_raw = block.get("closest_approach_utc")
        closest: str | None = None
        if ca_raw is not None:
            if not isinstance(ca_raw, str) or not ca_raw.strip():
                raise ValueError(
                    f"{raw_path}: encounter[{i}].closest_approach_utc must be a non-empty string"
                )
            closest = _validate_closest_approach_utc(raw_path, i, ca_raw)

        out.append(
            Encounter(
                id=eid,
                target_body=target_body,
                meta_kernels=kernels,
                default_fits_directory=fits_dir,
                hill_sphere_km=hill,
                closest_approach_utc=closest,
            )
        )
    return tuple(out)


def get_encounters() -> tuple[Encounter, ...]:
    global _encounters
    if _encounters is None:
        _encounters = load_encounters_from_disk()
        if any(e.closest_approach_utc is None for e in _encounters):
            import encounter_tca

            if encounter_tca.fill_missing_closest_approach(encounters_toml_path(), _encounters):
                _encounters = load_encounters_from_disk()
    return _encounters


def get_active_encounter() -> Encounter | None:
    if _active_encounter_id is None:
        return None
    for e in get_encounters():
        if e.id == _active_encounter_id:
            return e
    return None


def get_active_encounter_id() -> str | None:
    """Active encounter id from :func:`set_active_encounter_id`, or ``None`` before init."""
    return _active_encounter_id


def init_encounters_default() -> None:
    """Select the first encounter in ``encounters.toml`` as active (call after :func:`get_config`)."""
    global _active_encounter_id
    encs = get_encounters()
    if _active_encounter_id is None:
        _active_encounter_id = encs[0].id


def set_active_encounter_id(enc_id: str) -> None:
    """Switch active encounter; clears session meta-kernel override and reloads SPICE."""
    global _active_encounter_id, _meta_kernel_session_override
    enc_id = enc_id.strip()
    names = [e.id for e in get_encounters()]
    if enc_id not in names:
        raise ValueError(f"Unknown encounter id {enc_id!r}; valid: {names}")
    _meta_kernel_session_override = None
    _active_encounter_id = enc_id
    import lucy_spice

    lucy_spice.reload_spice_kernels()


def activate_encounter_spice_only(enc_id: str) -> None:
    """Set active encounter id and reload SPICE without calling :func:`get_encounters`.

    Used while building encounters (e.g. TCA auto-fill) when the encounters cache is not yet
    assigned. ``enc_id`` must match an encounter that exists in ``encounters.toml``.
    """
    global _active_encounter_id, _meta_kernel_session_override
    enc_id = enc_id.strip()
    _meta_kernel_session_override = None
    _active_encounter_id = enc_id
    import lucy_spice

    lucy_spice.reload_spice_kernels()


def canonical_meta_kernels() -> tuple[Path, ...]:
    """Resolved meta-kernel paths for the active encounter (from ``encounters.toml`` + ``meta_kernel_dir``)."""
    enc = get_active_encounter()
    if enc is None:
        raise RuntimeError("No active encounter")
    return enc.meta_kernels


def set_meta_kernel_session_override(paths: tuple[Path, ...] | None) -> None:
    """Override meta-kernel list for this process only (``None`` = use encounter / ``satsearch.toml``)."""
    global _meta_kernel_session_override
    if paths == _meta_kernel_session_override:
        return
    _meta_kernel_session_override = paths
    import lucy_spice

    lucy_spice.reload_spice_kernels()


def get_spice_runtime() -> SpiceRuntime:
    """Resolved SPICE for CSPICE (encounter ``target_body`` + kernels; optional session kernel path override)."""
    base = get_config().spice
    enc = get_active_encounter()
    if enc is None:
        raise RuntimeError("No active encounter")
    mk = enc.meta_kernels
    tb = enc.target_body
    if _meta_kernel_session_override is not None:
        mk = _meta_kernel_session_override
    return SpiceRuntime(
        meta_kernels=mk,
        target_body=tb,
        observer_body=base.observer_body,
        sun_body=base.sun_body,
        frame=base.frame,
        abcorr=base.abcorr,
    )


def default_hill_sphere_km() -> float:
    """Default Hill radius (km) from the active encounter, or 711 if encounters are not initialized."""
    enc = get_active_encounter()
    if enc is not None:
        return float(enc.hill_sphere_km)
    return 711.0
