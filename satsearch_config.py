"""Load ``satsearch.toml`` (SPICE kernels, refcat, default FITS directory).

Search order:

1. Path in ``SATSEARCH_CONFIG`` environment variable (absolute or relative to cwd).
2. ``satsearch.toml`` next to this file (package directory).
3. ``satsearch.toml`` in the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpiceConfig:
    meta_kernels: tuple[Path, ...]
    target_body: str
    observer_body: str
    sun_body: str
    frame: str
    abcorr: str


@dataclass(frozen=True)
class PathsConfig:
    refcat_exe: Path
    refcat_dir: Path
    default_fits_directory: Path


@dataclass(frozen=True)
class SatsearchConfig:
    spice: SpiceConfig
    paths: PathsConfig


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
    out.append(Path.cwd() / "satsearch.toml")
    return out


def load_satsearch_config() -> SatsearchConfig:
    raw_path: Path | None = None
    for p in config_path_candidates():
        if p.is_file():
            raw_path = p
            break
    if raw_path is None:
        raise FileNotFoundError(
            "No satsearch.toml found. Copy satsearch.toml.example to satsearch.toml "
            "or set SATSEARCH_CONFIG. See satsearch.toml.example."
        )
    d = _load_toml(raw_path)
    sp = d.get("spice") or {}
    pa = d.get("paths") or {}
    mk = sp.get("meta_kernels") or []
    if isinstance(mk, str):
        mk = [mk]
    kernels = tuple(Path(str(x)).expanduser().resolve() for x in mk)
    spice = SpiceConfig(
        meta_kernels=kernels,
        target_body=str(sp.get("target_body", "DONALDJOHANSON")).strip(),
        observer_body=str(sp.get("observer_body", "LUCY")).strip(),
        sun_body=str(sp.get("sun_body", "SUN")).strip(),
        frame=str(sp.get("frame", "J2000")).strip(),
        abcorr=str(sp.get("abcorr", "NONE")).strip(),
    )
    paths = PathsConfig(
        refcat_exe=Path(str(pa["refcat_exe"])).expanduser().resolve(),
        refcat_dir=Path(str(pa["refcat_dir"])).expanduser().resolve(),
        default_fits_directory=Path(str(pa["default_fits_directory"])).expanduser().resolve(),
    )
    return SatsearchConfig(spice=spice, paths=paths)


_cached: SatsearchConfig | None = None


def get_config() -> SatsearchConfig:
    global _cached
    if _cached is None:
        _cached = load_satsearch_config()
    return _cached


def reload_config() -> SatsearchConfig:
    global _cached
    _cached = load_satsearch_config()
    return _cached
