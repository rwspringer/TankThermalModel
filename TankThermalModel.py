#!/usr/bin/env python3
from __future__ import annotations

# ---- Integrated directional shading/wind wrappers ----
def compute_shadow(elev_deg, az_deg, sector_az_deg, edge_position):
    return shadow_fraction_directional(elev_deg, az_deg, sector_az_deg, H, TANK_SPACING_EDGE_TO_EDGE, edge_position)

def compute_h(wind_speed, wind_dir_deg, sector_az_deg, edge_position):
    v_eff = effective_wind(wind_speed, wind_dir_deg, sector_az_deg, edge_position)
    return wall_h(v_eff)

"""
Azimuthal-sector water-tank freezing model with configurable edge position.

This script extends the earlier reduced-order tank models by resolving the tank
into multiple azimuthal wall sectors so that it can represent:
- directional solar heating of the cylindrical wall,
- configurable neighbor shading for edge-position cases,
- directional wind shelter/exposure using hourly wind direction,
- sector-by-sector wall ice and sector-by-sector surface ice,
- familiar plots/animations/CSV exports similar to the previous scripts.

It remains a reduced-order engineering model, not CFD.

The phase-change handling in this version explicitly conserves sensible and
latent energy at the top surface and wall interface, so melt energy is carried
through into water warming instead of being silently lost or double-counted.

This revision also exposes a simple material-stack model for the bladder and
wall-protection geotextile, and it uses a volume-aware wall-layer liquid heat
capacity so that sector-layers with thick wall ice do not keep the full liquid
thermal mass they had before freezing.
"""

import argparse
import csv
import gc
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import requests


# ------------------------- Tuning profile selector -------------------------
# Choose between:
#   "max_contrast"  – exaggerate shading/wind effects to make differences obvious
#   "realistic"     – more physically plausible, but still directional
TUNING_PROFILE = "max_contrast"  # or "realistic"


# Tunable Parameters
# --------------------------- Geometry and physics ---------------------------
R = 2.6
H = 4.1
N_Z_DEFAULT = 32
N_SECT_DEFAULT = 8

RHO_W = 1000.0
CP_W = 4180.0
K_W = 0.58
ALPHA_W = K_W / (RHO_W * CP_W)
RHO_I = 917.0
K_I = 2.2
L_F = 334e3
T_FREEZE = 0.0

RHO_A = 9e-6
GAP_HEIGHT = 0.40
K_SNOW = 0.30
RHO_SNOW = 300.0

DEFAULT_MEMBRANE_TOTAL_THICKNESS_M = 17.0 * 25.4e-6
DEFAULT_MEMBRANE_K = 0.22
DEFAULT_WALL_GEOTEXTILE_THICKNESS_M = 0.0025
DEFAULT_WALL_GEOTEXTILE_K = 0.12

ALPHA_ROOF = 0.85
EPS_ROOF = 0.90
SIGMA = 5.670374419e-8
GAP_AIR_HEAT_CAPACITY = 1.15 * 1005.0 * GAP_HEIGHT
H_GAP_ROOF = 2.5
H_GAP_WATER = 1.8
H_GAP_VENT = 0.35
ALPHA_WALL_SOLAR = 0.60
GROUND_ALBEDO_BARE = 0.20
GROUND_ALBEDO_SNOW = 0.65
WALL_VIEW_REDUCTION_PER_ROW = 0.12
WALL_GROUND_REDUCTION_PER_ROW = 0.08
WIND_WAKE_MAX_DEFICIT = 0.58
WIND_WAKE_RECOVERY_LENGTH_FACTOR = 1.75

T_GROUND_MEAN = 6.0
A_GROUND_ANNUAL = 2.0
GROUND_LAG_DAYS = 60.0
H_GROUND = 1.0

K_WIND_CLEAR = 2e-4
C_SNOW_MELT = 2e-7

MIN_TEMP_CLIP = -2.0
MAX_TEMP_CLIP = 40.0
MIN_LIQUID_CORE_RADIUS = 0.05
MAX_WALL_ICE_THICKNESS = R - MIN_LIQUID_CORE_RADIUS
C_ROOF_PER_AREA = 95000.0  # J/m²/K
H_WALL_WATER_ICE = 6.0     # W/m²/K

# Two-node radial liquid model (near-wall shell + interior core)
RADIAL_SHELL_THICKNESS_M = 0.18
RADIAL_SHELL_MIN_THICKNESS_M = 0.05
RADIAL_EXCHANGE_BASE = 1.5 # was 4.0
RADIAL_EXCHANGE_BUOY = 2.0 #was4.5
RADIAL_EXCHANGE_MAX = 10.0 # should be 12.0 # was  28.0
RADIAL_OVERTURN_MIX = 0.30

# Array-aware shielding defaults (base values; overridden by profile below)
TANK_SPACING_EDGE_TO_EDGE = 1.0
NEIGHBOR_CENTER_DISTANCE = 2 * R + TANK_SPACING_EDGE_TO_EDGE
ARRAY_PRIMARY_DEPTH = 4
ARRAY_SIDE_DEPTH = 2
ARRAY_SHADE_STRENGTH = 0.92
ARRAY_WIND_SHELTER_STRENGTH = 0.78
ROOF_SHADE_STRENGTH = 0.72
ROOF_WIND_SHELTER_STRENGTH = 0.55
ROW_SHADE_DECAY = 0.78
ROW_WIND_DECAY = 0.84
SUN_AZ_WIDTH_DEG = 34.0
SECTOR_AZ_WIDTH_DEG = 58.0
WIND_AZ_WIDTH_DEG = 52.0
ROOF_DIR_WIDTH_DEG = 44.0

# Time-stepping and clamps – profile dependent
if TUNING_PROFILE == "max_contrast":
    # Make directional effects as visible as possible
    MAX_SUB_DT = 300.0

    # Allow solar and wall fluxes to vary more before clipping
    Q_WALL_CLAMP = 800.0
    Q_TOP_CLAMP = 800.0
    Q_ROOF_CLAMP = 2000.0

    # Strongly suppress azimuthal and top mixing so sectors keep their identity
    #AZIMUTHAL_MIXING_FACTOR = 0.02
    AZIMUTHAL_MIXING_FACTOR = 0.00
    TOP_MIXING_FACTOR = 0.05

    # Make array shading and wind shelter stronger and deeper
    ARRAY_SHADE_STRENGTH = 1.25
    ARRAY_WIND_SHELTER_STRENGTH = 1.10
    ROW_SHADE_DECAY = 0.90
    ROW_WIND_DECAY = 0.90

elif TUNING_PROFILE == "realistic":
    # More conservative, but still directional
    MAX_SUB_DT = 300.0

    # Clamps high enough not to squash solar too often, but not crazy
    Q_WALL_CLAMP = 600.0
    Q_TOP_CLAMP = 600.0
    Q_ROOF_CLAMP = 1500.0

    # Some azimuthal and top mixing, but weaker than your original
    AZIMUTHAL_MIXING_FACTOR = 0.05
    TOP_MIXING_FACTOR = 0.20

    # Slightly softened array effects
    ARRAY_SHADE_STRENGTH = 0.90
    ARRAY_WIND_SHELTER_STRENGTH = 0.75
    ROW_SHADE_DECAY = 0.80
    ROW_WIND_DECAY = 0.85

else:
    raise ValueError(f"Unknown TUNING_PROFILE: {TUNING_PROFILE}")


# ------------------------------- Data classes ------------------------------
@dataclass
class Meteo:
    times: List[datetime]
    tair: np.ndarray
    wind: np.ndarray
    wind_dir: np.ndarray
    ghi: np.ndarray
    psfc: np.ndarray
    rh: np.ndarray
    snowfall_cm: np.ndarray
    snowfall_swe_mm: np.ndarray
    ground_snow_depth_m: np.ndarray
    precip: np.ndarray
    sun_elev_deg: np.ndarray
    sun_az_deg: np.ndarray
    # NEW (required)
    dni: np.ndarray
    dhi: np.ndarray



def save_meteo_cache(cache_path: Path | str, met: Meteo, *, lat: float, lon: float, start_date: str, end_date: str) -> None:
    """Write a compressed local cache of meteorology + derived solar geometry."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    times_utc: List[str] = []
    for t in met.times:
        if t.tzinfo is None:
            t_utc = t.replace(tzinfo=timezone.utc)
        else:
            t_utc = t.astimezone(timezone.utc)
        times_utc.append(t_utc.isoformat())

    np.savez_compressed(
        cache_path,
        cache_version=np.asarray([1], dtype=np.int32),
        lat=np.asarray([lat], dtype=float),
        lon=np.asarray([lon], dtype=float),
        start_date=np.asarray([start_date], dtype=str),
        end_date=np.asarray([end_date], dtype=str),
        times=np.asarray(times_utc, dtype=str),
        tair=np.asarray(met.tair, dtype=float),
        wind=np.asarray(met.wind, dtype=float),
        wind_dir=np.asarray(met.wind_dir, dtype=float),
        ghi=np.asarray(met.ghi, dtype=float),
        psfc=np.asarray(met.psfc, dtype=float),
        rh=np.asarray(met.rh, dtype=float),
        snowfall_cm=np.asarray(met.snowfall_cm, dtype=float),
        snowfall_swe_mm=np.asarray(met.snowfall_swe_mm, dtype=float),
        ground_snow_depth_m=np.asarray(met.ground_snow_depth_m, dtype=float),
        precip=np.asarray(met.precip, dtype=float),
        sun_elev_deg=np.asarray(met.sun_elev_deg, dtype=float),
        sun_az_deg=np.asarray(met.sun_az_deg, dtype=float),
    )



def load_meteo_cache(cache_path: Path | str, *, lat: float | None = None, lon: float | None = None,
                     start_date: str | None = None, end_date: str | None = None) -> Meteo:
    """Load a compressed local cache of meteorology + derived solar geometry."""
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        if lat is not None and 'lat' in data and abs(float(data['lat'][0]) - float(lat)) > 1e-6:
            raise ValueError(f'Meteorology cache latitude mismatch: cache={float(data["lat"][0])}, requested={lat}')
        if lon is not None and 'lon' in data and abs(float(data['lon'][0]) - float(lon)) > 1e-6:
            raise ValueError(f'Meteorology cache longitude mismatch: cache={float(data["lon"][0])}, requested={lon}')
        if start_date is not None and 'start_date' in data and str(data['start_date'][0]) != str(start_date):
            raise ValueError(f'Meteorology cache start_date mismatch: cache={str(data["start_date"][0])}, requested={start_date}')
        if end_date is not None and 'end_date' in data and str(data['end_date'][0]) != str(end_date):
            raise ValueError(f'Meteorology cache end_date mismatch: cache={str(data["end_date"][0])}, requested={end_date}')

        times: List[datetime] = []
        for ts in data['times'].astype(str).tolist():
            t = datetime.fromisoformat(str(ts))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            else:
                t = t.astimezone(timezone.utc)
            times.append(t)

        return Meteo(
            times=times,
            tair=np.asarray(data['tair'], dtype=float),
            wind=np.asarray(data['wind'], dtype=float),
            wind_dir=np.asarray(data['wind_dir'], dtype=float),
            ghi=np.asarray(data['ghi'], dtype=float),
            psfc=np.asarray(data['psfc'], dtype=float),
            rh=np.asarray(data['rh'], dtype=float),
            snowfall_cm=np.asarray(data['snowfall_cm'], dtype=float),
            snowfall_swe_mm=np.asarray(data['snowfall_swe_mm'], dtype=float),
            ground_snow_depth_m=np.asarray(data['ground_snow_depth_m'], dtype=float),
            precip=np.asarray(data['precip'], dtype=float),
            sun_elev_deg=np.asarray(data['sun_elev_deg'], dtype=float),
            sun_az_deg=np.asarray(data['sun_az_deg'], dtype=float),
        )



def load_or_fetch_openmeteo_archive(lat: float, lon: float, start_date: str, end_date: str,
                                    met_cache: str = '', refresh_met_cache: bool = False) -> Tuple[Meteo, str]:
    """Load meteorology from a local cache when available, else download and optionally write a cache."""
    cache_msg = ''
    if met_cache:
        cache_path = Path(met_cache)
        if cache_path.exists() and not refresh_met_cache:
            met = load_meteo_cache(cache_path, lat=lat, lon=lon, start_date=start_date, end_date=end_date)
            return met, f'Loaded meteorology from cache: {cache_path}'

        met = load_openmeteo_archive(lat, lon, start_date, end_date)
        save_meteo_cache(cache_path, met, lat=lat, lon=lon, start_date=start_date, end_date=end_date)
        if refresh_met_cache and cache_path.exists():
            cache_msg = f'Refreshed meteorology cache: {cache_path}'
        else:
            cache_msg = f'Downloaded meteorology and wrote cache: {cache_path}'
        return met, cache_msg

    met = load_openmeteo_archive(lat, lon, start_date, end_date)
    return met, 'Downloaded meteorology from Open-Meteo (no local cache requested)'


@dataclass(frozen=True)
class MaterialStack:
    membrane_total_thickness_m: float = DEFAULT_MEMBRANE_TOTAL_THICKNESS_M
    membrane_k: float = DEFAULT_MEMBRANE_K
    wall_geotextile_thickness_m: float = DEFAULT_WALL_GEOTEXTILE_THICKNESS_M
    wall_geotextile_k: float = DEFAULT_WALL_GEOTEXTILE_K


# ------------------------------- Utilities ---------------------------------
def _finite_minmax(*arrays: np.ndarray) -> Tuple[float, float]:
    vals: List[np.ndarray] = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return 0.0, 1.0
    merged = np.concatenate(vals)
    return float(np.min(merged)), float(np.max(merged))


def _expanded_limits(vmin: float, vmax: float, frac: float = 0.08, min_pad: float = 1e-6, include: Tuple[float, ...] = ()) -> Tuple[float, float]:
    vals = [vmin, vmax, *include]
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return -1.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        pad = max(abs(lo) * frac, min_pad, 0.5)
        return lo - pad, hi + pad
    pad = max((hi - lo) * frac, min_pad)
    return lo - pad, hi + pad


def _set_dynamic_ylim(ax, *arrays: np.ndarray, frac: float = 0.08, include: Tuple[float, ...] = ()) -> None:
    vmin, vmax = _finite_minmax(*arrays)
    ax.set_ylim(*_expanded_limits(vmin, vmax, frac=frac, include=include))


def rho_of_T(T_c: np.ndarray | float) -> np.ndarray | float:
    return RHO_W * (1.0 - RHO_A * (np.asarray(T_c) - 4.0) ** 2)


def h_roof_from_wind(v: float) -> float:
    return float(np.clip(5.7 + 3.8 * max(v, 0.0), 2.0, 60.0))


def h_wall_from_wind(v: float) -> float:
    return float(np.clip(4.0 + 2.8 * max(v, 0.0), 2.0, 45.0))


def sky_temperature(tair_c: float, rh_pct: float) -> float:
    t_air_k = tair_c + 273.15
    rh = np.clip(rh_pct / 100.0, 0.02, 1.0)
    es = 610.94 * np.exp(17.625 * tair_c / (tair_c + 243.04))
    ea = rh * es
    eps_sky = np.clip(1.24 * (ea / t_air_k) ** (1.0 / 7.0), 0.55, 0.98)
    return float((eps_sky ** 0.25) * t_air_k)


def ground_temperature(current_dt: datetime) -> float:
    year_start = datetime(current_dt.year, 1, 1, tzinfo=current_dt.tzinfo)
    day_of_year = (current_dt - year_start).total_seconds() / 86400.0
    phase = 2.0 * np.pi * (day_of_year - GROUND_LAG_DAYS) / 365.0
    return T_GROUND_MEAN + A_GROUND_ANNUAL * np.sin(phase)


def wrap180(angle_deg: np.ndarray | float) -> np.ndarray | float:
    a = (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0
    return a


def time_axis(times: List[datetime]) -> Tuple[np.ndarray, str]:
    t0 = times[0]
    span_days = (times[-1] - times[0]).total_seconds() / 86400.0
    if span_days <= 3:
        x = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        return x, "Hours since start"
    if span_days <= 60:
        x = np.array([(t - t0).total_seconds() / 86400.0 for t in times])
        return x, "Days since start"
    x = mdates.date2num(times)
    return x, "Date"


def month_ticks(times: List[datetime]) -> Tuple[np.ndarray, List[str]]:
    start = times[0]
    last = datetime(times[-1].year, times[-1].month, 1, tzinfo=times[-1].tzinfo)
    ticks = [mdates.date2num(start)]
    labels = [start.strftime("%b\n%Y")]

    if start.month == 12:
        cur = datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
    else:
        cur = datetime(start.year, start.month + 1, 1, tzinfo=start.tzinfo)

    while cur <= last:
        ticks.append(mdates.date2num(cur))
        labels.append(cur.strftime("%b\n%Y") if cur.month == 1 else cur.strftime("%b"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=cur.tzinfo)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=cur.tzinfo)
    return np.array(ticks), labels


def choose_animation_stride(times: List[datetime]) -> float:
    span_days = (times[-1] - times[0]).total_seconds() / 86400.0
    if span_days <= 7:
        return 3.0
    if span_days <= 60:
        return 12.0
    return 24.0


def membrane_resistance(stack: MaterialStack) -> float:
    return stack.membrane_total_thickness_m / max(stack.membrane_k, 1e-9)


def wall_geotextile_resistance(stack: MaterialStack) -> float:
    return stack.wall_geotextile_thickness_m / max(stack.wall_geotextile_k, 1e-9)


def wall_interface_solid_resistance(stack: MaterialStack) -> float:
    return membrane_resistance(stack) + wall_geotextile_resistance(stack)


def wall_liquid_heat_capacity_per_area(wice: np.ndarray | float) -> np.ndarray | float:
    r_liq = np.clip(R - np.asarray(wice, dtype=float), MIN_LIQUID_CORE_RADIUS, R)
    return RHO_W * CP_W * (r_liq ** 2) / (2.0 * R)


def radial_partition_per_wall_area(
    wice: np.ndarray | float,
    shell_thickness_m: float = RADIAL_SHELL_THICKNESS_M,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return shell/core heat capacities per unit wall area and shell-core interface area ratio.

    The liquid attached to each wall sector-layer is split into an outer shell and
    an inner core so the model can represent radial lag between wall forcing and
    the sector-average temperature.
    """
    w = np.asarray(wice, dtype=float)
    r_liq = np.clip(R - w, MIN_LIQUID_CORE_RADIUS, R)
    delta = np.minimum(shell_thickness_m, 0.45 * r_liq)
    delta = np.maximum(delta, np.minimum(RADIAL_SHELL_MIN_THICKNESS_M, 0.45 * r_liq))
    delta = np.clip(delta, 1e-4, np.maximum(r_liq - 1e-4, 1e-4))
    r_core = np.maximum(r_liq - delta, 0.0)

    shell_vol_per_area = (r_liq * delta - 0.5 * delta ** 2) / R
    core_vol_per_area = 0.5 * r_core ** 2 / R

    shell_cap = RHO_W * CP_W * shell_vol_per_area
    core_cap = RHO_W * CP_W * core_vol_per_area
    interface_area_ratio = np.clip((r_liq - 0.5 * delta) / R, 0.05, 1.0)
    return shell_cap, core_cap, delta, interface_area_ratio


def shell_core_mean_temperature(T_shell: np.ndarray | float, T_core: np.ndarray | float, wice: np.ndarray | float) -> np.ndarray | float:
    shell_cap, core_cap, _, _ = radial_partition_per_wall_area(wice)
    total_cap = shell_cap + core_cap
    return np.where(total_cap > 1e-12, (shell_cap * np.asarray(T_shell, dtype=float) + core_cap * np.asarray(T_core, dtype=float)) / total_cap, np.asarray(T_shell, dtype=float))


def radial_exchange_coefficient(
    T_shell: np.ndarray | float,
    T_core: np.ndarray | float,
    shell_thickness_m: np.ndarray | float,
) -> np.ndarray | float:
    dT = np.abs(np.asarray(T_shell, dtype=float) - np.asarray(T_core, dtype=float))
    h_cond = K_W / np.maximum(np.asarray(shell_thickness_m, dtype=float), 1e-6)
    h_eff = np.maximum(h_cond, RADIAL_EXCHANGE_BASE + RADIAL_EXCHANGE_BUOY * np.sqrt(dT))
    return np.clip(h_eff, h_cond, RADIAL_EXCHANGE_MAX)


def apply_shell_core_exchange(
    T_shell: float,
    T_core: float,
    wice: float,
    dt: float,
) -> Tuple[float, float, float]:
    shell_cap, core_cap, shell_thickness_m, interface_area_ratio = radial_partition_per_wall_area(float(wice))
    if core_cap <= 1e-12:
        return float(T_shell), float(T_core), 0.0
    h_eff = float(radial_exchange_coefficient(T_shell, T_core, shell_thickness_m))
    q_radial = h_eff * float(interface_area_ratio) * (float(T_shell) - float(T_core))
    q_radial = float(np.clip(q_radial, -2.0 * Q_WALL_CLAMP, 2.0 * Q_WALL_CLAMP))

    T_shell_new = float(T_shell) - q_radial * dt / max(float(shell_cap), 1e-9)
    T_core_new = float(T_core) + q_radial * dt / max(float(core_cap), 1e-9)
    return float(np.clip(T_shell_new, MIN_TEMP_CLIP, MAX_TEMP_CLIP)), float(np.clip(T_core_new, MIN_TEMP_CLIP, MAX_TEMP_CLIP)), q_radial


def apply_mean_shift_to_shell_core(T_shell: np.ndarray | float, T_core: np.ndarray | float, wice: np.ndarray | float, target_mean: np.ndarray | float) -> Tuple[np.ndarray | float, np.ndarray | float]:
    current_mean = shell_core_mean_temperature(T_shell, T_core, wice)
    delta = np.asarray(target_mean, dtype=float) - np.asarray(current_mean, dtype=float)
    return np.asarray(T_shell, dtype=float) + delta, np.asarray(T_core, dtype=float) + delta

def save_sector_energy_ice_plot(out: Dict[str, np.ndarray], plotpng: str) -> Path:
    """Save sector-wise cumulative wall solar, cumulative wall cooling, and final wall-ice plot."""
    base = Path(plotpng)
    theta_rad = np.deg2rad(out["theta_deg"])
    ang = np.degrees(theta_rad)
    cum_solar = np.asarray(out.get("cum_sector_wall_solar_J", np.zeros_like(ang)), dtype=float)
    cum_cooling = np.asarray(out.get("cum_sector_wall_cooling_J", np.zeros_like(ang)), dtype=float)
    cum_ambient = np.asarray(out.get("cum_sector_wall_ambient_J", np.zeros_like(ang)), dtype=float)
    cum_interface = np.asarray(out.get("cum_sector_wall_interface_J", np.zeros_like(ang)), dtype=float)
    final_wall_ice = np.mean(out["wall_ice"][-1], axis=1)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.8))
    ax1.plot(ang, cum_solar / 1e9, 'o-', label='Cumulative wall solar absorbed [GJ]')
    ax1.plot(ang, cum_ambient / 1e9, 'x-', label='Ambient-driven wall exchange [GJ]')
    ax1.plot(ang, cum_interface / 1e9, 's-', label='Wall-interface water flux [GJ]')
    ax1.plot(ang, cum_cooling / 1e9, 'd-', label='Wall cooling of water [GJ]')
    ax1.set_xlabel('Sector azimuth [deg, 0=N clockwise]')
    ax1.set_ylabel('Energy [GJ]')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(ang, final_wall_ice, 'k^-', label='Final average wall ice [m]')
    ax2.set_ylabel('Wall ice [m]')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
    ax1.set_title('Sector-wise wall forcing vs wall ice')
    p = base.with_name('sector_energy_vs_ice.png')
    fig.tight_layout(); fig.savefig(p, dpi=200); plt.close(fig)
    return p


def run_and_plot_per_sector_MJ_from_args(met, args, plotpng, material_stack=None):
    """
    Uses the runtime argument args.edge_position to run the model
    and plot per-sector cumulative MJ.

    Parameters
    ----------
    met : Meteo
        Meteorology object.
    args : argparse.Namespace
        Must contain: Tinit, n_z, n_sector, edge_position.
    material_stack : MaterialStack or None
        Optional material stack.
    """

    edge_position = args.edge_position  # <-- runtime parameter

    # --- Run the model ---
    out = run_model(
        met,
        T_init=args.Tinit,
        n_z=args.n_z,
        n_sector=args.n_sector,
        edge_position=edge_position,
        material_stack=material_stack,
    )
  
    # --- Compute cumulative MJ ---
    dt = (met.times[1] - met.times[0]).total_seconds()
    n_times, n_sectors = out["q_wall_eq_sector"].shape

    cum_wall_MJ   = np.cumsum(out["q_wall_eq_sector"] * dt / 1e6, axis=0)
    cum_top_MJ    = np.cumsum(out["q_top_sector"]      * dt / 1e6, axis=0)
    cum_bottom_MJ = np.cumsum(out["q_bottom_sector"]   * dt / 1e6, axis=0)

    # Time axis in days
    t = np.arange(n_times) * dt / 86400.0

    # --- Plot ---
    base = Path(plotpng)
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Wall
    for s in range(n_sectors):
        ax[0].plot(t, cum_wall_MJ[:, s], label=f"Sector {s}")
    ax[0].set_ylabel("Wall cumulative MJ")
    ax[0].set_title(f"{edge_position.capitalize()} edge — Wall")
    ax[0].grid(True)

    # Top
    for s in range(n_sectors):
        ax[1].plot(t, cum_top_MJ[:, s], label=f"Sector {s}")
    ax[1].set_ylabel("Top cumulative MJ")
    ax[1].set_title(f"{edge_position.capitalize()} edge — Top")
    ax[1].grid(True)

    # Bottom
    for s in range(n_sectors):
        ax[2].plot(t, cum_bottom_MJ[:, s], label=f"Sector {s}")
    ax[2].set_ylabel("Bottom cumulative MJ")
    ax[2].set_xlabel("Days since start")
    ax[2].set_title(f"{edge_position.capitalize()} edge — Bottom")
    ax[2].grid(True)

    ax[0].legend(ncol=4, fontsize=9)
    plt.tight_layout()
    p19 = base.with_name('sector_diagnostics_cumulative.png')
    fig.tight_layout(); fig.savefig(p19, dpi=200); plt.close(fig)

    save_sector_energy_ice_plot(out, plotpng)
    return out



# ---------------------------- Solar geometry -------------------------------
def equation_of_time_minutes(doy: int) -> float:
    """Approximate equation of time [minutes]."""
    B = np.deg2rad(360.0 * (doy - 81) / 364.0)
    return float(9.87 * np.sin(2.0 * B) - 7.53 * np.cos(B) - 1.50 * np.sin(B))


def solar_position_simple(times: List[datetime], lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Solar elevation and azimuth from UTC timestamps.

    Parameters
    ----------
    times : list[datetime]
        Timestamps. If timezone-aware they are converted to UTC internally.
        If naive they are assumed to already be UTC.
    lat_deg, lon_deg : float
        Latitude and longitude in degrees (+E, -W).

    Returns
    -------
    elev_deg, az_deg : ndarray
        Solar elevation and azimuth in degrees, with azimuth measured
        clockwise from North (0=N, 90=E, 180=S, 270=W).
    """
    lat = np.deg2rad(lat_deg)
    elev = np.zeros(len(times), dtype=float)
    az = np.zeros(len(times), dtype=float)

    for i, t in enumerate(times):
        if t.tzinfo is None:
            t_utc = t
        else:
            t_utc = t.astimezone(timezone.utc)

        doy = t_utc.timetuple().tm_yday
        utc_hour = t_utc.hour + t_utc.minute / 60.0 + t_utc.second / 3600.0

        # Local solar time from UTC, longitude, and equation of time.
        lst_hour = (utc_hour + lon_deg / 15.0 + equation_of_time_minutes(doy) / 60.0) % 24.0
        hra = np.deg2rad(15.0 * (lst_hour - 12.0))

        decl = np.deg2rad(23.45) * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))

        sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hra)
        sin_el = np.clip(sin_el, -1.0, 1.0)
        el = np.arcsin(sin_el)

        # Solar azimuth from north, clockwise, via local ENU components.
        east = -np.cos(decl) * np.sin(hra)
        north = np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(hra)
        azimuth = (np.rad2deg(np.arctan2(east, north)) + 360.0) % 360.0

        elev[i] = np.rad2deg(el)
        az[i] = azimuth

    return elev, az


def earth_sun_distance_factor(doy: np.ndarray | float) -> np.ndarray | float:
    """Earth-sun distance correction factor E0 for TOA irradiance."""
    doy_arr = np.asarray(doy, dtype=float)
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * doy_arr / 365.0)


def relative_air_mass_kasten_young(sun_elev_deg: np.ndarray | float) -> np.ndarray:
    """Relative optical air mass using the Kasten-Young 1989 formula."""
    elev = np.asarray(sun_elev_deg, dtype=float)
    zen = np.clip(90.0 - elev, 0.0, 90.0)
    m = np.full_like(elev, np.inf, dtype=float)
    mask = elev > 0.0
    if np.any(mask):
        z = zen[mask]
        m[mask] = 1.0 / (np.cos(np.deg2rad(z)) + 0.50572 * (96.07995 - z) ** -1.6364)
    return m


def ghi_caps_from_solar(times: List[datetime], sun_elev_deg: np.ndarray, psfc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return TOA horizontal and a conservative pressure-aware clear-sky GHI envelope.

    The clear-sky envelope is intentionally simple and conservative. It is meant
    as a sanity-check / clipping guide, not as a radiative-transfer model.
    """
    doy = np.asarray([t.timetuple().tm_yday for t in times], dtype=float)
    mu = np.clip(np.sin(np.deg2rad(np.asarray(sun_elev_deg, dtype=float))), 0.0, None)
    e0 = np.asarray(earth_sun_distance_factor(doy), dtype=float)
    toa_horizontal = 1361.0 * e0 * mu

    m_rel = relative_air_mass_kasten_young(sun_elev_deg)
    p_ratio = np.clip(np.asarray(psfc, dtype=float) / 101325.0, 0.15, 1.2)
    m_press = np.minimum(m_rel * p_ratio, 40.0)

    clear_horizontal = 1361.0 * e0 * mu * 0.82 * np.exp(-0.075 * m_press)
    clear_horizontal = np.minimum(clear_horizontal, toa_horizontal)
    clear_horizontal = np.where(mu > 0.0, clear_horizontal, 0.0)
    toa_horizontal = np.where(mu > 0.0, toa_horizontal, 0.0)
    return toa_horizontal, clear_horizontal


def apply_ghi_sanity_clip(met: Meteo, mode: str = 'off', clip_factor: float = 1.08) -> Meteo:
    """Attach GHI sanity-check arrays and optionally clip implausible hourly GHI."""
    raw = np.asarray(met.ghi, dtype=float).copy()
    toa_horizontal, clear_horizontal = ghi_caps_from_solar(met.times, met.sun_elev_deg, met.psfc)

    mode = str(mode).lower()
    if mode not in {'off', 'toa', 'clear'}:
        raise ValueError(f'Unsupported ghi clip mode: {mode}')

    if mode == 'toa':
        cap_used = toa_horizontal.copy()
    else:
        cap_used = np.minimum(toa_horizontal, np.maximum(0.0, float(clip_factor)) * clear_horizontal)

    if mode == 'off':
        used = raw.copy()
    else:
        used = np.minimum(raw, cap_used)

    met.ghi_raw = raw
    met.ghi_toa_horizontal = toa_horizontal
    met.ghi_clear_cap = clear_horizontal
    met.ghi_clip_cap = cap_used
    met.ghi_clip_mode = mode
    met.ghi_clip_factor = float(clip_factor)
    met.ghi_clip_mask = used < (raw - 1e-9)
    met.ghi = used
    return met


def ghi_clip_summary(met: Meteo) -> str:
    raw = np.asarray(getattr(met, 'ghi_raw', met.ghi), dtype=float)
    used = np.asarray(met.ghi, dtype=float)
    clear_cap = np.asarray(getattr(met, 'ghi_clear_cap', np.zeros_like(raw)), dtype=float)
    mask = used < (raw - 1e-9)
    n_clip = int(np.count_nonzero(mask))
    n = int(len(raw))
    max_raw = float(np.nanmax(raw)) if raw.size else 0.0
    max_used = float(np.nanmax(used)) if used.size else 0.0
    max_clear = float(np.nanmax(clear_cap)) if clear_cap.size else 0.0
    if n_clip > 0:
        max_reduction = float(np.nanmax(raw - used))
        return (f'GHI clip mode={getattr(met, "ghi_clip_mode", "off")}; clipped {n_clip}/{n} hours '
                f'({100.0*n_clip/max(n,1):.1f}%). Peak raw={max_raw:.1f} W/m², '
                f'peak used={max_used:.1f} W/m², max clear-sky cap={max_clear:.1f} W/m², '
                f'max reduction={max_reduction:.1f} W/m².')
    return (f'GHI clip mode={getattr(met, "ghi_clip_mode", "off")}; no hours clipped. '
            f'Peak raw={max_raw:.1f} W/m², peak used={max_used:.1f} W/m², '
            f'max clear-sky cap={max_clear:.1f} W/m².')

#Added Solar Utilities



# ------------------ Improved shading & wind ------------------
def shadow_fraction(elev_deg, H, spacing, blocked):
    if elev_deg <= 0:
        return 1.0
    if not blocked:
        return 0.0
    L = H / np.tan(np.deg2rad(elev_deg))
    return np.clip(L/spacing, 0.0, 1.0)

def effective_wind(v, wind_dir_deg, wall_az_deg, blocked):
    align = max(0.0, np.cos(np.deg2rad(wind_dir_deg - wall_az_deg)))
    shelter = 0.3 if blocked else 1.0
    return v * (0.3 + 0.7*align) * shelter

def wall_h(v_eff):
    return 5.0 + 2.4*v_eff

# ------------------------------- Meteorology -------------------------------
def load_openmeteo_archive(lat: float, lon: float, start_date: str, end_date: str) -> Meteo:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,"
        "shortwave_radiation,direct_radiation,diffuse_radiation,"
        "surface_pressure,relative_humidity_2m,snowfall,snow_depth,precipitation"
        "&wind_speed_unit=ms"
        "&timezone=UTC"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    times = [datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) for ts in hourly["time"]]

    wind = np.asarray(hourly["wind_speed_10m"], dtype=float)
    if np.nanmedian(wind) > 35.0:
        wind = wind / 3.6

    dni = np.asarray(hourly["direct_radiation"], dtype=float)
    dhi = np.asarray(hourly["diffuse_radiation"], dtype=float)

    sun_elev, sun_az = solar_position_simple(times, lat, lon)
    mu = np.maximum(np.sin(np.deg2rad(sun_elev)), 0.0)

    # recompute GHI consistently
    ghi = dni * mu + dhi

    snowfall_cm = np.asarray(hourly["snowfall"], dtype=float)

    met = Meteo(
        times=times,
        tair=np.asarray(hourly["temperature_2m"], dtype=float),
        wind=wind,
        wind_dir=np.asarray(hourly["wind_direction_10m"], dtype=float),
        ghi=ghi,
        psfc=np.asarray(hourly["surface_pressure"], dtype=float),
        rh=np.asarray(hourly["relative_humidity_2m"], dtype=float),
        snowfall_cm=snowfall_cm,
        snowfall_swe_mm=snowfall_cm / 7.0,
        ground_snow_depth_m=np.asarray(hourly["snow_depth"], dtype=float),
        precip=np.asarray(hourly["precipitation"], dtype=float),
        sun_elev_deg=sun_elev,
        sun_az_deg=sun_az,
        # NEW
        dni=dni,
        dhi=dhi,
    )

    # store DNI/DHI for later use
    met.dni = dni
    met.dhi = dhi

    return met

def load_openmeteo_archive_old(lat: float, lon: float, start_date: str, end_date: str) -> Meteo:
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation,"
        "surface_pressure,relative_humidity_2m,snowfall,snow_depth,precipitation"
        "&wind_speed_unit=ms"
        "&timezone=UTC"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    times = [datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) for ts in hourly["time"]]
    wind = np.asarray(hourly["wind_speed_10m"], dtype=float)
    if np.nanmedian(wind) > 35.0:
        wind = wind / 3.6
    sun_elev, sun_az = solar_position_simple(times, lat, lon)
    snowfall_cm = np.asarray(hourly["snowfall"], dtype=float)
    return Meteo(
        times=times,
        tair=np.asarray(hourly["temperature_2m"], dtype=float),
        wind=wind,
        wind_dir=np.asarray(hourly["wind_direction_10m"], dtype=float),
        ghi=np.asarray(hourly["shortwave_radiation"], dtype=float),
        psfc=np.asarray(hourly["surface_pressure"], dtype=float),
        rh=np.asarray(hourly["relative_humidity_2m"], dtype=float),
        snowfall_cm=snowfall_cm,
        snowfall_swe_mm=snowfall_cm / 7.0,
        ground_snow_depth_m=np.asarray(hourly["snow_depth"], dtype=float),
        precip=np.asarray(hourly["precipitation"], dtype=float),
        sun_elev_deg=sun_elev,
        sun_az_deg=sun_az,
    )


# ------------------------ Reduced-order sector model -----------------------
def convective_adjustment_column(T: np.ndarray, max_iter: int = 200, rho_tol: float = 1e-6) -> np.ndarray:
    T = T.copy().astype(float)
    nz = len(T)
    for _ in range(max_iter):
        rho = np.asarray(rho_of_T(T), dtype=float)
        unstable = np.where(rho[:-1] < rho[1:] - rho_tol)[0]
        if len(unstable) == 0:
            break
        j = int(unstable[0])
        start, end = j, j + 1
        mixed_T = float(np.mean(T[start:end + 1]))
        while True:
            mixed_rho = float(rho_of_T(mixed_T))
            expanded = False
            if end < nz - 1 and mixed_rho < float(rho_of_T(T[end + 1])) - rho_tol:
                end += 1
                mixed_T = float(np.mean(T[start:end + 1]))
                expanded = True
            if start > 0 and float(rho_of_T(T[start - 1])) < mixed_rho - rho_tol:
                start -= 1
                mixed_T = float(np.mean(T[start:end + 1]))
                expanded = True
            if not expanded:
                break
        T[start:end + 1] = mixed_T
    return T


def sector_angles_deg(n_sector: int) -> np.ndarray:
    return np.arange(n_sector) * (360.0 / n_sector)



def blocked_directions_with_rows(edge_position: str) -> Dict[float, int]:
    edge = edge_position.lower()
    if edge == 'isolated':
        return {}
    if edge == 'south':
        return {0.0: ARRAY_PRIMARY_DEPTH, 90.0: ARRAY_SIDE_DEPTH, 270.0: ARRAY_SIDE_DEPTH}
    if edge == 'north':
        return {180.0: ARRAY_PRIMARY_DEPTH, 90.0: ARRAY_SIDE_DEPTH, 270.0: ARRAY_SIDE_DEPTH}
    if edge == 'west':
        return {90.0: ARRAY_PRIMARY_DEPTH, 0.0: ARRAY_SIDE_DEPTH, 180.0: ARRAY_SIDE_DEPTH}
    if edge == 'east':
        return {270.0: ARRAY_PRIMARY_DEPTH, 0.0: ARRAY_SIDE_DEPTH, 180.0: ARRAY_SIDE_DEPTH}
    if edge == 'interior':
        return {0.0: ARRAY_PRIMARY_DEPTH, 90.0: ARRAY_PRIMARY_DEPTH, 180.0: ARRAY_PRIMARY_DEPTH, 270.0: ARRAY_PRIMARY_DEPTH}
    raise ValueError(f'Unsupported edge_position: {edge_position}')


def row_wall_gap_distance(row: int) -> float:
    return TANK_SPACING_EDGE_TO_EDGE + max(row - 1, 0) * (2.0 * R + TANK_SPACING_EDGE_TO_EDGE)


def ground_albedo_from_snow(ground_snow_depth_m: float) -> float:
    return float(GROUND_ALBEDO_SNOW if ground_snow_depth_m > 0.02 else GROUND_ALBEDO_BARE)


def _sector_face_weight(sector_az_deg: np.ndarray, blocked_az_deg: float) -> np.ndarray:
    return np.maximum(np.cos(np.deg2rad(wrap180(np.asarray(sector_az_deg, dtype=float) - blocked_az_deg))), 0.0)


def wall_solar_geometry_factors(
    sector_az_deg: np.ndarray,
    sun_az_deg: float,
    sun_elev_deg: float,
    edge_position: str,
) -> Dict[str, np.ndarray]:
    sector_az_deg = np.asarray(sector_az_deg, dtype=float)
    beam_proj = np.maximum(np.cos(np.deg2rad(wrap180(sector_az_deg - sun_az_deg))), 0.0)
    beam_proj *= max(np.cos(np.deg2rad(sun_elev_deg)), 0.0)

    direct_trans = np.ones_like(sector_az_deg, dtype=float)
    sky_view = np.ones_like(sector_az_deg, dtype=float)
    ground_view = np.ones_like(sector_az_deg, dtype=float)
    shaded_height = np.zeros_like(sector_az_deg, dtype=float)

    if sun_elev_deg <= 0.0:
        return {
            'beam_projection': np.zeros_like(sector_az_deg, dtype=float),
            'direct_transmission': np.zeros_like(sector_az_deg, dtype=float),
            'sky_view_factor': sky_view,
            'ground_view_factor': ground_view,
            'shaded_height_fraction': np.ones_like(sector_az_deg, dtype=float),
        }

    tan_elev = np.tan(np.deg2rad(max(sun_elev_deg, 1e-6)))
    blocked = blocked_directions_with_rows(edge_position)
    for blocked_az, n_rows in blocked.items():
        delta_sun = abs(float(wrap180(sun_az_deg - blocked_az)))
        if delta_sun >= 89.5:
            continue
        align = max(np.cos(np.deg2rad(delta_sun)), 0.0)
        if align <= 0.0:
            continue
        sector_weight = _sector_face_weight(sector_az_deg, blocked_az)
        for row in range(1, n_rows + 1):
            distance = row_wall_gap_distance(row)
            effective_distance = distance / max(align, 1e-6)
            shadow_height = max(H - effective_distance * tan_elev, 0.0)
            row_shadow_frac = float(np.clip(shadow_height / H, 0.0, 1.0))
            local_block = np.clip(row_shadow_frac * sector_weight, 0.0, 1.0)
            direct_trans *= (1.0 - local_block)
            shaded_height = 1.0 - (1.0 - shaded_height) * (1.0 - local_block)
            view_decay = math.exp(-distance / max(NEIGHBOR_CENTER_DISTANCE, 1e-6))
            sky_view -= WALL_VIEW_REDUCTION_PER_ROW * local_block * view_decay
            ground_view -= WALL_GROUND_REDUCTION_PER_ROW * local_block * view_decay

    return {
        'beam_projection': beam_proj,
        'direct_transmission': np.clip(direct_trans, 0.0, 1.0),
        'sky_view_factor': np.clip(sky_view, 0.35, 1.0),
        'ground_view_factor': np.clip(ground_view, 0.45, 1.0),
        'shaded_height_fraction': np.clip(shaded_height, 0.0, 1.0),
    }



def wall_solar_components(
    sector_az_deg: np.ndarray,
    sun_az_deg: float,
    sun_elev_deg: float,
    dni: float,
    dhi: float,
    ground_snow_depth_m: float,
    edge_position: str,
) -> Dict[str, np.ndarray]:
    geom = wall_solar_geometry_factors(
    sector_az_deg, sun_az_deg, sun_elev_deg, edge_position
)
    ground_albedo = ground_albedo_from_snow(ground_snow_depth_m)

    mu = max(np.sin(np.deg2rad(sun_elev_deg)), 0.0)
    ghi_local = dni * mu + dhi

    # --- Direct beam on vertical wall ---
    q_direct = ALPHA_WALL_SOLAR * dni * geom['beam_projection'] * geom['direct_transmission']

    # --- Diffuse sky ---
    q_diffuse = ALPHA_WALL_SOLAR * 0.5 * dhi * geom['sky_view_factor']

    # --- Ground reflected ---
    q_ground = ALPHA_WALL_SOLAR * 0.5 * ground_albedo * ghi_local * geom['ground_view_factor']

    q_total = q_direct + q_diffuse + q_ground

    return {
        'q_direct': q_direct,
        'q_diffuse': q_diffuse,
        'q_ground': q_ground,
        'q_total': q_total,
        **geom,
    }



def sector_wind_speed_factor(sector_az_deg: np.ndarray, wind_dir_deg: float, edge_position: str) -> np.ndarray:
    sector_az_deg = np.asarray(sector_az_deg, dtype=float)
    blocked = blocked_directions_with_rows(edge_position)
    cumulative_deficit = 0.0
    recovery_length = max(WIND_WAKE_RECOVERY_LENGTH_FACTOR * NEIGHBOR_CENTER_DISTANCE, 2.0 * H)
    for blocked_az, n_rows in blocked.items():
        delta = abs(float(wrap180(wind_dir_deg - blocked_az)))
        if delta >= 89.5:
            continue
        align = max(np.cos(np.deg2rad(delta)), 0.0) ** 1.5
        for row in range(1, n_rows + 1):
            distance = row_wall_gap_distance(row)
            row_deficit = WIND_WAKE_MAX_DEFICIT * align * math.exp(-distance / max(recovery_length, 1e-6))
            cumulative_deficit = 1.0 - (1.0 - cumulative_deficit) * (1.0 - row_deficit)
    wake_factor = float(np.clip(1.0 - cumulative_deficit, 0.12, 1.0))

    cos_rel = np.cos(np.deg2rad(wrap180(sector_az_deg - wind_dir_deg)))
    local_factor = np.where(cos_rel >= 0.0, 0.45 + 0.55 * cos_rel, 0.18 + 0.10 * (-cos_rel))
    return np.clip(wake_factor * local_factor, 0.05, 1.10)


def roof_wake_wind_factor(wind_dir_deg: float, edge_position: str) -> float:
    blocked = blocked_directions_with_rows(edge_position)
    cumulative_deficit = 0.0
    recovery_length = max(WIND_WAKE_RECOVERY_LENGTH_FACTOR * NEIGHBOR_CENTER_DISTANCE, 2.0 * H)
    for blocked_az, n_rows in blocked.items():
        delta = abs(float(wrap180(wind_dir_deg - blocked_az)))
        if delta >= 89.5:
            continue
        align = max(np.cos(np.deg2rad(delta)), 0.0) ** 1.5
        for row in range(1, n_rows + 1):
            distance = row_wall_gap_distance(row)
            row_deficit = 0.85 * WIND_WAKE_MAX_DEFICIT * align * math.exp(-distance / max(recovery_length, 1e-6))
            cumulative_deficit = 1.0 - (1.0 - cumulative_deficit) * (1.0 - row_deficit)
    return float(np.clip(1.0 - cumulative_deficit, 0.18, 1.0))


def _gaussian_align(delta_deg: np.ndarray | float, width_deg: float) -> np.ndarray | float:
    d = np.asarray(wrap180(delta_deg), dtype=float)
    return np.exp(-(d / max(width_deg, 1e-6)) ** 2)


def neighbor_shade_factor(sector_az_deg: np.ndarray, sun_az_deg: float, sun_elev_deg: float, edge_position: str) -> np.ndarray:
    if sun_elev_deg <= 0.0:
        return np.ones_like(sector_az_deg, dtype=float)
    blocked = blocked_directions_with_rows(edge_position)
    sector_block = np.zeros_like(sector_az_deg, dtype=float)
    for blocked_az, n_rows in blocked.items():
        sector_facing = _gaussian_align(sector_az_deg - blocked_az, SECTOR_AZ_WIDTH_DEG)
        sun_align = float(_gaussian_align(sun_az_deg - blocked_az, SUN_AZ_WIDTH_DEG))
        for row in range(1, n_rows + 1):
            crit_elev = np.rad2deg(np.arctan2(H, row * NEIGHBOR_CENTER_DISTANCE))
            low_sun = np.clip(1.0 - sun_elev_deg / max(crit_elev, 1e-6), 0.0, 1.0)
            sector_block += ARRAY_SHADE_STRENGTH * (ROW_SHADE_DECAY ** (row - 1)) * sun_align * low_sun * sector_facing
    return np.clip(1.0 - sector_block, 0.02, 1.0)


def directional_wind_shelter_factor(sector_az_deg: np.ndarray, wind_dir_deg: float, edge_position: str) -> np.ndarray:
    # wind_dir is meteorological direction FROM which wind comes.
    blocked = blocked_directions_with_rows(edge_position)
    sector_block = np.zeros_like(sector_az_deg, dtype=float)
    for blocked_az, n_rows in blocked.items():
        sector_facing = _gaussian_align(sector_az_deg - blocked_az, SECTOR_AZ_WIDTH_DEG)
        wind_align = float(_gaussian_align(wind_dir_deg - blocked_az, WIND_AZ_WIDTH_DEG))
        for row in range(1, n_rows + 1):
            sector_block += ARRAY_WIND_SHELTER_STRENGTH * (ROW_WIND_DECAY ** (row - 1)) * wind_align * sector_facing
    return np.clip(1.0 - sector_block, 0.08, 1.2)


def roof_solar_shade_factor(sun_az_deg: float, sun_elev_deg: float, edge_position: str) -> float:
    if sun_elev_deg <= 0.0:
        return 1.0
    roof_block = 0.0
    for blocked_az, n_rows in blocked_directions_with_rows(edge_position).items():
        sun_align = float(_gaussian_align(sun_az_deg - blocked_az, ROOF_DIR_WIDTH_DEG))
        for row in range(1, n_rows + 1):
            crit_elev = np.rad2deg(np.arctan2(H, row * NEIGHBOR_CENTER_DISTANCE))
            low_sun = np.clip(1.0 - sun_elev_deg / max(crit_elev, 1e-6), 0.0, 1.0)
            roof_block += ROOF_SHADE_STRENGTH * (ROW_SHADE_DECAY ** (row - 1)) * sun_align * low_sun
    return float(np.clip(1.0 - roof_block, 0.03, 1.0))


def roof_wind_shelter_factor(wind_dir_deg: float, edge_position: str) -> float:
    roof_block = 0.0
    for blocked_az, n_rows in blocked_directions_with_rows(edge_position).items():
        wind_align = float(_gaussian_align(wind_dir_deg - blocked_az, ROOF_DIR_WIDTH_DEG))
        for row in range(1, n_rows + 1):
            roof_block += ROOF_WIND_SHELTER_STRENGTH * (ROW_WIND_DECAY ** (row - 1)) * wind_align
    return float(np.clip(1.0 - roof_block, 0.12, 1.0))


def diffuse_vertical(T: np.ndarray, dt: float, dz: float) -> np.ndarray:
    r = ALPHA_W * dt / dz**2
    out = T.copy()
    out[:, 1:-1] = T[:, 1:-1] + r * (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2])
    return out


def diffuse_azimuthal(T: np.ndarray, dt: float, r_mid: float, dtheta: float) -> np.ndarray:
    alpha_theta = AZIMUTHAL_MIXING_FACTOR * ALPHA_W
    coeff = alpha_theta * dt / max((r_mid * dtheta) ** 2, 1e-9)
    return T + coeff * (np.roll(T, -1, axis=0) - 2.0 * T + np.roll(T, 1, axis=0))


def apply_top_mixing(T: np.ndarray) -> np.ndarray:
    out = T.copy()
    mean_top = np.mean(out[:, -1])
    out[:, -1] = (1.0 - TOP_MIXING_FACTOR) * out[:, -1] + TOP_MIXING_FACTOR * mean_top
    return out


def apply_top_mixing_shell_core(T_shell: np.ndarray, T_core: np.ndarray, wice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_shell = T_shell.copy()
    out_core = T_core.copy()
    top_mean = shell_core_mean_temperature(out_shell[:, -1], out_core[:, -1], wice[:, -1])
    mixed_top = (1.0 - TOP_MIXING_FACTOR) * top_mean + TOP_MIXING_FACTOR * float(np.mean(top_mean))
    new_shell, new_core = apply_mean_shift_to_shell_core(out_shell[:, -1], out_core[:, -1], wice[:, -1], mixed_top)
    out_shell[:, -1] = np.asarray(new_shell, dtype=float)
    out_core[:, -1] = np.asarray(new_core, dtype=float)
    return out_shell, out_core


def update_surface_phase_sector(Ttop: float, zice: float, q_into_water: float, dt: float, dz: float) -> Tuple[float, float]:
    """Advance top-layer temperature and surface-ice thickness conservatively.

    The update is done per unit top area. When surface ice is present, the
    interface is held at the freezing point while net energy first melts/grows
    the ice. Any heat left after complete melt is carried into warming the top
    water layer, instead of being discarded.
    """
    heat_capacity_per_area = RHO_W * CP_W * dz
    latent_per_area_per_m = RHO_I * L_F
    net_energy = heat_capacity_per_area * Ttop + q_into_water * dt

    if zice > 0.0:
        if net_energy >= 0.0:
            melt = min(zice, net_energy / latent_per_area_per_m)
            z_new = zice - melt
            net_energy -= melt * latent_per_area_per_m
            if z_new > 0.0:
                return T_FREEZE, float(z_new)
            T_new = net_energy / heat_capacity_per_area
            return float(np.clip(T_new, MIN_TEMP_CLIP, MAX_TEMP_CLIP)), 0.0
        z_new = zice + (-net_energy) / latent_per_area_per_m
        return T_FREEZE, float(max(0.0, z_new))

    if net_energy >= 0.0:
        T_new = net_energy / heat_capacity_per_area
        return float(np.clip(T_new, MIN_TEMP_CLIP, MAX_TEMP_CLIP)), 0.0

    z_new = (-net_energy) / latent_per_area_per_m
    return T_FREEZE, float(max(0.0, z_new))


def update_wall_phase_cell(
    Tcell: float,
    wice: float,
    q_ext: float,
    dt: float,
    water_heat_capacity_per_wall_area: float,
    h_water_ice: float,
    max_wice: float,
) -> Tuple[float, float, float, float]:
    """Advance one sector-layer wall cell and its attached wall ice.

    Returns
    -------
    Tnew, wice_new, q_liquid_interface, q_latent_storage
        Updated bulk-water temperature, updated wall-ice thickness, liquid-side
        wall/interface flux (positive when heat leaves liquid toward the wall),
        and latent storage rate in wall ice (positive for ice growth, negative
        for melting), all per unit wall area.
    """
    latent_per_area_per_m = RHO_I * L_F
    q_latent = 0.0

    if wice <= 0.0:
        net_energy = water_heat_capacity_per_wall_area * Tcell - q_ext * dt
        if net_energy >= 0.0:
            Tnew = net_energy / water_heat_capacity_per_wall_area
            return float(np.clip(Tnew, MIN_TEMP_CLIP, MAX_TEMP_CLIP)), 0.0, float(q_ext), 0.0
        wice_new = min((-net_energy) / latent_per_area_per_m, max_wice)
        q_latent = float((wice_new - wice) * latent_per_area_per_m / max(dt, 1e-9))
        q_liquid_interface = float(q_ext - q_latent)
        return T_FREEZE, float(max(0.0, wice_new)), q_liquid_interface, q_latent

    q_water_ice = float(np.clip(h_water_ice * (Tcell - T_FREEZE), -Q_WALL_CLAMP, Q_WALL_CLAMP))
    water_energy = water_heat_capacity_per_wall_area * Tcell - q_water_ice * dt
    phase_energy = (q_ext - q_water_ice) * dt
    wice_new = wice + phase_energy / latent_per_area_per_m

    if wice_new < 0.0:
        water_energy += (-wice_new) * latent_per_area_per_m
        wice_new = 0.0
    elif wice_new > max_wice:
        water_energy -= (wice_new - max_wice) * latent_per_area_per_m
        wice_new = max_wice

    if wice_new > 0.0 and water_energy < 0.0:
        extra_freeze = min((-water_energy) / latent_per_area_per_m, max_wice - wice_new)
        wice_new += extra_freeze
        water_energy += extra_freeze * latent_per_area_per_m
        if water_energy < 0.0:
            water_energy = 0.0

    q_latent = float((wice_new - wice) * latent_per_area_per_m / max(dt, 1e-9))
    q_liquid_interface = float(q_ext - q_latent)
    Tnew = water_energy / water_heat_capacity_per_wall_area
    Tnew = float(np.clip(Tnew, T_FREEZE if wice_new > 0.0 else MIN_TEMP_CLIP, MAX_TEMP_CLIP))
    return Tnew, float(max(0.0, wice_new)), q_liquid_interface, q_latent


def make_energy_budget(out: Dict[str, np.ndarray], dt_seconds: float) -> Dict[str, np.ndarray]:
    sector_area = np.asarray(out["sector_top_area"], dtype=float)
    if sector_area.ndim == 1:
        sector_area = np.broadcast_to(sector_area, out["q_top_sector"].shape)

#Fix incorrect wall area calculation
    wall_area_factor = 2.0 * H / R
    wall_sector_area = sector_area * wall_area_factor

    top_area_total = float(out["top_area_total"])
    top_power_W = np.sum(out["q_top_sector"] * sector_area, axis=1)
    wall_external_power_W = np.sum(out["q_wall_eq_sector"] * sector_area, axis=1)
    wall_ambient_power_W = np.sum(out["q_wall_ambient_sector"] * sector_area, axis=1)
    wall_interface_power_W = np.sum(out["q_wall_interface_sector"] * sector_area, axis=1)
    wall_latent_power_W = np.sum(out["q_wall_ice_latent_sector"] * sector_area, axis=1)
#Fix incorrect wall area calculation
#   wall_solar_power_W = np.sum(out["q_wall_solar_sector"] * sector_area, axis=1)
#   wall_solar_direct_power_W = np.sum(out["q_wall_solar_direct_sector"] * sector_area, axis=1)
#   wall_solar_diffuse_power_W = np.sum(out["q_wall_solar_diffuse_sector"] * sector_area, axis=1)
#   wall_solar_ground_power_W = np.sum(out["q_wall_solar_ground_sector"] * sector_area, axis=1)
    wall_solar_power_W = np.sum(out["q_wall_solar_sector"] * wall_sector_area, axis=1)
    wall_solar_direct_power_W = np.sum(out["q_wall_solar_direct_sector"] * wall_sector_area, axis=1)
    wall_solar_diffuse_power_W = np.sum(out["q_wall_solar_diffuse_sector"] * wall_sector_area, axis=1)
    wall_solar_ground_power_W = np.sum(out["q_wall_solar_ground_sector"] * wall_sector_area, axis=1)
  
    bottom_power_W = np.sum(out["q_bottom_sector"] * sector_area, axis=1)
    gap_power_W = np.sum(out["q_gap_water_sector"] * sector_area, axis=1)
    roof_gap_power_W = np.asarray(out["q_roof_gap"], dtype=float) * top_area_total
    gap_vent_power_W = np.asarray(out["q_gap_vent"], dtype=float) * top_area_total
    roof_lw_power_W = np.asarray(out["q_lw"], dtype=float) * top_area_total
    roof_conv_power_W = np.asarray(out["q_conv_roof"], dtype=float) * top_area_total
    roof_solar_power_W = np.asarray(out["q_solar_roof"], dtype=float) * top_area_total

    top_MJ = np.cumsum(top_power_W * dt_seconds) / 1e6
    wall_MJ = np.cumsum(wall_external_power_W * dt_seconds) / 1e6
    wall_ambient_MJ = np.cumsum(wall_ambient_power_W * dt_seconds) / 1e6
    wall_interface_MJ = np.cumsum(wall_interface_power_W * dt_seconds) / 1e6
    wall_latent_MJ = np.cumsum(wall_latent_power_W * dt_seconds) / 1e6
    wall_solar_MJ = np.cumsum(wall_solar_power_W * dt_seconds) / 1e6
    wall_solar_direct_MJ = np.cumsum(wall_solar_direct_power_W * dt_seconds) / 1e6
    wall_solar_diffuse_MJ = np.cumsum(wall_solar_diffuse_power_W * dt_seconds) / 1e6
    wall_solar_ground_MJ = np.cumsum(wall_solar_ground_power_W * dt_seconds) / 1e6
    bottom_MJ = np.cumsum(bottom_power_W * dt_seconds) / 1e6
    gap_MJ = np.cumsum(gap_power_W * dt_seconds) / 1e6
    roof_gap_MJ = np.cumsum(roof_gap_power_W * dt_seconds) / 1e6
    gap_vent_MJ = np.cumsum(gap_vent_power_W * dt_seconds) / 1e6
    roof_lw_MJ = np.cumsum(roof_lw_power_W * dt_seconds) / 1e6
    roof_conv_MJ = np.cumsum(roof_conv_power_W * dt_seconds) / 1e6
    roof_solar_MJ = np.cumsum(roof_solar_power_W * dt_seconds) / 1e6

    top_cool = np.cumsum(np.maximum(-top_power_W, 0.0) * dt_seconds) / 1e6
    wall_cool = np.cumsum(np.maximum(wall_interface_power_W, 0.0) * dt_seconds) / 1e6
    bottom_cool = np.cumsum(np.maximum(-bottom_power_W, 0.0) * dt_seconds) / 1e6
    total_cool = np.maximum(top_cool + wall_cool + bottom_cool, 1e-12)
    return {
        "cum_top_MJ": top_MJ,
        "cum_top_to_water_MJ": top_MJ,
        "cum_wall_MJ": wall_MJ,
        "cum_wall_external_MJ": wall_MJ,
        "cum_wall_ambient_MJ": wall_ambient_MJ,
        "cum_wall_interface_MJ": wall_interface_MJ,
        "cum_wall_ice_latent_MJ": wall_latent_MJ,
        "cum_wall_solar_abs_MJ": wall_solar_MJ,
        "cum_wall_solar_direct_MJ": wall_solar_direct_MJ,
        "cum_wall_solar_diffuse_MJ": wall_solar_diffuse_MJ,
        "cum_wall_solar_ground_MJ": wall_solar_ground_MJ,
        "cum_bottom_MJ": bottom_MJ,
        "cum_gap_to_water_MJ": gap_MJ,
        "cum_roof_gap_MJ": roof_gap_MJ,
        "cum_gap_vent_MJ": gap_vent_MJ,
        "cum_roof_lw_MJ": roof_lw_MJ,
        "cum_roof_conv_MJ": roof_conv_MJ,
        "cum_roof_solar_abs_MJ": roof_solar_MJ,
        "cum_top_cooling_MJ": top_cool,
        "cum_wall_cooling_MJ": wall_cool,
        "cum_bottom_cooling_MJ": bottom_cool,
        "freeze_budget_top_frac": top_cool / total_cool,
        "freeze_budget_wall_frac": wall_cool / total_cool,
        "freeze_budget_bottom_frac": bottom_cool / total_cool,
    }


def run_model(
    met: Meteo,
    T_init: float = 5.0,
    n_z: int = N_Z_DEFAULT,
    n_sector: int = N_SECT_DEFAULT,
    edge_position: str = "south",
    material_stack: MaterialStack | None = None,
) -> Dict[str, np.ndarray]:
    if material_stack is None:
        material_stack = MaterialStack()

    n = len(met.times)
    dz = H / n_z
    z_centers = np.linspace(dz / 2.0, H - dz / 2.0, n_z)
    dtheta = 2.0 * np.pi / n_sector
    theta_deg = sector_angles_deg(n_sector)

    A_top_total = np.pi * R**2
    sector_top_area = np.full(n_sector, A_top_total / n_sector)
    A_wall_layer_sector = np.full(n_sector, 2.0 * np.pi * R * dz / n_sector)
    A_wall_sector_total = A_wall_layer_sector * n_z
    C_roof = C_ROOF_PER_AREA * A_top_total
    C_gap = GAP_AIR_HEAT_CAPACITY * A_top_total
    top_membrane_R = membrane_resistance(material_stack)
    wall_solid_R = wall_interface_solid_resistance(material_stack)

    dt_met = (met.times[1] - met.times[0]).total_seconds()
    nsub = max(1, int(math.ceil(dt_met / MAX_SUB_DT)))
    dt = dt_met / nsub

    T = np.full((n, n_sector, n_z), T_init, dtype=float)
    T_shell = np.full((n, n_sector, n_z), T_init, dtype=float)
    T_core = np.full((n, n_sector, n_z), T_init, dtype=float)
    z_ice = np.zeros((n, n_sector), dtype=float)
    wall_ice = np.zeros((n, n_sector, n_z), dtype=float)
    snow_depth = np.zeros(n, dtype=float)
    T_roof = np.full(n, T_init, dtype=float)
    T_gap = np.full(n, T_init, dtype=float)
    T_sky = np.full(n, np.nan, dtype=float)
    rho = np.full((n, n_sector, n_z), rho_of_T(T_init), dtype=float)

    q_top_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_eq_sector = np.zeros((n, n_sector), dtype=float)
    q_bottom_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_solar_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_solar_direct_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_solar_diffuse_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_solar_ground_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_ambient_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_interface_sector = np.zeros((n, n_sector), dtype=float)
    q_wall_ice_latent_sector = np.zeros((n, n_sector), dtype=float)
    q_gap_water_sector = np.zeros((n, n_sector), dtype=float)
    q_roof_gap = np.zeros(n, dtype=float)
    q_gap_vent = np.zeros(n, dtype=float)
    q_conv_roof = np.zeros(n, dtype=float)
    q_lw = np.zeros(n, dtype=float)
    q_solar_roof_ts = np.zeros(n, dtype=float)
    q_radial_shell_core_sector = np.zeros((n, n_sector), dtype=float)
    radial_delta_mean_sector = np.zeros((n, n_sector), dtype=float)
    radial_delta_top_sector = np.zeros((n, n_sector), dtype=float)
    sector_solar_factor_ts = np.zeros((n, n_sector), dtype=float)
    sector_shade_factor_ts = np.zeros((n, n_sector), dtype=float)
    sector_wind_factor_ts = np.zeros((n, n_sector), dtype=float)
    sector_sky_view_factor_ts = np.zeros((n, n_sector), dtype=float)
    sector_ground_view_factor_ts = np.zeros((n, n_sector), dtype=float)
    sector_shaded_height_fraction_ts = np.zeros((n, n_sector), dtype=float)
    liquid_core_radius_sector = np.full((n, n_sector), R, dtype=float)
    liquid_core_radius_min = np.full(n, R, dtype=float)
    unstable_pairs = np.zeros(n, dtype=float)

    cum_sector_wall_solar_J = np.zeros(n_sector, dtype=float)
    cum_sector_wall_cooling_J = np.zeros(n_sector, dtype=float)
    cum_sector_wall_ambient_J = np.zeros(n_sector, dtype=float)
    cum_sector_wall_interface_J = np.zeros(n_sector, dtype=float)
    cum_sector_wall_ice_latent_J = np.zeros(n_sector, dtype=float)
    cum_sector_radial_exchange_J = np.zeros(n_sector, dtype=float)

    for i in range(1, n):
        prev_time = met.times[i - 1]
        current_time = met.times[i]
        if (current_time.year, current_time.month) != (prev_time.year, prev_time.month):
            print(f"[progress] Completed meteorology through {prev_time.strftime('%b %Y')}", flush=True)

        Tshell = T_shell[i - 1].copy()
        Tcore = T_core[i - 1].copy()
        zice = z_ice[i - 1].copy()
        wice = wall_ice[i - 1].copy()
        sdepth = float(snow_depth[i - 1])
        Troof = float(T_roof[i - 1])
        Tgap = float(T_gap[i - 1])

        tair = float(met.tair[i])
        wind = float(met.wind[i])
        wind_dir = float(met.wind_dir[i])
        ghi = float(met.ghi[i])
        rhm = float(met.rh[i])
        sun_elev = float(met.sun_elev_deg[i])
        sun_az = float(met.sun_az_deg[i])
    
        dni = met.dni[i]
        dhi = met.dhi[i]

        current_dt = met.times[i]
        snowfall_roof_depth_m = float(met.snowfall_cm[i]) / 100.0
        sdepth += snowfall_roof_depth_m
        if sdepth > 0.0 and wind > 0.0:
            sdepth *= np.exp(-K_WIND_CLEAR * wind * dt_met)

        wall_solar = wall_solar_components(
            theta_deg,
            sun_az,
            sun_elev,
            met.dni[i],
            met.dhi[i],
            float(met.ground_snow_depth_m[i]),
            edge_position
        )

        #add new solar factors
        q_wall_solar = wall_solar['q_total']
        q_wall_solar_direct = wall_solar['q_direct']
        q_wall_solar_diffuse = wall_solar['q_diffuse']
        q_wall_solar_ground = wall_solar['q_ground']
        #solar_fac = wall_solar['beam_projection']
        shade_fac = wall_solar['direct_transmission']
        sky_view_fac = wall_solar['sky_view_factor']
        ground_view_fac = wall_solar['ground_view_factor']
        shaded_height_fac = wall_solar['shaded_height_fraction']
        q_wall_solar_direct_base = wall_solar['q_direct']
        q_wall_solar_diffuse_base = wall_solar['q_diffuse']
        q_wall_solar_ground_base = wall_solar['q_ground']
        q_wall_solar_base = wall_solar['q_total']
        wind_fac = sector_wind_speed_factor(theta_deg, wind_dir, edge_position)
        roof_shade = roof_solar_shade_factor(sun_az, sun_elev, edge_position)
        roof_wind_fac = roof_wake_wind_factor(wind_dir, edge_position)

        q_top_last = np.zeros(n_sector)
        q_wall_last = np.zeros(n_sector)
        q_bottom_last = np.zeros(n_sector)
        q_gap_last = np.zeros(n_sector)
        q_wall_solar_last = np.zeros(n_sector)
        q_wall_solar_direct_last = np.zeros(n_sector)
        q_wall_solar_diffuse_last = np.zeros(n_sector)
        q_wall_solar_ground_last = np.zeros(n_sector)
        q_wall_ambient_last = np.zeros(n_sector)
        q_wall_interface_last = np.zeros(n_sector)
        q_wall_ice_latent_last = np.zeros(n_sector)
        q_radial_last = np.zeros(n_sector)
        q_lw_last = 0.0
        q_conv_roof_last = 0.0
        q_roof_gap_last = 0.0
        q_gap_vent_last = 0.0
        q_solar_roof_last = 0.0
        Tsky_last = np.nan

        printed = True # False to turn on printing

        for substep in range(nsub):
            h_roof = h_roof_from_wind(wind * roof_wind_fac)
            Tsky_k = sky_temperature(tair, rhm)
            Tsky_last = Tsky_k - 273.15
            T_roof_k = np.clip(Troof + 273.15, 180.0, 360.0)

            R_snow = sdepth / K_SNOW if sdepth > 0.0 else 0.0
            q_conv = (Troof - tair) / max(R_snow + 1.0 / h_roof, 1e-6)
            q_conv = float(np.clip(q_conv, -Q_ROOF_CLAMP, Q_ROOF_CLAMP))
            q_lw_now = EPS_ROOF * SIGMA * (T_roof_k**4 - Tsky_k**4)
            q_lw_now = float(np.clip(q_lw_now, -Q_ROOF_CLAMP, Q_ROOF_CLAMP))
            mu = max(np.sin(np.deg2rad(sun_elev)), 0.0)
            
            #q_solar_roof_now = ALPHA_ROOF * roof_shade * np.exp(-sdepth / 0.05) * (roof_direct + roof_diffuse)
            # --- NEW physically correct roof solar forcing ---

            dni = met.dni[i]
            dhi = met.dhi[i]

            mu = max(np.sin(np.deg2rad(sun_elev)), 0.0)

            roof_direct = dni * mu          # direct onto horizontal surface
            roof_diffuse = dhi              # diffuse already horizontal

            q_solar_roof_now = ALPHA_ROOF * roof_shade * np.exp(-sdepth / 0.05) * (
                roof_direct + roof_diffuse
            )

            q_roof_gap_now = H_GAP_ROOF * (Troof - Tgap)
            q_roof_gap_now = float(np.clip(q_roof_gap_now, -Q_ROOF_CLAMP, Q_ROOF_CLAMP))
            Troof += (A_top_total * (q_solar_roof_now - q_conv - q_lw_now - q_roof_gap_now) * dt) / C_roof
            Troof = float(np.clip(Troof, -50.0, 80.0))

            T_top_mean = shell_core_mean_temperature(Tshell[:, -1], Tcore[:, -1], wice[:, -1])
            T_interface = np.where(zice > 0.0, T_FREEZE, T_top_mean)
            R_gap_water = top_membrane_R + 1.0 / H_GAP_WATER + np.where(zice > 0, zice / K_I, 0.0)
            U_gap = 1.0 / np.maximum(R_gap_water, 1e-6)
            coeff = (A_top_total / C_gap) * (H_GAP_ROOF + H_GAP_VENT + float(np.mean(U_gap)))
            rhs = Tgap + (A_top_total * dt / C_gap) * (H_GAP_ROOF * Troof + H_GAP_VENT * tair + float(np.mean(U_gap * T_interface)))
            Tgap = rhs / (1.0 + dt * coeff)
            Tgap = float(np.clip(Tgap, -50.0, 80.0))
            q_roof_gap_now = H_GAP_ROOF * (Troof - Tgap)
            q_roof_gap_now = float(np.clip(q_roof_gap_now, -Q_ROOF_CLAMP, Q_ROOF_CLAMP))
            q_gap_vent_now = H_GAP_VENT * (Tgap - tair)
            q_gap_vent_now = float(np.clip(q_gap_vent_now, -Q_ROOF_CLAMP, Q_ROOF_CLAMP))
            q_gap_now = U_gap * (Tgap - T_interface)
            q_gap_now = np.clip(q_gap_now, -Q_TOP_CLAMP, Q_TOP_CLAMP)

            if Troof > 0.0 and sdepth > 0.0:
                sdepth = max(0.0, sdepth - C_SNOW_MELT * Troof * dt)

            Tshell = diffuse_vertical(Tshell, dt, dz)
            Tcore = diffuse_vertical(Tcore, dt, dz)
            if AZIMUTHAL_MIXING_FACTOR > 0.0:
                Tshell = diffuse_azimuthal(Tshell, dt, 0.85 * R, dtheta)
                Tcore = diffuse_azimuthal(Tcore, dt, 0.45 * R, dtheta)

            # Bottom forcing applied to the mean bottom-layer temperature, then shifted into both radial nodes.
            T_bottom_mean = shell_core_mean_temperature(Tshell[:, 0], Tcore[:, 0], wice[:, 0])
            Tground = ground_temperature(current_dt)
            q_bottom = H_GROUND * (Tground - T_bottom_mean)
            dT_bottom = q_bottom * dt / (RHO_W * dz * CP_W)
            Tshell[:, 0] += dT_bottom
            Tcore[:, 0] += dT_bottom

            # Top boundary / surface ice acts on the mean top-layer temperature.
            for s in range(n_sector):
                top_mean_before = float(shell_core_mean_temperature(Tshell[s, -1], Tcore[s, -1], wice[s, -1]))
                top_mean_after, zice[s] = update_surface_phase_sector(top_mean_before, float(zice[s]), float(q_gap_now[s]), dt, dz)
                delta_top = top_mean_after - top_mean_before
                Tshell[s, -1] += delta_top
                Tcore[s, -1] += delta_top
                zice[s] = float(max(0.0, zice[s]))

            # Wall forcing, wall ice, and shell-to-core radial exchange
            q_wall_eq = np.zeros(n_sector)
            q_wall_ambient_eq = np.zeros(n_sector)
            q_wall_interface_eq = np.zeros(n_sector)
            q_wall_ice_latent_eq = np.zeros(n_sector)
#           q_wall_solar = q_wall_solar_base.copy()
            q_wall_solar_direct = q_wall_solar_direct_base.copy()
            q_wall_solar_diffuse = q_wall_solar_diffuse_base.copy()
            q_wall_solar_ground = q_wall_solar_ground_base.copy()
            q_radial_eq = np.zeros(n_sector)

#Added wall solar factor to account for incorrect wall area calculation
            # --- Physically correct wall solar forcing (DNI + DHI) ---

            dni = met.dni[i]
            dhi = met.dhi[i]
            sun_elev = met.sun_elev_deg[i]
            sun_az = met.sun_az_deg[i]

            # --- DEBUG: check solar forcing ---
            if not printed and sun_elev > 60:
                print(f"[debug] hour {i}")
                print(f"  sun_elev = {sun_elev:.1f} deg")
                print(f"  DNI = {met.dni[i]:.1f} W/m^2")
                print(f"  DHI = {met.dhi[i]:.1f} W/m^2")
                print(f"  GHI = {met.ghi[i]:.1f} W/m^2")
                printed = True


            mu = max(np.sin(np.deg2rad(sun_elev)), 0.0)

            # Sector orientation
            cos_theta = np.maximum(np.cos(np.deg2rad(theta_deg - sun_az)), 0.0)

            # --- Direct beam ---
            q_wall_direct = dni * cos_theta

            # --- Diffuse sky ---
            F_sky = 0.5
            q_wall_diffuse = F_sky * dhi

            # --- Ground reflected ---
            rho_ground = 0.2
            F_ground = 0.5
            ghi_local = dni * mu + dhi
            q_wall_ground = F_ground * rho_ground * ghi_local

            # --- Total ---
            q_wall_solar = ALPHA_WALL_SOLAR * (
                q_wall_direct + q_wall_diffuse + q_wall_ground
            )

            for s in range(n_sector):
                h_wall = h_wall_from_wind(wind * wind_fac[s])
                for j in range(n_z):
                    shell_cap, _, _, _ = radial_partition_per_wall_area(float(wice[s, j]))
                    shell_cap = float(max(shell_cap, 1e-9))

                    if wice[s, j] > 0.0:
                        R_wall = wall_solid_R + 1.0 / h_wall + wice[s, j] / K_I
                        U_wall = 1.0 / max(R_wall, 1e-6)
                        q_wall_ambient = U_wall * (T_FREEZE - tair)
                    else:
                        R_wall = wall_solid_R + 1.0 / h_wall
                        U_wall = 1.0 / max(R_wall, 1e-6)
                        q_wall_ambient = U_wall * (Tshell[s, j] - tair)
                    q_wall_ambient = float(np.clip(q_wall_ambient, -Q_WALL_CLAMP, Q_WALL_CLAMP))
                    q_wall = float(np.clip(q_wall_ambient - q_wall_solar[s], -Q_WALL_CLAMP, Q_WALL_CLAMP))

                    Tshell[s, j], wice[s, j], q_wall_interface, q_wall_latent = update_wall_phase_cell(
                        float(Tshell[s, j]),
                        float(wice[s, j]),
                        q_wall,
                        dt,
                        shell_cap,
                        H_WALL_WATER_ICE,
                        MAX_WALL_ICE_THICKNESS,
                    )
                    Tshell[s, j], Tcore[s, j], q_radial = apply_shell_core_exchange(float(Tshell[s, j]), float(Tcore[s, j]), float(wice[s, j]), dt)
                    area_ratio = A_wall_layer_sector[s] / sector_top_area[s]
                    q_wall_eq[s] += q_wall * area_ratio
                    q_wall_ambient_eq[s] += q_wall_ambient * area_ratio
                    q_wall_interface_eq[s] += q_wall_interface * area_ratio
                    q_wall_ice_latent_eq[s] += q_wall_latent * area_ratio
                    q_radial_eq[s] += q_radial * area_ratio

            Tshell, Tcore = apply_top_mixing_shell_core(Tshell, Tcore, wice)

            # Convective adjustment on the mean column; damp shell-core deltas during overturn.
            if substep == 0:
                for s in range(n_sector):
                    mean_before = np.asarray(shell_core_mean_temperature(Tshell[s], Tcore[s], wice[s]), dtype=float)
                    mean_after = np.asarray(convective_adjustment_column(mean_before), dtype=float)
                    dT_col = mean_after - mean_before
                    Tshell[s] += dT_col
                    Tcore[s] += dT_col
                    Tshell[s] = (1.0 - RADIAL_OVERTURN_MIX) * Tshell[s] + RADIAL_OVERTURN_MIX * mean_after
                    Tcore[s] = (1.0 - RADIAL_OVERTURN_MIX) * Tcore[s] + RADIAL_OVERTURN_MIX * mean_after

            Tshell = np.clip(Tshell, MIN_TEMP_CLIP, MAX_TEMP_CLIP)
            Tcore = np.clip(Tcore, MIN_TEMP_CLIP, MAX_TEMP_CLIP)
            Tn = np.asarray(shell_core_mean_temperature(Tshell, Tcore, wice), dtype=float)

            q_top_last = q_gap_now.copy()
            q_bottom_last = q_bottom.copy()
            q_wall_last = q_wall_eq.copy()
            q_wall_solar_last = q_wall_solar.copy()
            q_wall_solar_direct_last = q_wall_solar_direct.copy()
            q_wall_solar_diffuse_last = q_wall_solar_diffuse.copy()
            q_wall_solar_ground_last = q_wall_solar_ground.copy()
            q_wall_ambient_last = q_wall_ambient_eq.copy()
            q_wall_interface_last = q_wall_interface_eq.copy()
            q_wall_ice_latent_last = q_wall_ice_latent_eq.copy()
            q_gap_last = q_gap_now.copy()
            q_radial_last = q_radial_eq.copy()
            q_lw_last = q_lw_now
            q_conv_roof_last = q_conv
            q_roof_gap_last = q_roof_gap_now
            q_gap_vent_last = q_gap_vent_now
            q_solar_roof_last = q_solar_roof_now

            cum_sector_wall_solar_J += q_wall_solar * A_wall_sector_total * dt
            cum_sector_wall_cooling_J += np.maximum(q_wall_interface_eq, 0.0) * sector_top_area * dt
            cum_sector_wall_ambient_J += q_wall_ambient_eq * sector_top_area * dt
            cum_sector_wall_interface_J += q_wall_interface_eq * sector_top_area * dt
            cum_sector_wall_ice_latent_J += q_wall_ice_latent_eq * sector_top_area * dt
            cum_sector_radial_exchange_J += q_radial_eq * sector_top_area * dt

        T[i] = Tn
        T_shell[i] = Tshell
        T_core[i] = Tcore
        z_ice[i] = zice
        wall_ice[i] = wice
        snow_depth[i] = sdepth
        T_roof[i] = Troof
        T_gap[i] = Tgap
        T_sky[i] = Tsky_last
        rho[i] = rho_of_T(Tn)
        q_top_sector[i] = q_top_last
        q_wall_eq_sector[i] = q_wall_last
        q_bottom_sector[i] = q_bottom_last
        q_wall_solar_sector[i] = q_wall_solar_last
        q_wall_solar_direct_sector[i] = q_wall_solar_direct_last
        q_wall_solar_diffuse_sector[i] = q_wall_solar_diffuse_last
        q_wall_solar_ground_sector[i] = q_wall_solar_ground_last
        q_wall_ambient_sector[i] = q_wall_ambient_last
        q_wall_interface_sector[i] = q_wall_interface_last
        q_wall_ice_latent_sector[i] = q_wall_ice_latent_last
        q_gap_water_sector[i] = q_gap_last
        q_roof_gap[i] = q_roof_gap_last
        q_gap_vent[i] = q_gap_vent_last
        q_conv_roof[i] = q_conv_roof_last
        q_lw[i] = q_lw_last
        q_solar_roof_ts[i] = q_solar_roof_last
        q_radial_shell_core_sector[i] = q_radial_last
        radial_delta_mean_sector[i] = np.mean(np.abs(Tshell - Tcore), axis=1)
        radial_delta_top_sector[i] = np.abs(Tshell[:, -1] - Tcore[:, -1])
        sector_solar_factor_ts[i] = q_wall_solar
    
        sector_shade_factor_ts[i] = shade_fac
        sector_wind_factor_ts[i] = wind_fac
        sector_sky_view_factor_ts[i] = sky_view_fac
        sector_ground_view_factor_ts[i] = ground_view_fac
        sector_shaded_height_fraction_ts[i] = shaded_height_fac
        liquid_core_radius_sector[i] = np.maximum(R - np.max(wice, axis=1), MIN_LIQUID_CORE_RADIUS)
        liquid_core_radius_min[i] = float(np.min(liquid_core_radius_sector[i]))
        unstable_pairs[i] = float(np.sum(rho[i, :, :-1] < rho[i, :, 1:]))

    if n > 0:
        print(f"[progress] Completed meteorology through {met.times[-1].strftime('%b %Y')}", flush=True)
        print("[progress] Computing cumulative budgets...", flush=True)

    surface_ice_equiv = np.mean(z_ice, axis=1)
    surface_ice_cover_fraction = np.mean(z_ice > 0.0, axis=1)
    wall_ice_avg = np.mean(wall_ice, axis=(1, 2))
    wall_ice_max = np.max(wall_ice, axis=(1, 2))

    out: Dict[str, np.ndarray] = {
        "T": T,
        "T_shell": T_shell,
        "T_core": T_core,
        "z_ice_sector": z_ice,
        "wall_ice": wall_ice,
        "snow_depth": snow_depth,
        "T_roof": T_roof,
        "T_gap": T_gap,
        "T_sky": T_sky,
        "rho": rho,
        "q_top_sector": q_top_sector,
        "q_wall_eq_sector": q_wall_eq_sector,
        "q_bottom_sector": q_bottom_sector,
        "q_wall_solar_sector": q_wall_solar_sector,
        "q_wall_solar_direct_sector": q_wall_solar_direct_sector,
        "q_wall_solar_diffuse_sector": q_wall_solar_diffuse_sector,
        "q_wall_solar_ground_sector": q_wall_solar_ground_sector,
        "q_wall_ambient_sector": q_wall_ambient_sector,
        "q_wall_interface_sector": q_wall_interface_sector,
        "q_wall_ice_latent_sector": q_wall_ice_latent_sector,
        "q_gap_water_sector": q_gap_water_sector,
        "q_roof_gap": q_roof_gap,
        "q_gap_vent": q_gap_vent,
        "q_conv_roof": q_conv_roof,
        "q_lw": q_lw,
        "q_solar_roof": q_solar_roof_ts,
        "q_radial_shell_core_sector": q_radial_shell_core_sector,
        "radial_delta_mean_sector": radial_delta_mean_sector,
        "radial_delta_top_sector": radial_delta_top_sector,
        "cum_sector_wall_solar_J": cum_sector_wall_solar_J,
        "cum_sector_wall_cooling_J": cum_sector_wall_cooling_J,
        "cum_sector_wall_ambient_J": cum_sector_wall_ambient_J,
        "cum_sector_wall_interface_J": cum_sector_wall_interface_J,
        "cum_sector_wall_ice_latent_J": cum_sector_wall_ice_latent_J,
        "cum_sector_radial_exchange_J": cum_sector_radial_exchange_J,
        "surface_ice_equiv": surface_ice_equiv,
        "surface_ice_cover_fraction": surface_ice_cover_fraction,
        "wall_ice_avg": wall_ice_avg,
        "wall_ice_max": wall_ice_max,
        "theta_deg": theta_deg,
        "edge_position": edge_position,
        "sector_solar_factor": sector_solar_factor_ts,
        "sector_shade_factor": sector_shade_factor_ts,
        "sector_wind_factor": sector_wind_factor_ts,
        "sector_sky_view_factor": sector_sky_view_factor_ts,
        "sector_ground_view_factor": sector_ground_view_factor_ts,
        "sector_shaded_height_fraction": sector_shaded_height_fraction_ts,
        "liquid_core_radius_sector": liquid_core_radius_sector,
        "liquid_core_radius_min": liquid_core_radius_min,
        "unstable_pairs": unstable_pairs,
        "sector_top_area": np.broadcast_to(sector_top_area, (n, n_sector)),
        "top_area_total": A_top_total,
        "wall_area_total": 2.0 * np.pi * R * H,
        "top_membrane_resistance": top_membrane_R,
        "wall_geotextile_resistance": wall_geotextile_resistance(material_stack),
        "wall_interface_solid_resistance": wall_solid_R,
        "membrane_total_thickness_m": material_stack.membrane_total_thickness_m,
        "membrane_k": material_stack.membrane_k,
        "wall_geotextile_thickness_m": material_stack.wall_geotextile_thickness_m,
        "wall_geotextile_k": material_stack.wall_geotextile_k,
        "radial_shell_thickness_m": RADIAL_SHELL_THICKNESS_M,
        "radial_exchange_base": RADIAL_EXCHANGE_BASE,
        "radial_exchange_buoy": RADIAL_EXCHANGE_BUOY,
        "radial_exchange_max": RADIAL_EXCHANGE_MAX,
        "radial_overturn_mix": RADIAL_OVERTURN_MIX,
        "z_centers": z_centers,
    }
    out.update(make_energy_budget(out, dt_met))
    return out


# ------------------------------ Plotting -----------------------------------
def apply_month_ticks_if_needed(ax, times: List[datetime], xlabel: str) -> None:
    if xlabel == "Date":
        ticks, labels = month_ticks(times)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if len(times) > 1 and times[0] != times[-1]:
            ax.set_xlim(mdates.date2num(times[0]), mdates.date2num(times[-1]))


def save_ghi_sanity_plot(met: Meteo, plotpng: str) -> Path:
    base = Path(plotpng)
    p = base.with_name('ghi_sanity_check.png')

    raw = np.asarray(getattr(met, 'ghi_raw', met.ghi), dtype=float)
    used = np.asarray(met.ghi, dtype=float)
    toa = np.asarray(getattr(met, 'ghi_toa_horizontal', np.zeros_like(raw)), dtype=float)
    clear = np.asarray(getattr(met, 'ghi_clear_cap', np.zeros_like(raw)), dtype=float)

    x = mdates.date2num(met.times)
    xlabel = 'Date'
    days = np.array([(t - met.times[0]).total_seconds() / 86400.0 for t in met.times])
    day_id = np.floor(days).astype(int)
    unique_days = np.unique(day_id)
    xd = np.array([met.times[0] + timedelta(days=int(d)) for d in unique_days])
    xx = mdates.date2num(xd.tolist())
    daily_mean_raw = np.array([raw[day_id == d].mean() for d in unique_days])
    daily_mean_used = np.array([used[day_id == d].mean() for d in unique_days])
    daily_max_raw = np.array([raw[day_id == d].max() for d in unique_days])
    daily_max_used = np.array([used[day_id == d].max() for d in unique_days])
    daily_max_clear = np.array([clear[day_id == d].max() for d in unique_days])
    daily_max_toa = np.array([toa[day_id == d].max() for d in unique_days])

    clear_safe = np.maximum(clear, 1.0)
    clear_index_raw = raw / clear_safe
    clear_index_used = used / clear_safe

    fig, ax = plt.subplots(3, 1, figsize=(12.5, 12.0), sharex=False)
    ax[0].plot(x, raw, linewidth=0.6, alpha=0.5, label='Hourly GHI raw')
    if np.any(np.abs(raw - used) > 1e-9):
        ax[0].plot(x, used, linewidth=0.9, label='Hourly GHI used after clip')
    ax[0].plot(x, clear, linewidth=0.9, label='Pressure-aware clear-sky cap')
    ax[0].plot(x, toa, linewidth=0.8, label='TOA horizontal cap')
    ax[0].set_ylabel('Hourly GHI [W/m²]')
    ax[0].set_title(f'Hourly GHI sanity check (mode={getattr(met, "ghi_clip_mode", "off")}, factor={getattr(met, "ghi_clip_factor", 1.0):.2f})')
    ax[0].grid(True)
    ax[0].legend(fontsize=8, ncol=2)

    ax[1].plot(xx, daily_mean_raw, label='Daily mean GHI raw')
    if np.any(np.abs(daily_mean_raw - daily_mean_used) > 1e-9):
        ax[1].plot(xx, daily_mean_used, label='Daily mean GHI used')
    ax[1].plot(xx, daily_max_raw, label='Daily max GHI raw', alpha=0.8)
    if np.any(np.abs(daily_max_raw - daily_max_used) > 1e-9):
        ax[1].plot(xx, daily_max_used, label='Daily max GHI used', alpha=0.9)
    ax[1].plot(xx, daily_max_clear, label='Daily max clear-sky cap')
    ax[1].plot(xx, daily_max_toa, label='Daily max TOA horizontal cap', alpha=0.8)
    ax[1].set_ylabel('Daily GHI [W/m²]')
    ax[1].grid(True)
    ax[1].legend(fontsize=8, ncol=2)

    ax[2].plot(x, clear_index_raw, linewidth=0.6, alpha=0.55, label='Hourly GHI / clear-sky cap (raw)')
    if np.any(np.abs(raw - used) > 1e-9):
        ax[2].plot(x, clear_index_used, linewidth=0.8, label='Hourly GHI / clear-sky cap (used)')
    ax[2].axhline(1.0, linewidth=1.0, linestyle='--', label='Clear-sky cap = 1.0')
    ax[2].set_ylabel('Clear-sky index [-]')
    ax[2].set_xlabel(xlabel)
    ax[2].grid(True)
    ax[2].legend(fontsize=8, ncol=2)

    ticks_all, labels_all = month_ticks(met.times)
    ticks_days, labels_days = month_ticks(xd.tolist())
    ax[0].set_xticks(ticks_all); ax[0].set_xticklabels(labels_all)
    ax[2].set_xticks(ticks_all); ax[2].set_xticklabels(labels_all)
    ax[1].set_xticks(ticks_days); ax[1].set_xticklabels(labels_days)
    if len(x) > 1:
        ax[0].set_xlim(x[0], x[-1]); ax[2].set_xlim(x[0], x[-1])
    if len(xx) > 1:
        ax[1].set_xlim(xx[0], xx[-1])

    summary = ghi_clip_summary(met)
    ax[2].text(0.01, 0.98, summary, transform=ax[2].transAxes, va='top', ha='left', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.7'))

    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def make_plots(met: Meteo, out: Dict[str, np.ndarray], plotpng: str) -> List[Path]:
    x, xlabel = time_axis(met.times)
    base = Path(plotpng)
    saved: List[Path] = []
    theta_deg = out["theta_deg"]
    sector_labels = [f"{int(td):03d}°" for td in theta_deg]

    # Main panel set
    fig, ax = plt.subplots(6, 2, figsize=(14, 18), sharex=True)
    ax = ax.ravel()
    ax[0].plot(x, met.tair)
    ax[0].set_ylabel("Air T [°C]")
    _set_dynamic_ylim(ax[0], met.tair)
    ax[0].grid(True)

    ax[1].plot(x, out["T"].mean(axis=1)[:, -1], label="Sector-mean top layer")
    ax[1].plot(x, out["T"].mean(axis=1)[:, 0], label="Sector-mean bottom layer")
    ax[1].set_ylabel("Water T [°C]")
    _set_dynamic_ylim(ax[1], out["T"].mean(axis=1)[:, -1], out["T"].mean(axis=1)[:, 0], include=(0.0, 4.0))
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(x, out["surface_ice_equiv"], label="Mean uniform full-surface-equivalent ice thickness")
    ax[2].plot(x, np.max(out["z_ice_sector"], axis=1), label="Max sector surface ice thickness")
    ax[2].set_ylabel("Surface ice [m]")
    _set_dynamic_ylim(ax[2], out["surface_ice_equiv"], np.max(out["z_ice_sector"], axis=1), include=(0.0,))
    ax[2].legend(fontsize=8)
    ax[2].grid(True)

    ax[3].plot(x, out["wall_ice_avg"], label="Average wall ice")
    ax[3].plot(x, out["wall_ice_max"], label="Maximum wall ice")
    ax[3].set_ylabel("Wall ice [m]")
    _set_dynamic_ylim(ax[3], out["wall_ice_avg"], out["wall_ice_max"], include=(0.0,))
    ax[3].legend(fontsize=8)
    ax[3].grid(True)

    ax[4].plot(x, out["T_roof"], label="Roof")
    ax[4].plot(x, out["T_gap"], label="Gap air")
    ax[4].plot(x, met.tair, label="Ambient air")
    ax[4].plot(x, out["T_sky"], label="Sky")
    ax[4].set_ylabel("Temperatures [°C]")
    _set_dynamic_ylim(ax[4], out["T_roof"], out["T_gap"], met.tair, out["T_sky"])
    ax[4].legend(fontsize=8, ncol=2)
    ax[4].grid(True)

    ax[5].plot(x, met.wind, label="Wind speed")
    ax[5].set_ylabel("Wind [m/s]")
    _set_dynamic_ylim(ax[5], met.wind, include=(0.0,))
    ax[5].grid(True)

    for s in range(len(theta_deg)):
        ax[6].plot(x, out["q_top_sector"][:, s], label=sector_labels[s])
    ax[6].set_ylabel("Top flux [W/m²]\n(+ heats water)")
    _set_dynamic_ylim(ax[6], out["q_top_sector"])
    ax[6].grid(True)

    for s in range(len(theta_deg)):
        ax[7].plot(x, out["q_wall_eq_sector"][:, s], label=sector_labels[s])
    ax[7].set_ylabel("Wall-eq flux [W/m²]\n(+ cools water)")
    _set_dynamic_ylim(ax[7], out["q_wall_eq_sector"])
    ax[7].grid(True)

    ax[8].plot(x, out["q_bottom_sector"].mean(axis=1))
    ax[8].set_ylabel("Bottom flux [W/m²]\n(+ heats water)")
    _set_dynamic_ylim(ax[8], out["q_bottom_sector"].mean(axis=1), include=(0.0,))
    ax[8].grid(True)

    ax[9].plot(x, met.ghi)
    ax[9].set_ylabel("GHI [W/m²]")
    _set_dynamic_ylim(ax[9], met.ghi, include=(0.0,))
    ax[9].grid(True)

    ax[10].plot(x, met.sun_elev_deg, label="Sun elevation")
    ax[10].plot(x, met.wind_dir, label="Wind direction [deg]")
    ax[10].set_ylabel("Sun elev / wind dir")
    _set_dynamic_ylim(ax[10], met.sun_elev_deg, met.wind_dir)
    ax[10].legend(fontsize=8)
    ax[10].grid(True)

    ax[11].plot(x, out["liquid_core_radius_min"], label="Min liquid-core radius")
    ax[11].set_ylabel("Liquid-core radius [m]")
    _set_dynamic_ylim(ax[11], out["liquid_core_radius_min"], include=(0.0, R))
    ax[11].grid(True)

    for a in ax[-2:]:
        a.set_xlabel(xlabel)
        apply_month_ticks_if_needed(a, met.times, xlabel)
    fig.tight_layout()
    fig.savefig(base, dpi=200)
    saved.append(base)
    plt.close(fig)

    saved.append(save_ghi_sanity_plot(met, plotpng))

    # Mean depth heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    im = ax2.imshow(out["T"].mean(axis=1).T, aspect='auto', origin='lower', extent=[x.min(), x.max(), 0, H])
    cb = fig2.colorbar(im, ax=ax2)
    cb.set_label("Sector-mean water T [°C]")
    ax2.set_xlabel(xlabel)
    apply_month_ticks_if_needed(ax2, met.times, xlabel)
    ax2.set_ylabel("Height above bottom [m]")
    ax2.set_title("Sector-mean temperature vs time and depth")
    p2 = base.with_name(base.stem + "_depth.png")
    fig2.tight_layout(); fig2.savefig(p2, dpi=200); saved.append(p2); plt.close(fig2)

    # Sector temperatures
    fig3, ax3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for s in range(len(theta_deg)):
        ax3[0].plot(x, out["T"][:, s, -1], label=sector_labels[s])
        ax3[1].plot(x, out["T"][:, s, 0], label=sector_labels[s])
    ax3[0].set_title("Top-layer water temperature by sector")
    ax3[1].set_title("Bottom-layer water temperature by sector")
    ax3[0].set_ylabel("Top T [°C]")
    ax3[1].set_ylabel("Bottom T [°C]")
    ax3[1].set_xlabel(xlabel)
    apply_month_ticks_if_needed(ax3[1], met.times, xlabel)
    ax3[0].legend(fontsize=7, ncol=4)
    for a in ax3:
        a.grid(True)
        _set_dynamic_ylim(a, *[a.lines[k].get_ydata() for k in range(len(a.lines))], include=(0.0, 4.0))
    p3 = base.with_name(base.stem + "_sector_temperatures.png")
    fig3.tight_layout(); fig3.savefig(p3, dpi=200); saved.append(p3); plt.close(fig3)

    # Sector wall ice and sector surface ice
    fig4, ax4 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for s in range(len(theta_deg)):
        ax4[0].plot(x, np.mean(out["wall_ice"][:, s, :], axis=1), label=sector_labels[s])
        ax4[1].plot(x, out["z_ice_sector"][:, s], label=sector_labels[s])
    ax4[0].set_title("Average wall-ice thickness by sector")
    ax4[1].set_title("Surface-ice thickness by sector")
    ax4[0].set_ylabel("Wall ice [m]")
    ax4[1].set_ylabel("Surface ice [m]")
    ax4[1].set_xlabel(xlabel)
    apply_month_ticks_if_needed(ax4[1], met.times, xlabel)
    ax4[0].legend(fontsize=7, ncol=4)
    for a in ax4:
        a.grid(True)
        _set_dynamic_ylim(a, *[a.lines[k].get_ydata() for k in range(len(a.lines))], include=(0.0,))
    p4 = base.with_name(base.stem + "_sector_ice.png")
    fig4.tight_layout(); fig4.savefig(p4, dpi=200); saved.append(p4); plt.close(fig4)

    if "q_radial_shell_core_sector" in out and "radial_delta_mean_sector" in out:
        fig4b, ax4b = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for s in range(len(theta_deg)):
            ax4b[0].plot(x, out["q_radial_shell_core_sector"][:, s], label=sector_labels[s])
            ax4b[1].plot(x, out["radial_delta_mean_sector"][:, s], label=sector_labels[s])
        ax4b[0].set_title("Shell-to-core radial exchange by sector")
        ax4b[1].set_title("Mean |T_shell - T_core| by sector")
        ax4b[0].set_ylabel("Equivalent flux [W/m²]\n(+ shell -> core)")
        ax4b[1].set_ylabel("Radial ΔT [°C]")
        ax4b[1].set_xlabel(xlabel)
        apply_month_ticks_if_needed(ax4b[1], met.times, xlabel)
        ax4b[0].legend(fontsize=7, ncol=4)
        for a in ax4b:
            a.grid(True)
            _set_dynamic_ylim(a, *[a.lines[k].get_ydata() for k in range(len(a.lines))], include=(0.0,))
        p4b = base.with_name(base.stem + "_radial_transport.png")
        fig4b.tight_layout(); fig4b.savefig(p4b, dpi=200); saved.append(p4b); plt.close(fig4b)

    # Daily met stats
    days = np.array([(t - met.times[0]).total_seconds() / 86400.0 for t in met.times])
    day_id = np.floor(days).astype(int)
    unique_days = np.unique(day_id)
    xd = np.array([met.times[0] + timedelta(days=int(d)) for d in unique_days])
    tair_mean = np.array([met.tair[day_id == d].mean() for d in unique_days])
    wind_mean = np.array([met.wind[day_id == d].mean() for d in unique_days])
    ghi_mean = np.array([met.ghi[day_id == d].mean() for d in unique_days])
    psfc_mean = np.array([met.psfc[day_id == d].mean() for d in unique_days])
    rh_mean = np.array([met.rh[day_id == d].mean() for d in unique_days])
    snowfall_cm_sum = np.array([met.snowfall_cm[day_id == d].sum() for d in unique_days])
    snow_swe_sum = np.array([met.snowfall_swe_mm[day_id == d].sum() for d in unique_days])
    wind_dir_mean = np.array([
        (np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(met.wind_dir[day_id == d]))), np.mean(np.cos(np.deg2rad(met.wind_dir[day_id == d]))))) + 360.0) % 360.0
        for d in unique_days
    ])
    fig5, ax5 = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    ax5 = ax5.ravel()
    xx = mdates.date2num(xd.tolist())
    daily_ticks, daily_labels = month_ticks(xd.tolist())
    ax5[0].plot(xx, tair_mean); ax5[0].set_ylabel('Daily mean air T [°C]'); ax5[0].grid(True)
    ax5[1].plot(xx, wind_mean); ax5[1].set_ylabel('Daily mean wind [m/s]'); ax5[1].grid(True)
    ax5[2].plot(xx, wind_dir_mean); ax5[2].set_ylabel('Daily mean wind dir [deg]'); ax5[2].set_ylim(-5, 365); ax5[2].grid(True)
    ax5[3].plot(xx, ghi_mean); ax5[3].set_ylabel('Daily mean GHI [W/m²]'); ax5[3].grid(True)
    ax5[4].plot(xx, psfc_mean); ax5[4].set_ylabel('Daily mean pressure [Pa]'); ax5[4].grid(True)
    ax5[5].plot(xx, rh_mean); ax5[5].set_ylabel('Daily mean RH [%]'); ax5[5].grid(True)
    ax5[6].plot(xx, snowfall_cm_sum, label='Daily snowfall depth sum [cm/day]')
    ax5b = ax5[6].twinx(); ax5b.plot(xx, snow_swe_sum, linestyle='--', label='Daily snowfall SWE sum [mm/day]')
    ax5[6].set_ylabel('Snowfall depth [cm/day]'); ax5b.set_ylabel('Snowfall SWE [mm/day]'); ax5[6].grid(True)
    ax5[7].plot(xx, np.array([met.ground_snow_depth_m[day_id == d].mean() for d in unique_days])); ax5[7].set_ylabel('Daily mean ground snow depth [m]'); ax5[7].grid(True)
    for a in ax5:
        a.set_xticks(daily_ticks)
        a.set_xticklabels(daily_labels)
        if len(xx) > 1 and xx[0] != xx[-1]:
            a.set_xlim(xx[0], xx[-1])
    ax5[6].set_xlabel('Date')
    ax5[7].set_xlabel('Date')
    p5 = base.with_name('met_daily_stats.png')
    fig5.tight_layout(); fig5.savefig(p5, dpi=200); saved.append(p5); plt.close(fig5)

    # Roof/gap/sky temperatures
    fig6, ax6 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax6[0].plot(x, out['T_roof'], label='Roof temperature')
    ax6[0].plot(x, out['T_gap'], label='Gap-air temperature')
    ax6[0].plot(x, met.tair, label='Ambient-air temperature', alpha=0.9)
    ax6[0].plot(x, out['T_sky'], label='Sky temperature')
    ax6[0].legend(fontsize=8, ncol=2); ax6[0].set_ylabel('Temperature [°C]'); ax6[0].grid(True)
    sky_offset = float(np.nanmean(out['T_gap']) - np.nanmean(out['T_sky']))
    ax6[1].plot(x, out['T_roof'], label='Roof temperature')
    ax6[1].plot(x, out['T_gap'], label='Gap-air temperature')
    ax6[1].plot(x, met.tair, label='Ambient-air temperature', alpha=0.9)
    ax6[1].plot(x, out['T_sky'] + sky_offset, label=f'Sky temperature + {sky_offset:.1f} °C offset')
    ax6[1].legend(fontsize=8, ncol=2); ax6[1].set_ylabel('Temperature [°C]'); ax6[1].grid(True)
    delta_gap = out['T_gap'] - met.tair
    ax6[2].plot(x, delta_gap, label='Gap-air temperature minus ambient-air temperature')
    ax6[2].set_ylabel('ΔT [°C]'); ax6[2].set_xlabel(xlabel); apply_month_ticks_if_needed(ax6[2], met.times, xlabel); ax6[2].grid(True)
    p6 = base.with_name('roof_airgap_sky_temperatures.png')
    fig6.tight_layout(); fig6.savefig(p6, dpi=200); saved.append(p6); plt.close(fig6)

    # Roof and gap heat pathways
    fig7, ax7 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    ax7[0].plot(x, out['q_lw']); ax7[0].set_ylabel('Roof-sky LW [W/m²]\n(+ cools roof)'); ax7[0].grid(True)
    ax7[1].plot(x, out['q_conv_roof']); ax7[1].set_ylabel('Roof-ambient conv [W/m²]\n(+ cools roof)'); ax7[1].grid(True)
    ax7[2].plot(x, out['q_roof_gap']); ax7[2].set_ylabel('Roof-gap [W/m²]\n(+ roof→gap)'); ax7[2].grid(True)
    ax7[3].plot(x, out['q_gap_vent'], label='Gap-ambient venting flux (+ gap→ambient)')
    ax7[3].plot(x, out['q_gap_water_sector'].mean(axis=1), label='Gap-water / gap-ice flux (+ heats water)')
    ax7[3].set_ylabel('Gap pathway [W/m²]'); ax7[3].set_xlabel(xlabel); apply_month_ticks_if_needed(ax7[3], met.times, xlabel); ax7[3].legend(fontsize=8); ax7[3].grid(True)
    p7 = base.with_name('roof_and_gap_heat_pathways.png')
    fig7.tight_layout(); fig7.savefig(p7, dpi=200); saved.append(p7); plt.close(fig7)

    # Snow metrics
    fig8, ax8 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax8[0].plot(x, met.snowfall_cm); ax8[0].set_ylabel('Hourly snowfall depth\n[cm over preceding hour]'); ax8[0].grid(True)
    ax8[1].plot(x, met.snowfall_swe_mm); ax8[1].set_ylabel('Hourly snowfall SWE\n[mm over preceding hour]'); ax8[1].grid(True)
    ax8[2].plot(x, out['snow_depth'], label='Roof snow depth')
    ax8[2].plot(x, met.ground_snow_depth_m, label='Ground snow depth')
    ax8[2].set_ylabel('Snow depth [m]'); ax8[2].legend(); ax8[2].grid(True)
    ax8[2].set_xlabel(xlabel); apply_month_ticks_if_needed(ax8[2], met.times, xlabel)
    p8 = base.with_name('snow_metrics_vs_time.png')
    fig8.tight_layout(); fig8.savefig(p8, dpi=200); saved.append(p8); plt.close(fig8)

    # Nighttime temperatures and fluxes
    night_mask = met.sun_elev_deg <= 0.0
    if np.any(night_mask):
        xn = x[night_mask]
        fig9, ax9 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        roof_n = out['T_roof'][night_mask]
        gap_n = out['T_gap'][night_mask]
        sky_n = out['T_sky'][night_mask]
        air_n = met.tair[night_mask]
        sky_n_offset = float(np.nanmean(gap_n) - np.nanmean(sky_n))
        ax9[0].plot(xn, roof_n, label='Roof (night)')
        ax9[0].plot(xn, gap_n, label='Gap air (night)')
        ax9[0].plot(xn, sky_n, label='Sky (night)')
        ax9[0].plot(xn, air_n, label='Air (night)')
        ax9[0].legend(fontsize=8); ax9[0].set_ylabel('Temperature [°C]'); ax9[0].grid(True)
        ax9[1].plot(xn, roof_n, label='Roof (night)')
        ax9[1].plot(xn, gap_n, label='Gap air (night)')
        ax9[1].plot(xn, sky_n + sky_n_offset, label=f'Sky (night) + {sky_n_offset:.1f} °C offset')
        ax9[1].plot(xn, air_n, label='Air (night)')
        ax9[1].legend(fontsize=8); ax9[1].set_ylabel('Temperature [°C]'); ax9[1].grid(True)
        ax9[2].plot(xn, out['q_lw'][night_mask], label='Roof-sky longwave flux')
        ax9[2].plot(xn, out['q_conv_roof'][night_mask], label='Roof-ambient convective flux')
        ax9[2].plot(xn, out['q_roof_gap'][night_mask], label='Roof-gap flux (+ roof→gap)')
        ax9[2].legend(fontsize=8); ax9[2].set_ylabel('Flux [W/m²]'); ax9[2].set_xlabel(xlabel); apply_month_ticks_if_needed(ax9[2], met.times, xlabel); ax9[2].grid(True)
        p9 = base.with_name('nighttime_radiative_vs_convective_fluxes.png')
        fig9.tight_layout(); fig9.savefig(p9, dpi=200); saved.append(p9); plt.close(fig9)

    # Cumulative energy budget
    fig10, ax10 = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    ax10[0].plot(x, out['cum_top_to_water_MJ'], label='Top→water cumulative energy')
    ax10[0].plot(x, out['cum_wall_external_MJ'], label='Wall external cumulative energy')
    ax10[0].plot(x, out['cum_bottom_MJ'], label='Bottom cumulative energy')
    ax10[0].legend(fontsize=8); ax10[0].set_ylabel('Energy [MJ]'); ax10[0].grid(True)
    ax10[1].plot(x, out['cum_top_cooling_MJ'], label='Top cooling contribution')
    ax10[1].plot(x, out['cum_wall_cooling_MJ'], label='Wall cooling contribution')
    ax10[1].plot(x, out['cum_bottom_cooling_MJ'], label='Bottom cooling contribution')
    ax10[1].legend(fontsize=8); ax10[1].set_ylabel('Cumulative cooling [MJ]'); ax10[1].grid(True)
    ax10[2].plot(x, out['cum_wall_solar_abs_MJ'], label='Wall solar absorbed')
    ax10[2].plot(x, out['cum_wall_solar_direct_MJ'], label='  direct-beam component')
    ax10[2].plot(x, out['cum_wall_solar_diffuse_MJ'], label='  diffuse-sky component')
    ax10[2].plot(x, out['cum_wall_solar_ground_MJ'], label='  ground-reflected component')
    ax10[2].plot(x, out['cum_wall_ambient_MJ'], label='Ambient-driven wall exchange')
    ax10[2].plot(x, out['cum_wall_external_MJ'], label='Net wall external flux')
    ax10[2].legend(fontsize=8, ncol=2); ax10[2].set_ylabel('Wall forcing [MJ]'); ax10[2].grid(True)
    ax10[3].plot(x, out['cum_wall_interface_MJ'], label='Wall-interface water flux')
    ax10[3].plot(x, out['cum_wall_ice_latent_MJ'], label='Wall-ice latent storage')
    ax10[3].plot(x, out['cum_roof_solar_abs_MJ'], label='Roof solar absorbed', alpha=0.7)
    ax10[3].plot(x, out['cum_roof_gap_MJ'], label='Roof→gap', alpha=0.7)
    ax10[3].plot(x, out['cum_gap_vent_MJ'], label='Gap→ambient', alpha=0.7)
    ax10[3].legend(fontsize=8, ncol=2); ax10[3].set_ylabel('Wall internal / roof-gap [MJ]'); ax10[3].set_xlabel(xlabel); ax10[3].grid(True)
    apply_month_ticks_if_needed(ax10[3], met.times, xlabel)
    p10 = base.with_name('cumulative_energy_budget.png')
    fig10.tight_layout(); fig10.savefig(p10, dpi=200); saved.append(p10); plt.close(fig10)

    # Sector forcing plot
    fig11, ax11 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for s in range(len(theta_deg)):
        ax11[0].plot(x, out['sector_solar_factor'][:, s], label=sector_labels[s])
        ax11[1].plot(x, out['sector_shade_factor'][:, s], label=sector_labels[s])
        ax11[2].plot(x, out['sector_wind_factor'][:, s], label=sector_labels[s])
    ax11[0].set_ylabel('Wall beam projection [-]')
    ax11[1].set_ylabel('Direct transmission [-]')
    ax11[2].set_ylabel('Effective wind speed factor [-]')
    ax11[2].set_xlabel(xlabel); apply_month_ticks_if_needed(ax11[2], met.times, xlabel)
    for a in ax11:
        a.grid(True)
    ax11[0].legend(fontsize=7, ncol=4)
    p11 = base.with_name(base.stem + '_sector_forcing.png')
    fig11.tight_layout(); fig11.savefig(p11, dpi=200); saved.append(p11); plt.close(fig11)

    return saved


# ------------------------------ Animations ---------------------------------
def _save_animation(anim: animation.FuncAnimation, path: Path, fps: int = 8) -> None:
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(path), writer=writer)
    plt.close(anim._fig)
    del anim
    gc.collect()


def make_animation_3d(met: Meteo, out: Dict[str, np.ndarray], gif_path: str) -> Path:
    gif = Path(gif_path)
    stride_hours = choose_animation_stride(met.times)
    dt_hours = max((met.times[1] - met.times[0]).total_seconds() / 3600.0, 1e-9)
    stride_steps = max(1, int(round(stride_hours / dt_hours)))
    frame_indices = np.arange(0, len(met.times), stride_steps, dtype=int)
    theta_edges = np.linspace(0, 2 * np.pi, len(out['theta_deg']) + 1)
    z = np.linspace(0, H, 24)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx: int):
        ax.cla()
        i = frame_indices[frame_idx]
        # outer wall
        theta = np.linspace(0, 2 * np.pi, 80)
        Theta, Z = np.meshgrid(theta, z)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        ax.plot_surface(X, Y, Z, alpha=0.12, linewidth=0, color='gray')
        # sector wall ice shell markers and liquid core boundary
        for s in range(len(out['theta_deg'])):
            th0 = theta_edges[s]
            th1 = theta_edges[s + 1]
            th = np.linspace(th0, th1, 12)
            Th, Zs = np.meshgrid(th, z)
            r_core = float(np.clip(np.min(out['liquid_core_radius_sector'][i, s]), MIN_LIQUID_CORE_RADIUS, R))
            r_shell = 0.5 * (R + r_core)
            Xs = r_shell * np.cos(Th)
            Ys = r_shell * np.sin(Th)
            ax.plot_surface(Xs, Ys, Zs, alpha=0.25, linewidth=0, color='lightskyblue')
            Xc = r_core * np.cos(Th)
            Yc = r_core * np.sin(Th)
            ax.plot_surface(Xc, Yc, Zs, alpha=0.28, linewidth=0, color='navy')
        # top ice wedges
        for s in range(len(out['theta_deg'])):
            zice = out['z_ice_sector'][i, s]
            if zice <= 0:
                continue
            th0 = theta_edges[s]
            th1 = theta_edges[s + 1]
            rr = np.linspace(0, R, 18)
            th = np.linspace(th0, th1, 12)
            RR, TH = np.meshgrid(rr, th)
            XX = RR * np.cos(TH)
            YY = RR * np.sin(TH)
            ZZ = np.full_like(XX, H + min(zice, 0.4))
            ax.plot_surface(XX, YY, ZZ, alpha=0.35, linewidth=0, color='cyan')
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, H + 0.5)
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
        ax.set_title(
            f"Sector tank animation\n{met.times[i].strftime('%Y-%m-%d %H:%M UTC')} | "
            f"mean top ice={out['surface_ice_equiv'][i]:.2f} m | max wall ice={out['wall_ice_max'][i]:.2f} m"
        )
        ax.view_init(elev=20, azim=35)

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=150)
    _save_animation(anim, gif)
    return gif


def make_animation_profile(met: Meteo, out: Dict[str, np.ndarray], gif_path: str) -> Path:
    gif = Path(gif_path)
    stride_hours = choose_animation_stride(met.times)
    dt_hours = max((met.times[1] - met.times[0]).total_seconds() / 3600.0, 1e-9)
    stride_steps = max(1, int(round(stride_hours / dt_hours)))
    frame_indices = np.arange(0, len(met.times), stride_steps, dtype=int)
    zc = out['z_centers']
    labels = [f"{int(td):03d}°" for td in out['theta_deg']]

    fig, ax = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    def update(frame_idx: int):
        i = frame_indices[frame_idx]
        for a in ax:
            a.cla()
        for s in range(len(labels)):
            ax[0].plot(out['T'][i, s, :], zc, label=labels[s])
            ax[1].plot(out['wall_ice'][i, s, :], zc, label=labels[s])
        ax[2].bar(np.arange(len(labels)), out['z_ice_sector'][i], tick_label=labels)
        ax[0].set_xlabel('Water T [°C]')
        ax[1].set_xlabel('Wall ice [m]')
        ax[2].set_xlabel('Sector azimuth')
        ax[0].set_ylabel('Height above bottom [m]')
        ax[2].set_ylabel('Surface ice [m]')
        ax[0].grid(True); ax[1].grid(True); ax[2].grid(True)
        ax[0].set_title('Sector temperature profiles')
        ax[1].set_title('Sector wall-ice profiles')
        ax[2].set_title('Surface ice by sector')
        if frame_idx == 0:
            ax[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{met.times[i].strftime('%Y-%m-%d %H:%M UTC')}")
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=150)
    _save_animation(anim, gif)
    return gif


def make_animation_topdown(met: Meteo, out: Dict[str, np.ndarray], gif_path: str) -> Path:
    gif = Path(gif_path)
    stride_hours = choose_animation_stride(met.times)
    dt_hours = max((met.times[1] - met.times[0]).total_seconds() / 3600.0, 1e-9)
    stride_steps = max(1, int(round(stride_hours / dt_hours)))
    frame_indices = np.arange(0, len(met.times), stride_steps, dtype=int)
    theta = np.deg2rad(out['theta_deg'])
    labels = [f"{int(td):03d}°" for td in out['theta_deg']]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')

    def update(frame_idx: int):
        i = frame_indices[frame_idx]
        ax.cla()
        width = 2 * np.pi / len(theta)
        ax.bar(theta, out['wall_ice_max'][i] * np.ones_like(theta), width=width, bottom=R - np.max(out['wall_ice'][i], axis=1), alpha=0.35, label='Wall ice shell')
        ax.bar(theta, out['liquid_core_radius_sector'][i], width=width, bottom=0.0, alpha=0.25, label='Liquid core radius')
        colors = plt.cm.cool(np.clip(out['z_ice_sector'][i] / max(np.max(out['z_ice_sector']), 1e-6), 0.0, 1.0))
        ax.bar(theta, np.full_like(theta, 0.25), width=width, bottom=R + 0.05, color=colors, alpha=0.75, label='Surface ice color scale')
        ax.set_ylim(0, R + 0.5)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels)
        ax.set_title(f"Top-down sector state\n{met.times[i].strftime('%Y-%m-%d %H:%M UTC')}")

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=150)
    _save_animation(anim, gif)
    return gif


# ---------------------------- CSV / summaries -------------------------------
def write_csvs(met: Meteo, out: Dict[str, np.ndarray], outdir: Path) -> List[Path]:
    paths: List[Path] = []
    labels = [f"sector_{int(td):03d}" for td in out['theta_deg']]

    # Core time series CSV
    p1 = outdir / 'tank_sector_timeseries.csv'
    with p1.open('w', newline='') as f:
        w = csv.writer(f)
        header = [
            'time_utc', 'air_T_C', 'wind_m_s', 'wind_dir_deg', 'ghi_W_m2', 'sun_elev_deg', 'sun_az_deg',
            'roof_T_C', 'gap_T_C', 'sky_T_C', 'roof_snow_depth_m',
            'surface_ice_equiv_m', 'surface_ice_cover_fraction', 'wall_ice_avg_m', 'wall_ice_max_m',
            'cum_top_MJ', 'cum_wall_MJ', 'cum_wall_ambient_MJ', 'cum_wall_interface_MJ', 'cum_wall_ice_latent_MJ', 'cum_wall_solar_abs_MJ', 'cum_bottom_MJ', 'cum_gap_to_water_MJ',
            'cum_roof_gap_MJ', 'cum_gap_vent_MJ', 'cum_roof_lw_MJ', 'cum_roof_conv_MJ', 'cum_roof_solar_abs_MJ',
            'freeze_budget_top_frac', 'freeze_budget_wall_frac', 'freeze_budget_bottom_frac',
        ]
        for lab in labels:
            header += [f'{lab}_top_water_C', f'{lab}_bottom_water_C', f'{lab}_surface_ice_m', f'{lab}_avg_wall_ice_m', f'{lab}_wall_flux_W_m2', f'{lab}_top_flux_W_m2']
        w.writerow(header)
        for i, t in enumerate(met.times):
            row = [
                t.isoformat(), met.tair[i], met.wind[i], met.wind_dir[i], met.ghi[i], met.sun_elev_deg[i], met.sun_az_deg[i],
                out['T_roof'][i], out['T_gap'][i], out['T_sky'][i], out['snow_depth'][i],
                out['surface_ice_equiv'][i], out['surface_ice_cover_fraction'][i], out['wall_ice_avg'][i], out['wall_ice_max'][i],
                out['cum_top_MJ'][i], out['cum_wall_MJ'][i], out['cum_wall_ambient_MJ'][i], out['cum_wall_interface_MJ'][i], out['cum_wall_ice_latent_MJ'][i], out['cum_wall_solar_abs_MJ'][i], out['cum_bottom_MJ'][i], out['cum_gap_to_water_MJ'][i],
                out['cum_roof_gap_MJ'][i], out['cum_gap_vent_MJ'][i], out['cum_roof_lw_MJ'][i], out['cum_roof_conv_MJ'][i], out['cum_roof_solar_abs_MJ'][i],
                out['freeze_budget_top_frac'][i], out['freeze_budget_wall_frac'][i], out['freeze_budget_bottom_frac'][i],
            ]
            for s in range(len(labels)):
                row += [out['T'][i, s, -1], out['T'][i, s, 0], out['z_ice_sector'][i, s], np.mean(out['wall_ice'][i, s, :]), out['q_wall_eq_sector'][i, s], out['q_top_sector'][i, s]]
            w.writerow(row)
    paths.append(p1)

    # Energy budget CSV
    p2 = outdir / 'tank_sector_energy_budget_timeseries.csv'
    with p2.open('w', newline='') as f:
        w = csv.writer(f)
        header = ['time_utc', 'cum_top_MJ', 'cum_wall_MJ', 'cum_wall_ambient_MJ', 'cum_wall_interface_MJ', 'cum_wall_ice_latent_MJ', 'cum_wall_solar_abs_MJ', 'cum_wall_solar_direct_MJ', 'cum_wall_solar_diffuse_MJ', 'cum_wall_solar_ground_MJ', 'cum_bottom_MJ', 'cum_gap_to_water_MJ', 'cum_roof_gap_MJ', 'cum_gap_vent_MJ', 'cum_roof_lw_MJ', 'cum_roof_conv_MJ', 'cum_roof_solar_abs_MJ', 'cum_top_cooling_MJ', 'cum_wall_cooling_MJ', 'cum_bottom_cooling_MJ', 'freeze_budget_top_frac', 'freeze_budget_wall_frac', 'freeze_budget_bottom_frac']
        w.writerow(header)
        for i, t in enumerate(met.times):
            w.writerow([t.isoformat(), out['cum_top_MJ'][i], out['cum_wall_MJ'][i], out['cum_wall_ambient_MJ'][i], out['cum_wall_interface_MJ'][i], out['cum_wall_ice_latent_MJ'][i], out['cum_wall_solar_abs_MJ'][i], out['cum_wall_solar_direct_MJ'][i], out['cum_wall_solar_diffuse_MJ'][i], out['cum_wall_solar_ground_MJ'][i], out['cum_bottom_MJ'][i], out['cum_gap_to_water_MJ'][i], out['cum_roof_gap_MJ'][i], out['cum_gap_vent_MJ'][i], out['cum_roof_lw_MJ'][i], out['cum_roof_conv_MJ'][i], out['cum_roof_solar_abs_MJ'][i], out['cum_top_cooling_MJ'][i], out['cum_wall_cooling_MJ'][i], out['cum_bottom_cooling_MJ'][i], out['freeze_budget_top_frac'][i], out['freeze_budget_wall_frac'][i], out['freeze_budget_bottom_frac'][i]])
    paths.append(p2)

    # Sector forcing CSV
    p3 = outdir / 'tank_sector_forcing_timeseries.csv'
    with p3.open('w', newline='') as f:
        w = csv.writer(f)
        header = ['time_utc']
        for lab in labels:
            header += [f'{lab}_beam_projection', f'{lab}_direct_transmission', f'{lab}_effective_wind_factor', f'{lab}_sky_view_factor', f'{lab}_ground_view_factor', f'{lab}_wall_shaded_height_frac', f'{lab}_wall_solar_flux_W_m2', f'{lab}_wall_solar_direct_W_m2', f'{lab}_wall_solar_diffuse_W_m2', f'{lab}_wall_solar_ground_W_m2', f'{lab}_wall_ambient_flux_W_m2', f'{lab}_wall_interface_flux_W_m2', f'{lab}_wall_ice_latent_flux_W_m2']
        w.writerow(header)
        for i, t in enumerate(met.times):
            row = [t.isoformat()]
            for s in range(len(labels)):
                row += [out['sector_solar_factor'][i, s], out['sector_shade_factor'][i, s], out['sector_wind_factor'][i, s], out['sector_sky_view_factor'][i, s], out['sector_ground_view_factor'][i, s], out['sector_shaded_height_fraction'][i, s], out['q_wall_solar_sector'][i, s], out['q_wall_solar_direct_sector'][i, s], out['q_wall_solar_diffuse_sector'][i, s], out['q_wall_solar_ground_sector'][i, s], out['q_wall_ambient_sector'][i, s], out['q_wall_interface_sector'][i, s], out['q_wall_ice_latent_sector'][i, s]]
            w.writerow(row)
    paths.append(p3)
    return paths


def write_summary(met: Meteo, out: Dict[str, np.ndarray], outdir: Path, T_init: float) -> Path:
    total_volume = np.pi * R**2 * H
    m_total = RHO_W * total_volume
    e_initial = m_total * CP_W * max(T_init, 0.0) / 1e6
    e_near_freezing = m_total * CP_W * 0.1 / 1e6
    e_freeze_all = m_total * L_F / 1e6
    e_total_to_frozen = e_initial + e_freeze_all
    p = outdir / 'tank_sector_summary.txt'
    with p.open('w') as f:
        f.write('Azimuthal-sector tank model summary\n')
        f.write(f'Edge position: {out.get("edge_position", "south")}\n')
        f.write('================================================\n\n')
        f.write(f'Run start (UTC): {met.times[0].isoformat()}\n')
        f.write(f'Run end   (UTC): {met.times[-1].isoformat()}\n')
        f.write(f'Number of sectors: {len(out["theta_deg"])}\n')
        f.write(f'Water volume [m^3]: {total_volume:.3f}\n')
        f.write(f'Initial thermal energy above 0 °C [MJ]: {e_initial:.3f}\n')
        f.write(f'Thermal energy at 0.1 °C [MJ]: {e_near_freezing:.3f}\n')
        f.write(f'Latent energy required to freeze all water [MJ]: {e_freeze_all:.3f}\n')
        f.write(f'Total energy to cool from initial condition to fully frozen [MJ]: {e_total_to_frozen:.3f}\n\n')
        median_wind = float(np.nanmedian(met.wind))
        mean_wind_fac = float(np.nanmean(out['sector_wind_factor']))
        rep_h_wall = h_wall_from_wind(median_wind * mean_wind_fac)
        rep_wall_conv_R = 1.0 / max(rep_h_wall, 1e-9)
        rep_wall_total_R = out['wall_interface_solid_resistance'] + rep_wall_conv_R
        rep_wall_U_noice = 1.0 / max(rep_wall_total_R, 1e-9)
        rep_wall_UA_noice = rep_wall_U_noice * out['wall_area_total']
        rep_top_to_water_R = out['top_membrane_resistance'] + 1.0 / H_GAP_WATER
        rep_top_to_water_U = 1.0 / max(rep_top_to_water_R, 1e-9)
        rep_top_to_water_UA = rep_top_to_water_U * out['top_area_total']
        wall_solid_share = out['wall_interface_solid_resistance'] / max(rep_wall_total_R, 1e-9)

        f.write('Material-stack assumptions used in this run:\n')
        f.write(f'  Membrane total thickness [mm]: {1000.0 * out["membrane_total_thickness_m"]:.3f}\n')
        f.write(f'  Membrane effective conductivity [W/m/K]: {out["membrane_k"]:.3f}\n')
        f.write(f'  Top membrane resistance [m²K/W]: {out["top_membrane_resistance"]:.5f}\n')
        f.write(f'  Wall geotextile thickness [mm]: {1000.0 * out["wall_geotextile_thickness_m"]:.3f}\n')
        f.write(f'  Wall geotextile effective conductivity [W/m/K]: {out["wall_geotextile_k"]:.3f}\n')
        f.write(f'  Wall geotextile resistance [m²K/W]: {out["wall_geotextile_resistance"]:.5f}\n')
        f.write(f'  Wall solid-side resistance (membrane + geotextile) [m²K/W]: {out["wall_interface_solid_resistance"]:.5f}\n\n')

        f.write('Representative no-ice conductance estimates:\n')
        f.write(f'  Median wind speed used for estimate [m/s]: {median_wind:.3f}\n')
        f.write(f'  Representative wall h [W/m²/K]: {rep_h_wall:.3f}\n')
        f.write(f'  Representative wall U (no wall ice) [W/m²/K]: {rep_wall_U_noice:.3f}\n')
        f.write(f'  Representative wall UA (no wall ice) [W/K]: {rep_wall_UA_noice:.1f}\n')
        f.write(f'  Representative top water-side U (gap to water path, no surface ice) [W/m²/K]: {rep_top_to_water_U:.3f}\n')
        f.write(f'  Representative top water-side UA (gap to water path, no surface ice) [W/K]: {rep_top_to_water_UA:.1f}\n')
        f.write(f'  Fraction of representative no-ice wall resistance from membrane+geotextile [-]: {wall_solid_share:.3f}\n\n')

        f.write('Radial shell-core transport model used in this run:\n')
        f.write(f'  Shell thickness [m]: {out["radial_shell_thickness_m"]:.3f}\n')
        f.write(f'  Base radial exchange coefficient [W/m²/K]: {out["radial_exchange_base"]:.3f}\n')
        f.write(f'  Buoyancy radial exchange coefficient factor [W/m²/K/√K]: {out["radial_exchange_buoy"]:.3f}\n')
        f.write(f'  Max radial exchange coefficient [W/m²/K]: {out["radial_exchange_max"]:.3f}\n')
        f.write(f'  Overturn radial damping factor [-]: {out["radial_overturn_mix"]:.3f}\n\n')

        f.write('Freeze-budget fractions based on cumulative cooling pathways:\n')
        f.write(f'  Top pathway fraction   [-]: {out["freeze_budget_top_frac"][-1]:.4f}\n')
        f.write(f'  Wall pathway fraction  [-]: {out["freeze_budget_wall_frac"][-1]:.4f}\n')
        f.write(f'  Bottom pathway fraction[-]: {out["freeze_budget_bottom_frac"][-1]:.4f}\n\n')
        f.write(f'Peak mean full-surface-equivalent ice thickness [m]: {np.max(out["surface_ice_equiv"]):.4f}\n')
        f.write(f'Peak sector surface ice thickness [m]: {np.max(out["z_ice_sector"]):.4f}\n')
        f.write(f'Peak average wall ice [m]: {np.max(out["wall_ice_avg"]):.4f}\n')
        f.write(f'Peak maximum wall ice [m]: {np.max(out["wall_ice_max"]):.4f}\n')
        f.write(f'Minimum liquid-core radius [m]: {np.min(out["liquid_core_radius_min"]):.4f}\n\n')
        theta = out['theta_deg']
        for s in range(len(theta)):
            f.write(f'Sector {int(theta[s]):03d}°:\n')
            f.write(f'  Peak surface ice [m]: {np.max(out["z_ice_sector"][:, s]):.4f}\n')
            f.write(f'  Peak avg wall ice [m]: {np.max(np.mean(out["wall_ice"][:, s, :], axis=1)):.4f}\n')
            f.write(f'  Mean solar factor [-]: {np.mean(out["sector_solar_factor"][:, s]):.4f}\n')
            f.write(f'  Mean direct-transmission factor [-]: {np.mean(out["sector_shade_factor"][:, s]):.4f}\n')
            f.write(f'  Mean effective wind factor [-]: {np.mean(out["sector_wind_factor"][:, s]):.4f}\n\n')
    return p


# --------------------------------- Main ------------------------------------
def main() -> int:
    global R, H, GAP_HEIGHT, MAX_WALL_ICE_THICKNESS, NEIGHBOR_CENTER_DISTANCE
    global RADIAL_SHELL_THICKNESS_M, RADIAL_EXCHANGE_BASE, RADIAL_EXCHANGE_BUOY, RADIAL_EXCHANGE_MAX, RADIAL_OVERTURN_MIX
    parser = argparse.ArgumentParser(description='Azimuthal-sector tank freezing model with array-aware directional shielding and configurable edge position')
    #parser.add_argument('--lat', type=float, default=-23.02) # this is closer to AtLast
    #parser.add_argument('--lon', type=float, default=-67.76)
    # Pathfinder -22.946148, -67.677739
    parser.add_argument('--lat', type=float, default=-22.946148)
    parser.add_argument('--lon', type=float, default=-67.677739)   
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--met-cache', type=str, default='', help='Optional local .npz cache for meteorology + derived solar forcing. If the file exists it is loaded; otherwise it is written after download.')
    parser.add_argument('--refresh-met-cache', action='store_true', help='Ignore an existing --met-cache file, re-download the meteorology, and overwrite the cache.')
    parser.add_argument('--ghi-clip-mode', choices=['off','toa','clear'], default='off', help='Optional hourly GHI cap: off=no clipping, toa=cap at TOA horizontal irradiance, clear=cap at a conservative pressure-aware clear-sky envelope.')
    parser.add_argument('--ghi-clip-factor', type=float, default=1.08, help='Multiplier applied to the clear-sky GHI envelope when --ghi-clip-mode clear. Recommended range: 1.03 to 1.10.')
    parser.add_argument('--Tinit', type=float, default=5.0)
    parser.add_argument('--n-sector', type=int, default=N_SECT_DEFAULT)
    parser.add_argument('--edge-position', choices=['south','north','west','east','interior','isolated'], default='south', help='Array-position mode: south=northern neighbors, north=southern neighbors (no tanks to the north), west=eastern neighbors, east=western neighbors, interior=neighbors on all four sides, isolated=no neighbors')
    parser.add_argument('--n-z', type=int, default=N_Z_DEFAULT)
    parser.add_argument('--membrane-thickness-mil', type=float, default=17.0, help='Total bladder / laminate thickness used as the effective membrane thickness')
    parser.add_argument('--membrane-k', type=float, default=0.22, help='Effective through-thickness conductivity of the bladder / laminate stack [W/m/K]')
    parser.add_argument('--wall-geotextile-thickness-mm', type=float, default=2.5, help='Protective wall geotextile thickness [mm]')
    parser.add_argument('--wall-geotextile-k', type=float, default=0.12, help='Effective through-thickness conductivity of the protective wall geotextile [W/m/K]')
    parser.add_argument('--radial-shell-thickness-m', type=float, default=RADIAL_SHELL_THICKNESS_M, help='Thickness of the near-wall liquid shell used for radial wall-to-core transport [m]')
    parser.add_argument('--radial-exchange-base', type=float, default=RADIAL_EXCHANGE_BASE, help='Base shell-to-core radial exchange coefficient [W/m²/K]')
    parser.add_argument('--radial-exchange-buoy', type=float, default=RADIAL_EXCHANGE_BUOY, help='Additional buoyancy-driven shell-to-core exchange coefficient factor [W/m²/K per sqrt(K)]')
    parser.add_argument('--radial-exchange-max', type=float, default=RADIAL_EXCHANGE_MAX, help='Maximum shell-to-core radial exchange coefficient [W/m²/K]')
    parser.add_argument('--radial-overturn-mix', type=float, default=RADIAL_OVERTURN_MIX, help='Fractional damping of shell-core temperature differences during vertical convective adjustment')
    parser.add_argument('--plotpng', type=str, default='freezing_timeseries_sector.png')
    parser.add_argument('--gif', type=str, default='freezing_tank_sector.gif')
    parser.add_argument('--profile-gif', type=str, default='freezing_vertical_profile_sector.gif')
    parser.add_argument('--topdown-gif', type=str, default='freezing_topdown_sector.gif')
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--output-dir', type=str, default='tank_sector_outputs')
    parser.add_argument('--tank-radius', type=float, default=R, help='Tank radius [m]')
    parser.add_argument('--tank-height', type=float, default=H, help='Tank height [m]')
    parser.add_argument('--tank-air-gap', type=float, default=GAP_HEIGHT, help='Gap Height [m]')

    
    args = parser.parse_args()

    R = max(float(args.tank_radius), MIN_LIQUID_CORE_RADIUS + 1e-6)
    H = max(float(args.tank_height), 1e-6)
    GAP_HEIGHT = max(float(args.tank_air_gap), 0.0)
    MAX_WALL_ICE_THICKNESS = max(R - MIN_LIQUID_CORE_RADIUS, 1e-6)
    NEIGHBOR_CENTER_DISTANCE = 2 * R + TANK_SPACING_EDGE_TO_EDGE
    RADIAL_SHELL_THICKNESS_M = max(float(args.radial_shell_thickness_m), 1e-3)
    RADIAL_EXCHANGE_BASE = max(float(args.radial_exchange_base), 0.0)
    RADIAL_EXCHANGE_BUOY = max(float(args.radial_exchange_buoy), 0.0)
    RADIAL_EXCHANGE_MAX = max(float(args.radial_exchange_max), RADIAL_EXCHANGE_BASE)
    RADIAL_OVERTURN_MIX = float(np.clip(args.radial_overturn_mix, 0.0, 1.0))

    print(f"[progress] Initializing array-aware sector model for {args.edge_position}-edge configuration...", flush=True)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotpng = str(outdir / args.plotpng)
    gif_path = str(outdir / args.gif)
    profile_gif = str(outdir / args.profile_gif)
    topdown_gif = str(outdir / args.topdown_gif)

    if args.met_cache:
        if Path(args.met_cache).exists() and not args.refresh_met_cache:
            print(f"[progress] Loading meteorology cache from {args.met_cache}...", flush=True)
        else:
            print(f"[progress] Downloading meteorology from Open-Meteo and caching to {args.met_cache}...", flush=True)
    else:
        print("[progress] Downloading meteorology from Open-Meteo (no local cache requested)...", flush=True)

    met, met_msg = load_or_fetch_openmeteo_archive(
        args.lat,
        args.lon,
        args.start_date,
        args.end_date,
        met_cache=args.met_cache,
        refresh_met_cache=args.refresh_met_cache,
    )
    if len(met.times) < 2:
        raise RuntimeError('Need at least two meteorological time steps')
    met = apply_ghi_sanity_clip(met, mode=args.ghi_clip_mode, clip_factor=args.ghi_clip_factor)
    print(f"[progress] {met_msg}", flush=True)
    print(f"[progress] {ghi_clip_summary(met)}", flush=True)
    print(f"[progress] Retrieved {len(met.times)} meteorology steps. Starting simulation...", flush=True)

    material_stack = MaterialStack(
        membrane_total_thickness_m=max(args.membrane_thickness_mil, 0.0) * 25.4e-6,
        membrane_k=max(args.membrane_k, 1e-6),
        wall_geotextile_thickness_m=max(args.wall_geotextile_thickness_mm, 0.0) / 1000.0,
        wall_geotextile_k=max(args.wall_geotextile_k, 1e-6),
    )

  #  out = run_model(
  #      met,
  #      T_init=args.Tinit,
  #      n_z=args.n_z,
  #      n_sector=args.n_sector,
  #      edge_position=args.edge_position,
  #      material_stack=material_stack,
  #  )

    out = run_and_plot_per_sector_MJ_from_args(met, args, plotpng, material_stack)

    print("[progress] Simulation complete. Generating plots...", flush=True)
    make_plots(met, out, plotpng)
    print("[progress] Writing CSV exports and summary...", flush=True)
    write_csvs(met, out, outdir)
    write_summary(met, out, outdir, args.Tinit)

    if not args.no_animation:
        print("[progress] Generating GIF animations...", flush=True)
        make_animation_3d(met, out, gif_path)
        make_animation_profile(met, out, profile_gif)
        make_animation_topdown(met, out, topdown_gif)

    print(f"[progress] Finished. Outputs written to {outdir}", flush=True)
    return 0


# ================== Directional Geometric Shading ==================
def directional_blocked(sun_az_deg, sector_az_deg, edge_position):
    diff = (sun_az_deg - sector_az_deg + 360) % 360
    if diff > 90 and diff < 270:
        return False
    if edge_position == 'isolated':
        return False
    if edge_position == 'south':
        return (sun_az_deg >= 315 or sun_az_deg <= 45)
    if edge_position == 'north':
        return (135 <= sun_az_deg <= 225)
    if edge_position == 'west':
        return (45 <= sun_az_deg <= 135)
    if edge_position == 'east':
        return (225 <= sun_az_deg <= 315)
    if edge_position == 'interior':
        return True
    return False

def shadow_fraction_directional(elev_deg, sun_az_deg, sector_az_deg, H, spacing, edge_position):
    blocked = directional_blocked(sun_az_deg, sector_az_deg, edge_position)
    if elev_deg <= 0:
        return 1.0
    if not blocked:
        return 0.0
    L = H / np.tan(np.deg2rad(elev_deg))
    return np.clip(L/spacing, 0.0, 1.0)

# ================== Sector Diagnostics ==================
def init_sector_diagnostics(N):
    return np.zeros(N), np.zeros(N)

def update_sector_diagnostics(i, solar_flux, q_wall, dt, cum_solar, cum_cooling):
    cum_solar[i] += solar_flux * dt
    if q_wall < 0:
        cum_cooling[i] += (-q_wall) * dt

def plot_sector_diagnostics(sector_angles, cum_solar, cum_cooling, ice, plotpng):
    import matplotlib.pyplot as plt
    base = Path(plotpng)
    ang = np.degrees(sector_angles)
    fig, ax1 = plt.subplots()
    ax1.plot(ang, cum_solar/1e9, 'o-', label='Solar (GJ)')
    ax1.plot(ang, cum_cooling/1e9, 's-', label='Cooling (GJ)')
    ax2 = ax1.twinx()
    ax2.plot(ang, ice, 'k^-', label='Ice (m)')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2)
    p20 = base.with_name('sector_diagnostics.png')
    fig.tight_layout(); fig.savefig(p20, dpi=200); plt.close(fig)
    #plt.show()


if __name__ == '__main__':
    raise SystemExit(main())
