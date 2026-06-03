# Tank Thermal Modeling Simulation

This repository contains a Python-based simulation tool that models the thermal behavior and freezing dynamics of water storage tanks across various geographical and meteorological conditions. Overview

The core simulation engine is driven by TankThermalModel.py. To facilitate running simulations across various parameter sets and configurations, an execution script is provided via runit.sh.

The simulation generates numerous analytical plots and animations visualizing the tank's thermal profile over time. 

# Example Outputs

Pre-generated baseline simulations and example animations for specific site configurations are available in their respective directories:

/Pathfinder
/AtLast

# Getting Started Prerequisites

Ensure you have Python installed along with the required dependencies (e.g., numpy, scipy, matplotlib, argparse, csv, gc, dataclasses, datetime, pathlib, typing, mpl_toolkits.mplot3d, and requests, as required by the script). import argparse Running the Simulation

To execute a standard simulation, use the wrapper shell script or call the Python script directly. You must specify a start date, an end date, and an output directory for the results.
Example execution using the Python script directly

python TankThermalModel.py --start-date "2026-06-01" --end-date "2026-07-01" --output-dir ./my_simulation_results

# The available command line parameters are listed below:

'--lat', type=float, default=-22.946148

'--lon', type=float, default=-67.677739

'--start-date', type=str, required=True

'--end-date', type=str, required=True

'--met-cache', type=str, default='', help='Optional local .npz cache for meteorology + derived solar forcing. If the file exists it is loaded; otherwise it is written after download.'

'--refresh-met-cache', action='store_true', help='Ignore an existing --met-cache file, re-download the meteorology, and overwrite the cache.'

'--ghi-clip-mode', choices=['off','toa','clear'], default='off', help='Optional hourly GHI cap: off=no clipping, toa=cap at TOA horizontal irradiance, clear=cap at a conservative pressure-aware clear-sky envelope.'

'--ghi-clip-factor', type=float, default=1.08, help='Multiplier applied to the clear-sky GHI envelope when --ghi-clip-mode clear. Recommended range: 1.03 to 1.10.'

'--Tinit', type=float, default=5.0

'--n-sector', type=int, default=N_SECT_DEFAULT

'--edge-position', choices=['south','north','west','east','interior','isolated'], default='south', help='Array-position mode: south=northern neighbors, north=southern neighbors (no tanks to the north), west=eastern neighbors, east=western neighbors, interior=neighbors on all four sides, isolated=no neighbors'

'--n-z', type=int, default=N_Z_DEFAULT

'--membrane-thickness-mil', type=float, default=17.0, help='Total bladder / laminate thickness used as the effective membrane thickness'

'--membrane-k', type=float, default=0.22, help='Effective through-thickness conductivity of the bladder / laminate stack [W/m/K]'

'--wall-geotextile-thickness-mm', type=float, default=2.5, help='Protective wall geotextile thickness [mm]'

'--wall-geotextile-k', type=float, default=0.12, help='Effective through-thickness conductivity of the protective wall geotextile [W/m/K]'

'--radial-shell-thickness-m', type=float, default=RADIAL_SHELL_THICKNESS_M, help='Thickness of the near-wall liquid shell used for radial wall-to-core transport [m]'

'--radial-exchange-base', type=float, default=RADIAL_EXCHANGE_BASE, help='Base shell-to-core radial exchange coefficient [W/m²/K]'

'--radial-exchange-buoy', type=float, default=RADIAL_EXCHANGE_BUOY, help='Additional buoyancy-driven shell-to-core exchange coefficient factor [W/m²/K per sqrt(K)]'

'--radial-exchange-max', type=float, default=RADIAL_EXCHANGE_MAX, help='Maximum shell-to-core radial exchange coefficient [W/m²/K]'

'--radial-overturn-mix', type=float, default=RADIAL_OVERTURN_MIX, help='Fractional damping of shell-core temperature differences during vertical convective adjustment'

'--plotpng', type=str, default='freezing_timeseries_sector.png'

'--gif', type=str, default='freezing_tank_sector.gif'

'--profile-gif', type=str, default='freezing_vertical_profile_sector.gif'

'--topdown-gif', type=str, default='freezing_topdown_sector.gif'

'--no-animation', action='store_true'

'--output-dir', type=str, default='tank_sector_outputs'

'--tank-radius', type=float, default=R, help='Tank radius [m]'

'--tank-height', type=float, default=H, help='Tank height [m]'

'--tank-air-gap', type=float, default=GAP_HEIGHT, help='Gap Height [m]'
