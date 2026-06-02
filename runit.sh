
python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position isolated \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir Pathfinder_isolated \

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position west \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_west 

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position north\
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir everest_north \
  --lat 27.988157 --lon 86.925369

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position isolated \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_isolated

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_south 

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position north \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_north 

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position east \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_east

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position interior \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_interior

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position isolated \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir pampalabola_mercedes \
  --tank-radius 2.0 \
  --tank-height 1.75

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position north \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir ford_center_north \
  --lat 46.64217  --lon  -88.48135 

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir ford_center_south \
  --lat 46.64217  --lon  -88.48135 

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir everest_south \
  --lat 27.988157 --lon 86.925369

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position north\
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir everest_north \
  --lat 27.988157 --lon 86.925369

python TankThermalModel.py \
  --start-date 2024-04-01 \
  --end-date 2025-04-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir imata_south \
  --lat -15.844556 --lon -71.065750   

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position isolated \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir hawc_isolated \
  --lat 18.994931 --lon -97.308435 

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir hawc_south \  
  --lat 18.994931 --lon -97.308435  

  python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position north \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir hawc_north \  
  --lat 18.994931 --lon -97.308435  

python TankThermalModel.py \
  --start-date 2017-10-01 \
  --end-date 2017-12-01 \
  --Tinit 7.0 \
  --edge-position south \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir hawc_outrigger_nov2017 \
  --lat 18.994931 --lon -97.308435 \
  --tank-radius 0.775 \
  --tank-height 1.65  

python TankThermalModel.py \
  --start-date 2017-10-01 \
  --end-date 2017-12-01 \
  --Tinit 7.0 \
  --edge-position north\
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir hawc_outrigger_north_nov2017 \
  --lat 18.994931 --lon -97.308435 \
  --tank-radius 0.775 \
  --tank-height 1.65  

python TankThermalModel.py \
  --start-date 2024-10-01 \
  --end-date 2025-10-01 \
  --Tinit 7.0 \
  --edge-position north \
  --ghi-clip-mode clear --ghi-clip-factor 1.05 \
  --output-dir milagro_north \
  --lat 35.88105 --lon -106.67476