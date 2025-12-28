import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SBCK import QM
import config
import argparse
from scipy.interpolate import interp1d
import scipy.stats

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, required=True, help="City name, first letter uppercase")
parser.add_argument("--lat", type=float, required=True, help="City lat")
parser.add_argument("--lon", type=float, required=True, help="City lon")
args = parser.parse_args()

target_city = args.city
target_lat = args.lat
target_lon = args.lon

locations = {target_city: (target_lat, target_lon)}

model_paths = [
    f"{config.MODELS_DIR}/temp_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/temp_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/tmin_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmin_r01_coarse_masked.nc",
    f"{config.MODELS_DIR}/tmax_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/tmax_r01_coarse_masked.nc"
]

obs_paths = [
    f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TminD_step2_coarse.nc",
    f"{config.DATASETS_TRAINING_DIR}/TmaxD_step2_coarse.nc"
]

var_names = ["temp", "precip", "tmin", "tmax"]
obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]

output_path = f"{config.OUTPUTS_MODELS_DIR}/EQM_{target_city}_4vars_corrected.nc"

lat_vals = xr.open_dataset(model_paths[0])['lat'].values
lon_vals = xr.open_dataset(model_paths[0])['lon'].values

dist = np.sqrt((lat_vals - target_lat)**2 + (lon_vals - target_lon)**2)
i_city, j_city = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Closest grid cell to {target_city}: i={i_city}, j={j_city}")
print(f"Location: lat={lat_vals[i_city, j_city]}, lon={lon_vals[i_city, j_city]}")

calib_start = "1981-01-01"
calib_end = "2010-12-31"
scenario_start = "2011-01-01"
scenario_end = "2099-12-31"

data_vars = {}
for model_path, obs_path, var, obs_var in zip(model_paths, obs_paths, var_names, obs_var_names):
    print(f"\nProcessing {var} for {target_city}...")

    model_output = xr.open_dataset(model_path)[var]
    obs_output = xr.open_dataset(obs_path)[obs_var]

    calib_obs = obs_output.sel(time=slice(calib_start, calib_end))
    calib_mod = model_output.sel(time=slice(calib_start, calib_end))

    calib_times = calib_obs['time'].values
    calib_doys = xr.DataArray(calib_times).dt.dayofyear.values

    quantiles = np.linspace(0, 1, 101)
    model_cell = calib_mod[:, i_city, j_city].values
    obs_cell = calib_obs[:, i_city, j_city].values

    # Precip clipping to non-negative
    if var == "precip":
        model_cell = np.clip(model_cell, 0, None)
        obs_cell = np.clip(obs_cell, 0, None)

    doy_corrections = []
    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = obs_cell[window_mask]
        mod_window = model_cell[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]

        quantiles_inner = np.linspace(0.01, 0.99, 99)
        obs_q_inner = np.quantile(obs_window, quantiles_inner)
        mod_q_inner = np.quantile(mod_window, quantiles_inner)

        eqm = QM()
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))
        correction_inner = eqm.predict(mod_q_inner.reshape(-1, 1)).flatten() - mod_q_inner
        interp_corr = interp1d(
            quantiles_inner, correction_inner, kind='linear', fill_value='extrapolate'
        )
        correction = interp_corr(quantiles)
        doy_corrections.append(correction)

    doy_corrections = np.array(doy_corrections)  # (366 days, 101 quantiles)

    full_model_cell = model_output[:, i_city, j_city].values
    full_times = model_output['time'].values
    full_doys = xr.DataArray(full_times).dt.dayofyear.values

    corrected_cell = np.full_like(full_model_cell, np.nan)
    for idx, (val, doy) in enumerate(zip(full_model_cell, full_doys)):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        mod_window = model_cell[window_mask]
        obs_window = obs_cell[window_mask]
        mod_window = mod_window[~np.isnan(mod_window)]
        obs_window = obs_window[~np.isnan(obs_window)]
        if mod_window.size == 0 or obs_window.size == 0 or np.isnan(val):
            continue
        mod_q = np.quantile(mod_window, quantiles)
        value_quantile = np.interp(val, mod_q, quantiles)
        value_quantile = np.clip(value_quantile, 0, 1)
        correction_fx = doy_corrections[doy - 1]
        interp_corr = interp1d(quantiles, correction_fx, kind='linear', fill_value='extrapolate')
        corrected_val = val + interp_corr(value_quantile)
        if var == "precip":
            corrected_val = np.clip(corrected_val, 0, None)
        corrected_cell[idx] = corrected_val

    data_vars[var] = (("time", "lat", "lon"), corrected_cell.reshape(-1, 1, 1))

coords = {
    "time": full_times,
    "lat": [lat_vals[i_city, j_city]],
    "lon": [lon_vals[i_city, j_city]]
}

ds_out = xr.Dataset(data_vars, coords=coords)
ds_out.to_netcdf(output_path)
print(f"All four corrected variables saved to {output_path}")