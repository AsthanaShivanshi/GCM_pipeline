import importlib.util
spec = importlib.util.spec_from_file_location("config", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP2.6/config.py")
#problems with config impirt ,, had to use absolute path . 

config = importlib.util.module_from_spec(spec)
import glob
import os

from tqdm import tqdm

spec.loader.exec_module(config)

import xarray as xr

import numpy as np
from SBCK import QM

from joblib import Parallel, delayed
from scipy.interpolate import interp1d


def eqm_cell(model_cell, obs_cell, calib_start, calib_end, model_times, obs_times):
    
    ntime = model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)
    
    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return qm_series

    model_dates = np.array(model_times, dtype='datetime64[D]')
    obs_dates = np.array(obs_times, dtype='datetime64[D]')


    calib_dates = np.arange(calib_start, calib_end + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    
    
    model_calib_idx = np.isin(model_dates, calib_dates)
    obs_calib_idx = np.isin(obs_dates, calib_dates)
    common_dates = np.intersect1d(model_dates[model_calib_idx], obs_dates[obs_calib_idx])
    if len(common_dates) == 0:
        return qm_series

    model_common_idx = np.isin(model_dates, common_dates)
    obs_common_idx = np.isin(obs_dates, common_dates)

    calib_mod_cell = model_cell[model_common_idx]
    calib_obs_cell = obs_cell[obs_common_idx]

    
    def get_doy(d): 
        return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1
    
    calib_doys = np.array([get_doy(d) for d in common_dates])
    model_doys = np.array([get_doy(d) for d in model_dates])

    quantiles_inner = np.linspace(0.01, 0.99, 99)

    doy_corrections = []
    
    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))
        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]
        obs_window = obs_window[~np.isnan(obs_window)]
        mod_window = mod_window[~np.isnan(mod_window)]
        
        
        
        if obs_window.size == 0 or mod_window.size == 0:
            continue

        # QM on inner quantiles
        quantiles_inner = np.linspace(0.01, 0.99, 99)
        mod_q_inner = np.quantile(mod_window, quantiles_inner)
        obs_q_inner = np.quantile(obs_window, quantiles_inner)
        eqm = QM()
        
        
        
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))
        correction_inner = eqm.predict(mod_q_inner.reshape(-1, 1)).flatten() - mod_q_inner
        interp_corr = interp1d(
            quantiles_inner, correction_inner, kind='linear', fill_value='extrapolate'
        )
        
        
        quantiles = np.linspace(0, 1, 101)
        correction = interp_corr(quantiles)
        doy_corrections.append(correction)
        indices = np.where(model_doys == doy)[0]

        if indices.size > 0:
            values = model_cell[indices]
            #value ----->quantile in the model window
            mod_q = np.quantile(mod_window, np.linspace(0, 1, 101))
            value_quantiles = np.searchsorted(mod_q, values, side='right') / 100.0
            value_quantiles = np.clip(value_quantiles, 0, 1)
            # interpolated corr fx
            corrected = values + interp_corr(value_quantiles)
            qm_series[indices] = corrected

    return qm_series

def main():
    print("EQM all cells all files started")

    tas_dir = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP26}/tasmax_Swiss/"
    obs_path = f"{config.DATASETS_TRAINING_DIR}/TmaxD_step2_coarse.nc"
    bias_corrected_dir = f"{config.BIAS_CORRECTED_DIR}/EQM"

    obs_ds = xr.open_dataset(obs_path)
    obs = obs_ds["TmaxD"] 
    tas_files = glob.glob(f"{tas_dir}/**/*.nc", recursive=True)

    print(f"Found {len(tas_files)} files in {tas_dir}")

    for model_path in tqdm(tas_files, desc="Processing individual files"):
        print(f"Processing {model_path}")
        try:
#Deprecation warning : tried to fix,,, but version not available. Safely ignored. 
# 
#             
            model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)
            model = model_ds["tasmax"]

            # Filter out invalid dates
            time_values = []
            valid_indices = []
            for idx, t in enumerate(model['time'].values):
                try:
                    np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
                    time_values.append(np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}"))
                    valid_indices.append(idx)
                except ValueError:
                    continue  

            # Filter the entire dataset along the time dimension, not just the model variable
            model_ds_filtered = model_ds.isel(time=valid_indices)
            model_ds_filtered['time'] = time_values
            model = model_ds_filtered["tasmax"]

            ntime, nN, nE = model.shape
            qm_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

            def process_cell(i, j):
                model_cell = model[:, i, j].values
                obs_cell = obs[:, i, j].values
                return eqm_cell(
                    model_cell,
                    obs_cell,
                    np.datetime64("1981-01-01"),
                    np.datetime64("2010-12-31"),
                    model['time'].values,
                    obs['time'].values
                )

            # Parallel processing over grid cells
            results = Parallel(n_jobs=8)(
                delayed(process_cell)(i, j)
                for i in range(nN) for j in range(nE)
            )

            idx = 0
            for i in range(nN):
                for j in range(nE):
                    qm_data[:, i, j] = results[idx]
                    idx += 1

            # Save output - use the filtered dataset as the base
            out_ds = model_ds_filtered.copy()
            out_ds["tasmax"] = (("time", "N", "E"), qm_data)

            rel_path = os.path.relpath(model_path, tas_dir)
            output_path = os.path.join(bias_corrected_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out_ds.to_netcdf(output_path)
            print(f"BC EQM tasmax for RCP 2.6 saved to {output_path}")

        except Exception as e:
            print(f"Error for file {model_path}: {e}")


if __name__ == "__main__":
    main()