import importlib.util
spec = importlib.util.spec_from_file_location("config", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP8.5/config.py")
#problems with config impirt ,, had to use absolute path . 

config = importlib.util.module_from_spec(spec)
import glob
import os

from tqdm import tqdm

spec.loader.exec_module(config)

import xarray as xr

import numpy as np
from SBCK import CDFt

from joblib import Parallel, delayed

from scipy.interpolate import interp1d

def cdft_cell(model_cell, obs_cell, model_times, obs_times, var_name):

    ntime = model_cell.shape[0]
    corrected_series = np.full(ntime, np.nan, dtype=np.float32)
    
    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return corrected_series

    model_dates = np.array(model_times, dtype='datetime64[D]')
    obs_dates = np.array(obs_times, dtype='datetime64[D]')
    
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")
    
    model_calib_mask = (model_dates >= calib_start) & (model_dates <= calib_end)
    obs_calib_mask = (obs_dates >= calib_start) & (obs_dates <= calib_end)
    
    common_calib_dates = np.intersect1d(model_dates[model_calib_mask], obs_dates[obs_calib_mask])
    if len(common_calib_dates) < 30:  
        return corrected_series
    
    model_calib_idx = np.in1d(model_dates, common_calib_dates)
    obs_calib_idx = np.in1d(obs_dates, common_calib_dates)
    
    calib_mod_data = model_cell[model_calib_idx]
    calib_obs_data = obs_cell[obs_calib_idx]
    



    def get_doy(dates):
        return np.array([(np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1 
                        for d in dates])
    
    calib_doys = get_doy(common_calib_dates)
    full_doys = get_doy(model_dates)

    for doy in range(1, 367):
        window_diffs = (calib_doys - doy + 366) % 366
        window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
        
        calib_mod_win = calib_mod_data[window_mask]
        calib_obs_win = calib_obs_data[window_mask]
        
        full_mask = (full_doys == doy)
        full_mod_win = model_cell[full_mask]
        
        if (calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or 
            full_mod_win.shape[0] == 0 or np.all(np.isnan(calib_mod_win)) or 
            np.all(np.isnan(calib_obs_win)) or np.all(np.isnan(full_mod_win))):
            continue
        
        valid_calib_mask = ~(np.isnan(calib_mod_win) | np.isnan(calib_obs_win))
        calib_mod_win = calib_mod_win[valid_calib_mask]
        calib_obs_win = calib_obs_win[valid_calib_mask]
        
        valid_full_mask = ~np.isnan(full_mod_win)
        valid_full_data = full_mod_win[valid_full_mask]
        
        if calib_mod_win.size < 10 or calib_obs_win.size < 10 or valid_full_data.size == 0:
            continue

        cdft = CDFt()
        cdft.fit(calib_obs_win.reshape(-1, 1), 
                 calib_mod_win.reshape(-1, 1), 
                 valid_full_data.reshape(-1, 1))
        corrected_full = cdft.predict(valid_full_data.reshape(-1, 1)).flatten()
        
        
        full_indices = np.where(full_mask)[0]
        valid_indices = full_indices[valid_full_mask]
        corrected_series[valid_indices] = corrected_full

    return corrected_series


def main():
    print("CDF-t (tas) started")

    tas_dir = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP85}/tas_Swiss"
    obs_path = f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc"
    bias_corrected_dir = f"{config.BIAS_CORRECTED_DIR}/CDFT"
    obs_ds = xr.open_dataset(obs_path)
    obs = obs_ds["TabsD"]

    tas_files = glob.glob(f"{tas_dir}/**/*.nc", recursive=True)

    for model_path in tqdm(tas_files, desc="Processing CDF-t tas"):
        print(f"Processing {model_path}")
        model_ds = xr.open_dataset(model_path)
        model = model_ds["tas"]

        ntime, nN, nE = model.shape
        qdm_data = np.full(model.shape, np.nan, dtype=np.float32)

        def process_cell(i, j):
            model_cell = model[:, i, j].values
            obs_cell = obs[:, i, j].values
            return cdft_cell(
                model_cell, obs_cell,
                model['time'].values, obs['time'].values, "tas" 
            )

        results = Parallel(n_jobs=8)(
            delayed(process_cell)(i, j)
            for i in range(nN) for j in range(nE)
        )

        idx = 0
        for i in range(nN):
            for j in range(nE):
                qdm_data[:, i, j] = results[idx]
                idx += 1

        out_ds = model_ds.copy()
        out_ds["tas"] = (("time", "N", "E"), qdm_data)

        rel_path = os.path.relpath(model_path, tas_dir)
        output_path = os.path.join(bias_corrected_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_ds.to_netcdf(output_path)

    print("CDF-t (tas) finished")

if __name__ == "__main__":
    main()