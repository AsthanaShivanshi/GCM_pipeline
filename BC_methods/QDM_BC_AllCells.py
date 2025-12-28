import xarray as xr
import numpy as np
import config
from SBCK import QDM
from joblib import Parallel, delayed

def qdm_cell(model_cell, obs_cell, model_times, obs_times, var_name):

    ntime = model_cell.shape[0]
    corrected_series = np.full(ntime, np.nan, dtype=np.float32)
    
    if ntime == 0 or np.all(np.isnan(model_cell)) or np.all(np.isnan(obs_cell)):
        return corrected_series

    # Convert times to datetime64
    model_dates = np.array(model_times, dtype='datetime64[D]')
    obs_dates = np.array(obs_times, dtype='datetime64[D]')
    
    # Define calibration period
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")
    
    # Get calibration indices
    model_calib_mask = (model_dates >= calib_start) & (model_dates <= calib_end)
    obs_calib_mask = (obs_dates >= calib_start) & (obs_dates <= calib_end)
    
    # Find common dates in calibration period
    common_calib_dates = np.intersect1d(model_dates[model_calib_mask], obs_dates[obs_calib_mask])
    if len(common_calib_dates) < 30:  # Need sufficient data
        return corrected_series
    
    # Get calibration data for common dates
    model_calib_idx = np.in1d(model_dates, common_calib_dates)
    obs_calib_idx = np.in1d(obs_dates, common_calib_dates)
    
    calib_mod_data = model_cell[model_calib_idx]
    calib_obs_data = obs_cell[obs_calib_idx]
    
    # Clip precipitation to non-negative values
    if var_name == "precip":
        calib_mod_data = np.clip(calib_mod_data, 0, None)
        calib_obs_data = np.clip(calib_obs_data, 0, None)
        model_cell = np.clip(model_cell, 0, None)
    
    # Get day of year for all data
    def get_doy(dates):
        return np.array([(np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1 
                        for d in dates])
    
    calib_doys = get_doy(common_calib_dates)
    full_doys = get_doy(model_dates)
    
    # ±45 day window 
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
        
        # NaN handling
        valid_calib_mask = ~(np.isnan(calib_mod_win) | np.isnan(calib_obs_win))
        calib_mod_win = calib_mod_win[valid_calib_mask]
        calib_obs_win = calib_obs_win[valid_calib_mask]
        
        valid_full_mask = ~np.isnan(full_mod_win)
        valid_full_data = full_mod_win[valid_full_mask]
        
        if calib_mod_win.size < 10 or calib_obs_win.size < 10 or valid_full_data.size == 0:
            continue
        
        try:
            qdm = QDM(bin_width=None, bin_origin=None)
            qdm.fit(calib_obs_win.reshape(-1, 1), 
                   calib_mod_win.reshape(-1, 1), 
                   valid_full_data.reshape(-1, 1))
            
            corrected_full = qdm.predict(valid_full_data.reshape(-1, 1)).flatten()
            
            if var_name == "precip":
                corrected_full = np.clip(corrected_full, 0, None)
            
            full_indices = np.where(full_mask)[0]
            valid_indices = full_indices[valid_full_mask]
            corrected_series[valid_indices] = corrected_full
            
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"QDM failed for DOY {doy}, var {var_name}: {e}. Using mean adjustment.")
            mean_diff = np.nanmean(calib_obs_win) - np.nanmean(calib_mod_win)
            corrected_full = valid_full_data + mean_diff
            if var_name == "precip":
                corrected_full = np.clip(corrected_full, 0, None)
            
            full_indices = np.where(full_mask)[0]
            valid_indices = full_indices[valid_full_mask]
            corrected_series[valid_indices] = corrected_full
    
    return corrected_series

def process_variable(var_name, obs_var_name):
    print(f"Processing {var_name}...")
    
    model_path = f"{config.MODELS_DIR}/{var_name}_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/{var_name}_r01_coarse_masked.nc"
    obs_path = f"{config.DATASETS_TRAINING_DIR}/{obs_var_name}_step2_coarse.nc"
    output_path = f"{config.BIAS_CORRECTED_DIR}/QDM/{var_name}_QDM_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc"

    model_ds = xr.open_dataset(model_path)
    obs_ds = xr.open_dataset(obs_path)
    model = model_ds[var_name]
    obs = obs_ds[obs_var_name]

    ntime, nN, nE = model.shape
    qdm_data = np.full(model.shape, np.nan, dtype=np.float32)

    def process_cell(i, j):
        model_cell = model[:, i, j].values
        obs_cell = obs[:, i, j].values
        return qdm_cell(
            model_cell, obs_cell,
            model['time'].values, obs['time'].values, var_name
        )

    print(f"Starting QDM correction for {var_name}")
    results = Parallel(n_jobs=8)(
        delayed(process_cell)(i, j)
        for i in range(nN) for j in range(nE)
    )

    idx = 0
    for i in range(nN):
        for j in range(nE):
            qdm_data[:, i, j] = results[idx]
            idx += 1

    # Create output dataset
    out_ds = model_ds.copy()
    out_ds[var_name] = (("time", "N", "E"), qdm_data)
    out_ds.to_netcdf(output_path)
    print(f"Bias-corrected {var_name} saved to {output_path}")

def main():
    print("QDM for All Cells started")
    
    var_names = ["temp", "precip", "tmin", "tmax"]
    obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]
    
    for var_name, obs_var_name in zip(var_names, obs_var_names):
        process_variable(var_name, obs_var_name)

    print("QDM for All Cells finished")

if __name__ == "__main__":
    main()