import xarray as xr
import numpy as np
import config
from SBCK import dOTC
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings  
warnings.filterwarnings("ignore", category=DeprecationWarning) 

var_names = ["temp", "precip", "tmin", "tmax"]
obs_var_names = ["TabsD", "RhiresD", "TminD", "TmaxD"]



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


model_datasets = [xr.open_dataset(p)[vn] for p, vn in zip(model_paths, var_names)]
obs_datasets = [xr.open_dataset(p)[ovn] for p, ovn in zip(obs_paths, obs_var_names)]




ntime, nN, nE = model_datasets[0].shape
nvars = len(var_names)
print(f"Data shape: {ntime} time steps, {nN}x{nE} grid cells, {nvars} variables")

model_times = model_datasets[0]['time'].values
obs_times = obs_datasets[0]['time'].values









def get_doy(d): 
    return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1

model_doys = np.array([get_doy(d) for d in model_times])




calib_start = np.datetime64("1981-01-01")
calib_end = np.datetime64("2010-12-31")





def process_cell(i, j):
    full_mod_cells = [ds[:, i, j].values for ds in model_datasets]  # Full period
    full_mod_stack = np.stack(full_mod_cells, axis=1)
    full_times = model_times
    full_doys = np.array([get_doy(d) for d in full_times])
    
    nan_fraction = np.isnan(full_mod_stack).sum() / full_mod_stack.size
    if nan_fraction > 0.5:
        return np.full_like(full_mod_stack, np.nan)
    
    calib_model_mask = (np.array(model_times, dtype='datetime64[D]') >= calib_start) & \
                       (np.array(model_times, dtype='datetime64[D]') <= calib_end)
    calib_obs_mask = (np.array(obs_times, dtype='datetime64[D]') >= calib_start) & \
                     (np.array(obs_times, dtype='datetime64[D]') <= calib_end)
    
    calib_mod_cells = [ds[:, i, j].values[calib_model_mask] for ds in model_datasets]
    calib_obs_cells = [ds[:, i, j].values[calib_obs_mask] for ds in obs_datasets]
    
    calib_mod_stack = np.stack(calib_mod_cells, axis=1)
    calib_obs_stack = np.stack(calib_obs_cells, axis=1)
    calib_doys = full_doys[calib_model_mask]




    if "precip" in var_names:
        precip_idx = var_names.index("precip")
        calib_obs_stack[:, precip_idx] = np.clip(calib_obs_stack[:, precip_idx], 0, None)
        calib_mod_stack[:, precip_idx] = np.clip(calib_mod_stack[:, precip_idx], 0, None)
        full_mod_stack[:, precip_idx] = np.clip(full_mod_stack[:, precip_idx], 0, None)

    full_corrected_stack = np.full_like(full_mod_stack, np.nan)




    for doy in range(1, 367):
        window_diffs = (calib_doys - doy + 366) % 366
        window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))
        calib_mod_win = calib_mod_stack[window_mask]
        calib_obs_win = calib_obs_stack[window_mask]
        
        full_mask = (full_doys == doy)
        full_mod_win_for_pred = full_mod_stack[full_mask]

        if (calib_mod_win.shape[0] == 0 or calib_obs_win.shape[0] == 0 or 
            full_mod_win_for_pred.shape[0] == 0):
            continue

        valid_calib_mask = ~(np.isnan(calib_mod_win).any(axis=1) | np.isnan(calib_obs_win).any(axis=1))
        valid_pred_mask = ~np.isnan(full_mod_win_for_pred).any(axis=1)
        
        calib_mod_win_clean = calib_mod_win[valid_calib_mask]
        calib_obs_win_clean = calib_obs_win[valid_calib_mask]
        full_mod_win_clean = full_mod_win_for_pred[valid_pred_mask]
        
        if (calib_mod_win_clean.shape[0] < 10 or 
            full_mod_win_clean.shape[0] == 0):
            continue
            
        if (np.all(calib_mod_win_clean == calib_mod_win_clean[0], axis=0).any() or 
            np.all(calib_obs_win_clean == calib_obs_win_clean[0], axis=0).any()):
            mean_diff = np.nanmean(calib_obs_win_clean, axis=0) - np.nanmean(calib_mod_win_clean, axis=0)
            corrected_full = full_mod_win_clean + mean_diff
        else:
            try:
                # dOTC.fit(Y0, X0, X1):
                # Y0 = ref obs
                # X0 = biased calib model
                # X1 = biased full series model
                dotc = dOTC(bin_width=None, bin_origin=None)
                dotc.fit(calib_obs_win_clean, calib_mod_win_clean, full_mod_win_clean)
                corrected_full = dotc.predict(full_mod_win_clean)
                
                # Validate output
                if (corrected_full is None or np.all(np.isnan(corrected_full)) or 
                    corrected_full.shape != full_mod_win_clean.shape):
                    raise ValueError("dOTC produced invalid OP")
                    
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                mean_diff = np.nanmean(calib_obs_win_clean, axis=0) - np.nanmean(calib_mod_win_clean, axis=0)
                corrected_full = full_mod_win_clean + mean_diff

        if "precip" in var_names:  #Debug step
            precip_idx = var_names.index("precip")
            corrected_full[:, precip_idx] = np.clip(corrected_full[:, precip_idx], 0, None)

        full_indices = np.where(full_mask)[0]
        valid_indices = full_indices[valid_pred_mask]
        
        for k, idx in enumerate(valid_indices):
            full_corrected_stack[idx, :] = corrected_full[k, :]

    return full_corrected_stack

print("initiating bc dotc")


cell_pairs = [(i, j) for i in range(nN) for j in range(nE)]
results = Parallel(n_jobs=8)(
    delayed(process_cell)(i, j)
    for i, j in tqdm(cell_pairs, desc="Processing cells")
)


print("Recon")
corrected_data = {var: np.full((ntime, nN, nE), np.nan, dtype=np.float32) for var in var_names}

idx = 0
for i in range(nN):
    for j in range(nE):
        cell_result = results[idx]
        if cell_result is not None and not np.all(np.isnan(cell_result)):
            for v, var in enumerate(var_names):
                corrected_data[var][:, i, j] = cell_result[:, v]
        idx += 1



for v, var_name in enumerate(var_names):
    original_model_ds = xr.open_dataset(model_paths[v])
    
    out_ds = original_model_ds.copy()
    out_ds[var_name] = (("time", "N", "E"), corrected_data[var_name])
    
    output_path = f"{config.BIAS_CORRECTED_DIR}/dOTC/{var_name}_dOTC_BC_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099_r01.nc"
    
    if 'lat' not in out_ds:
        out_ds['lat'] = (('N', 'E'), original_model_ds['lat'].values)
    if 'lon' not in out_ds:
        out_ds['lon'] = (('N', 'E'), original_model_ds['lon'].values)

    out_ds = out_ds.set_coords(['lat', 'lon'])

    out_ds.to_netcdf(output_path)
    print(f"Bias-corrected {var_name} saved to {output_path}")
    
    # Close the dataset to free memory
    original_model_ds.close()

print("diagnostics of valid points")
for var in var_names:
    non_nan_count = np.sum(~np.isnan(corrected_data[var]))
    total_count = corrected_data[var].size
    percentage = 100 * non_nan_count / total_count
    print(f"{var}: {non_nan_count}/{total_count} non-NaN values ({percentage:.1f}%)")

