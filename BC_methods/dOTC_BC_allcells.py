import importlib.util

spec = importlib.util.spec_from_file_location("config", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP8.5/config.py")
config = importlib.util.module_from_spec(spec)

spec.loader.exec_module(config)

import glob
import os
from tqdm import tqdm
import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from SBCK import dOTC
import warnings  
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def get_doy(d): 
    return (np.datetime64(d, 'D') - np.datetime64(str(d)[:4] + '-01-01', 'D')).astype(int) + 1



def filter_valid_dates(ds):
    time_values = []
    valid_indices = []
    for idx, t in enumerate(ds['time'].values):
        try:
            time_values.append(np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}"))
            valid_indices.append(idx)
        except ValueError:
            continue
    ds_filtered = ds.isel(time=valid_indices)
    ds_filtered['time'] = time_values
    return ds_filtered




def bivariate_dotc(tas_path, precip_path, obs_tas_path, obs_precip_path, out_tas_path, out_precip_path):
    var_names = ["tas", "pr"]
    obs_var_names = ["TabsD", "RhiresD"]
    model_paths = [tas_path, precip_path]
    obs_paths = [obs_tas_path, obs_precip_path]

    model_datasets = []
    for p in model_paths:
        ds = xr.open_dataset(p, decode_times=True, use_cftime=True)
        ds = filter_valid_dates(ds)
        model_datasets.append(ds[var_names[model_paths.index(p)]])

    obs_datasets = []
    for p, ovn in zip(obs_paths, obs_var_names):
        ds = xr.open_dataset(p, decode_times=True, use_cftime=True)
        obs_datasets.append(ds[ovn])




    ntime, nN, nE = model_datasets[0].shape
    model_times = model_datasets[0]['time'].values
    obs_times = obs_datasets[0]['time'].values



    def process_cell(i, j):
        full_mod_cells = [ds[:, i, j].values for ds in model_datasets]
        full_mod_stack = np.stack(full_mod_cells, axis=1)
        full_times = model_times
        full_doys = np.array([get_doy(d) for d in full_times])
        nan_fraction = np.isnan(full_mod_stack).sum() / full_mod_stack.size



        if nan_fraction > 0.5:
            return np.full_like(full_mod_stack, np.nan)

        calib_start = np.datetime64("1981-01-01")
        calib_end = np.datetime64("2010-12-31")

        # All times to datetime64 
        model_times_np = np.array([np.datetime64(str(t)[:10]) for t in model_times])
        obs_times_np = np.array([np.datetime64(str(t)[:10]) for t in obs_times])

        calib_start = np.datetime64("1981-01-01")
        calib_end = np.datetime64("2010-12-31")

        model_times_calib = model_times_np[(model_times_np >= calib_start) & (model_times_np <= calib_end)]
        obs_times_calib = obs_times_np[(obs_times_np >= calib_start) & (obs_times_np <= calib_end)]
        calib_dates = np.intersect1d(model_times_calib, obs_times_calib)

        calib_model_mask = np.isin(model_times_np, calib_dates)
        calib_obs_mask = np.isin(obs_times_np, calib_dates)


        calib_doys = np.array([get_doy(d) for d in calib_dates])  # calib_dates: intersection of model/obs dates in calibration period
        full_doys = np.array([get_doy(d) for d in model_times_np])  # model_times_np: all model times

        calib_mod_cells = [ds[:, i, j].values[calib_model_mask] for ds in model_datasets]
        calib_obs_cells = [ds[:, i, j].values[calib_obs_mask] for ds in obs_datasets]
        calib_mod_stack = np.stack(calib_mod_cells, axis=1)
        calib_obs_stack = np.stack(calib_obs_cells, axis=1)
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
                    dotc = dOTC(bin_width=None, bin_origin=None)
                    dotc.fit(calib_obs_win_clean, calib_mod_win_clean, full_mod_win_clean)
                    corrected_full = dotc.predict(full_mod_win_clean)
                    if (corrected_full is None or np.all(np.isnan(corrected_full)) or 
                        corrected_full.shape != full_mod_win_clean.shape):
                        raise ValueError("invalid OP")
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    mean_diff = np.nanmean(calib_obs_win_clean, axis=0) - np.nanmean(calib_mod_win_clean, axis=0)
                    corrected_full = full_mod_win_clean + mean_diff

            full_indices = np.where(full_mask)[0]
            valid_indices = full_indices[valid_pred_mask]
            for k, idx in enumerate(valid_indices):
                full_corrected_stack[idx, :] = corrected_full[k, :]
        return full_corrected_stack

    print(f"bivariate dOTC for:\n  {tas_path}\n  {precip_path}")

    
    
    
    cell_pairs = [(i, j) for i in range(model_datasets[0].shape[1]) for j in range(model_datasets[0].shape[2])]

    results = Parallel(n_jobs=4)(
        delayed(process_cell)(i, j)
        for i, j in tqdm(cell_pairs, desc="Processing cells")
    )

    corrected_data = {var: np.full(model_datasets[0].shape, np.nan, dtype=np.float32) for var in var_names}

    idx = 0
    
    

    for i in range(model_datasets[0].shape[1]):
        for j in range(model_datasets[0].shape[2]):
            cell_result = results[idx]
            if cell_result is not None and not np.all(np.isnan(cell_result)):
                for v, var in enumerate(var_names):
                    corrected_data[var][:, i, j] = cell_result[:, v]
            idx += 1

    out_ds = xr.open_dataset(tas_path, decode_times=True, use_cftime=True)
    out_ds = filter_valid_dates(out_ds)
    out_ds["tas"] = (("time", "N", "E"), corrected_data["tas"])
    os.makedirs(os.path.dirname(out_tas_path), exist_ok=True)
    out_ds.to_netcdf(out_tas_path)
    out_ds.close()

    out_ds = xr.open_dataset(precip_path, decode_times=True, use_cftime=True)
    out_ds = filter_valid_dates(out_ds)
    out_ds["pr"] = (("time", "N", "E"), corrected_data["pr"])
    os.makedirs(os.path.dirname(out_precip_path), exist_ok=True)
    out_ds.to_netcdf(out_precip_path)
    out_ds.close()



def main():

    tas_dir = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP85}/tas_Swiss/"
    precip_dir = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP85}/pr_Swiss/"
    obs_tas_path = f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc"
    obs_precip_path = f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc"
    bc_dir = f"{config.BIAS_CORRECTED_DIR}/dOTC/"
    tas_files = sorted(glob.glob(f"{tas_dir}/**/*.nc", recursive=True))
    precip_files = sorted(glob.glob(f"{precip_dir}/**/*.nc", recursive=True))



    def get_id(path, var_prefix):


        fname = os.path.basename(path)
        return fname.replace(f"{var_prefix}_day_", "")



    precip_dict = {get_id(f, "pr"): f for f in precip_files}

    for tas_path in tqdm(tas_files, desc="Processing model chains"):
        
        
        tas_id = get_id(tas_path, "tas")
        pr_path = precip_dict.get(tas_id, None)

        if pr_path is None:

            print(f"Warning: No matching pr file for {tas_path}, skipping.")
            continue


        rel_path = os.path.relpath(tas_path, tas_dir)
        out_tas_path = os.path.join(bc_dir, "tas", rel_path)
        out_pr_path = os.path.join(bc_dir, "pr", os.path.relpath(pr_path, precip_dir))
        
        
        bivariate_dotc(
            tas_path, pr_path, obs_tas_path, obs_precip_path,
            out_tas_path, out_pr_path
        )

    print("MC finished for bivariate temp and precip for RCP8.5")

if __name__ == "__main__":
    main()