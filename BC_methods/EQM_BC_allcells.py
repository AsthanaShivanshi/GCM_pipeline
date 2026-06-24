import importlib.util

spec = importlib.util.spec_from_file_location("config", "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP4.5/config.py")
config = importlib.util.module_from_spec(spec)

spec.loader.exec_module(config)

import os
import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from SBCK import QM
from scipy.interpolate import interp1d
import warnings  
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def eqm_cell(calib_model_cell, target_model_cell, obs_cell, calib_start, calib_end,
             calib_model_times, target_model_times, obs_times):

    ntime = target_model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    if (
        ntime == 0
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(calib_model_cell))
        or np.all(np.isnan(obs_cell))
    ):
        return qm_series

    calib_model_dates = np.array(calib_model_times, dtype="datetime64[D]")
    target_model_dates = np.array(target_model_times, dtype="datetime64[D]")
    obs_dates = np.array(obs_times, dtype="datetime64[D]")

    calib_dates = np.arange(
        calib_start,
        calib_end + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )

    model_calib_idx = np.isin(calib_model_dates, calib_dates)
    obs_calib_idx = np.isin(obs_dates, calib_dates)
    common_dates = np.intersect1d(
        calib_model_dates[model_calib_idx],
        obs_dates[obs_calib_idx],
    )

    if len(common_dates) == 0:
        return qm_series

    model_common_idx = np.isin(calib_model_dates, common_dates)
    obs_common_idx = np.isin(obs_dates, common_dates)

    calib_mod_cell = calib_model_cell[model_common_idx]
    calib_obs_cell = obs_cell[obs_common_idx]

    joint_valid = (~np.isnan(calib_mod_cell)) & (~np.isnan(calib_obs_cell))
    calib_mod_cell = calib_mod_cell[joint_valid]
    calib_obs_cell = calib_obs_cell[joint_valid]
    common_dates = common_dates[joint_valid]

    def get_doy(d):
        return (
            np.datetime64(d, "D")
            - np.datetime64(str(d)[:4] + "-01-01", "D")
        ).astype(int) + 1

    calib_doys = np.array([get_doy(d) for d in common_dates])
    target_doys = np.array([get_doy(d) for d in target_model_dates])

    for doy in range(1, 367):
        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))

        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]

        joint_window_valid = (~np.isnan(obs_window)) & (~np.isnan(mod_window))
        obs_window = obs_window[joint_window_valid]
        mod_window = mod_window[joint_window_valid]

        if obs_window.size == 0 or mod_window.size == 0:
            continue

        quantiles_inner = np.linspace(0.01, 0.99, 99)
        mod_q_inner = np.quantile(mod_window, quantiles_inner)
        obs_q_inner = np.quantile(obs_window, quantiles_inner)

        eqm = QM()



        
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))

        correction_inner = (
            eqm.predict(mod_q_inner.reshape(-1, 1)).flatten() - mod_q_inner
        )

        interp_corr = interp1d(
            quantiles_inner,
            correction_inner,
            kind="linear",
            fill_value="extrapolate",
        )

        indices = np.where(target_doys == doy)[0]
        if indices.size == 0:
            continue

        values = target_model_cell[indices]
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            continue

        corrected = np.full(values.shape, np.nan, dtype=np.float32)
        mod_q = np.quantile(mod_window, np.linspace(0, 1, 101))
        value_quantiles = np.searchsorted(mod_q, values[valid_mask], side="right") / 100.0
        value_quantiles = np.clip(value_quantiles, 0.01, 0.99)
        corrected[valid_mask] = values[valid_mask] + interp_corr(value_quantiles)

        qm_series[indices] = corrected

    return qm_series



def main():
    print("EQM all cells started")

    obs_ds_tas = xr.open_dataset(f"{config.TARGET_DIR}/TabsD_1971_2023.nc")
    obs_ds_pr = xr.open_dataset(f"{config.TARGET_DIR}/RhiresD_1971_2023.nc")

    obs_map = {
        "tas": obs_ds_tas["TabsD"],
        "pr": obs_ds_pr["RhiresD"],
    }

    model_var_map = {
        "tas": "tas",
        "pr": "pr",
    }

    for gcm in ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]:
        for var in ["tas", "pr"]:
            obs = obs_map[var]
            model_var_name = model_var_map[var]

            historical_path = (
                f"{config.ALPFINE_DIR}/Bilinear/{gcm}/historical/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                f"{gcm}_historical_{var}_bilinear.nc"
            )

            if not os.path.exists(historical_path):
                print(f"Missing historical calibration file: {historical_path}")
                continue

            hist_ds = xr.open_dataset(historical_path, decode_times=True, use_cftime=True)
            if model_var_name not in hist_ds.data_vars:
                print(f"Variable '{model_var_name}' not found in {historical_path}")
                hist_ds.close()
                continue

            hist_da_raw = hist_ds[model_var_name]

            hist_time_values = []
            hist_valid_indices = []
            for idx, t in enumerate(hist_da_raw["time"].values):
                try:
                    dt = np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
                    hist_time_values.append(dt)
                    hist_valid_indices.append(idx)
                except ValueError:
                    continue

            if not hist_valid_indices:
                print(f"No valid historical times in {historical_path}")
                hist_ds.close()
                continue

            hist_ds_filtered = hist_ds.isel(time=hist_valid_indices).copy()
            hist_ds_filtered["time"] = np.array(hist_time_values, dtype="datetime64[D]")
            hist_da = hist_ds_filtered[model_var_name]

            for time in ["historical", "ssp370"]:
                model_path = (
                    f"{config.ALPFINE_DIR}/Bilinear/{gcm}/{time}/"
                    f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                    f"{gcm}_{time}_{var}_bilinear.nc"
                )

                bias_corrected_dir = (
                    f"{config.ALPFINE_DIR}/BC/EQM/{gcm}/{time}/"
                    f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415"
                )

                if not os.path.exists(model_path):
                    print(f"Missing target file: {model_path}")
                    continue

                print(f"Processing {model_path}")

                try:
                    model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)

                    if model_var_name not in model_ds.data_vars:
                        print(f"Variable '{model_var_name}' not found in {model_path}")
                        print(f"Available variables: {list(model_ds.data_vars)}")
                        model_ds.close()
                        continue

                    model_da_raw = model_ds[model_var_name]

                    time_values = []
                    valid_indices = []
                    for idx, t in enumerate(model_da_raw["time"].values):
                        try:
                            dt = np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
                            time_values.append(dt)
                            valid_indices.append(idx)
                        except ValueError:
                            continue

                    if not valid_indices:
                        print(f"No valid times in {model_path}")
                        model_ds.close()
                        continue

                    model_ds_filtered = model_ds.isel(time=valid_indices).copy()
                    model_ds_filtered["time"] = np.array(time_values, dtype="datetime64[D]")
                    model_da = model_ds_filtered[model_var_name]

                    ntime, nN, nE = model_da.shape
                    qm_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

                    def process_cell(i, j):
                        calib_model_cell = hist_da[:, i, j].values
                        target_model_cell = model_da[:, i, j].values
                        obs_cell = obs[:, i, j].values

                        return eqm_cell(
                            calib_model_cell,
                            target_model_cell,
                            obs_cell,
                            np.datetime64("1981-01-01"),
                            np.datetime64("2010-12-31"),
                            hist_da["time"].values,
                            model_da["time"].values,
                            obs["time"].values,
                        )

                    results = Parallel(n_jobs=8)(
                        delayed(process_cell)(i, j)
                        for i in range(nN)
                        for j in range(nE)
                    )

                    idx = 0
                    for i in range(nN):
                        for j in range(nE):
                            qm_data[:, i, j] = results[idx]
                            idx += 1

                    out_ds = model_ds_filtered.copy()

                    target_masked_qm = np.where(
                        np.isnan(model_da.values),
                        np.nan,
                        qm_data,
                    ).astype(np.float32)

                    out_ds[model_var_name] = (model_da.dims, target_masked_qm)

                    output_path = os.path.join(
                        bias_corrected_dir,
                        f"{gcm}_{time}_{var}_EQM.nc",
                    )


                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    out_ds.to_netcdf(output_path)

                    print(f"Saved BCEQM  to {output_path}")
                    model_ds.close()

                except Exception as e:
                    print(f"Error for file {model_path}: {e}")

            hist_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()



if __name__ == "__main__":
    main()


