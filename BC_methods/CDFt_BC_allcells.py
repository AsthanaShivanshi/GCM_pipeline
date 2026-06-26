import importlib.util
import os
import warnings

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from SBCK import CDFt

warnings.filterwarnings("ignore", category=DeprecationWarning)

spec = importlib.util.spec_from_file_location(
    "config",
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/config.py",
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

CALIB_START = np.datetime64("1981-01-01")
CALIB_END = np.datetime64("2010-12-31")


def get_n_jobs():
    for env_var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.getenv(env_var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 8


def get_block_size():
    value = os.getenv("CDFT_BLOCK_SIZE")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return 4


def to_valid_datetime64_days(time_values):
    valid_times = []
    valid_indices = []

    for idx, t in enumerate(time_values):
        try:
            dt = np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
            valid_times.append(dt)
            valid_indices.append(idx)
        except ValueError:
            continue

    return valid_indices, np.array(valid_times, dtype="datetime64[D]")


def filter_valid_times(ds, varname):
    da_raw = ds[varname]
    valid_indices, valid_times = to_valid_datetime64_days(da_raw["time"].values)

    if not valid_indices:
        return None, None

    ds_filtered = ds.isel(time=valid_indices).assign_coords(time=("time", valid_times))
    return ds_filtered, ds_filtered[varname]


def dayofyear(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    return (dates - year_start).astype("timedelta64[D]").astype(np.int16) + 1


def build_time_context(hist_times, target_times, obs_times):
    hist_times_np = np.asarray(hist_times, dtype="datetime64[D]")
    target_times_np = np.asarray(target_times, dtype="datetime64[D]")
    obs_times_np = np.asarray(obs_times, dtype="datetime64[D]")

    hist_times_calib = hist_times_np[
        (hist_times_np >= CALIB_START) & (hist_times_np <= CALIB_END)
    ]
    obs_times_calib = obs_times_np[
        (obs_times_np >= CALIB_START) & (obs_times_np <= CALIB_END)
    ]
    common_dates = np.intersect1d(hist_times_calib, obs_times_calib)

    model_common_idx = np.isin(hist_times_np, common_dates)
    obs_common_idx = np.isin(obs_times_np, common_dates)

    calib_doys = dayofyear(common_dates)
    target_doys = dayofyear(target_times_np)
    active_target_doys = np.unique(target_doys).astype(np.int16)

    full_indices_by_doy = [np.empty(0, dtype=np.int32) for _ in range(367)]
    for doy in active_target_doys:
        full_indices_by_doy[int(doy)] = np.where(target_doys == doy)[0]

    calib_window_masks = [np.zeros(common_dates.size, dtype=bool) for _ in range(367)]
    for doy in active_target_doys:
        delta = (calib_doys - doy + 366) % 366
        calib_window_masks[int(doy)] = (delta <= 45) | (delta >= 321)

    return {
        "model_common_idx": model_common_idx,
        "obs_common_idx": obs_common_idx,
        "active_target_doys": active_target_doys,
        "calib_window_masks": calib_window_masks,
        "full_indices_by_doy": full_indices_by_doy,
    }


def cdft_cell_fast(calib_model_common_cell, target_model_cell, obs_common_cell, time_meta):
    ntime = target_model_cell.shape[0]
    corrected_series = np.full(ntime, np.nan, dtype=np.float32)

    if (
        ntime == 0
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(calib_model_common_cell))
        or np.all(np.isnan(obs_common_cell))
    ):
        return corrected_series

    joint_valid = (~np.isnan(calib_model_common_cell)) & (~np.isnan(obs_common_cell))

    for doy in time_meta["active_target_doys"]:
        window_mask = time_meta["calib_window_masks"][int(doy)] & joint_valid
        calib_mod_win = calib_model_common_cell[window_mask]
        calib_obs_win = obs_common_cell[window_mask]

        target_idx = time_meta["full_indices_by_doy"][int(doy)]
        if target_idx.size == 0:
            continue

        target_values = target_model_cell[target_idx]
        valid_target_mask = ~np.isnan(target_values)
        valid_target_data = target_values[valid_target_mask]

        cdft = CDFt()
        cdft.fit(
            calib_obs_win.reshape(-1, 1),
            calib_mod_win.reshape(-1, 1),
            valid_target_data.reshape(-1, 1),
        )
        corrected_valid = cdft.predict(valid_target_data.reshape(-1, 1)).ravel()

        corrected_block = np.full(target_values.shape, np.nan, dtype=np.float32)
        corrected_block[valid_target_mask] = corrected_valid
        corrected_series[target_idx] = corrected_block

    return corrected_series


def process_block(i0, i1, hist_common, model_values, obs_common, time_meta):
    ntime, _, nE = model_values.shape
    block = np.full((ntime, i1 - i0, nE), np.nan, dtype=np.float32)

    for local_i, i in enumerate(range(i0, i1)):
        for j in range(nE):
            block[:, local_i, j] = cdft_cell_fast(
                hist_common[:, i, j],
                model_values[:, i, j],
                obs_common[:, i, j],
                time_meta,
            )

    return i0, i1, block


def main():
    print("CDF-t all cells started")

    n_jobs = get_n_jobs()
    block_size = get_block_size()
    print(f"Using n_jobs={n_jobs}, block_size={block_size}")

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
            obs_values = np.asarray(obs.values, dtype=np.float32)
            obs_times = obs["time"].values
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

            hist_ds_filtered, hist_da = filter_valid_times(hist_ds, model_var_name)
            if hist_ds_filtered is None:
                print(f"No valid historical times in {historical_path}")
                hist_ds.close()
                continue

            hist_values = np.asarray(hist_da.values, dtype=np.float32)
            hist_times = hist_da["time"].values

            for time in ["historical", "ssp370"]:
                model_path = (
                    f"{config.ALPFINE_DIR}/Bilinear/{gcm}/{time}/"
                    f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                    f"{gcm}_{time}_{var}_bilinear.nc"
                )

                bias_corrected_dir = (
                    f"{config.ALPFINE_DIR}/BC/CDFt/{gcm}/{time}/"
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

                    model_ds_filtered, model_da = filter_valid_times(model_ds, model_var_name)
                    if model_ds_filtered is None:
                        print(f"No valid times in {model_path}")
                        model_ds.close()
                        continue

                    model_values = np.asarray(model_da.values, dtype=np.float32)
                    model_times = model_da["time"].values

                    time_meta = build_time_context(
                        hist_times,
                        model_times,
                        obs_times,
                    )

                    hist_common = hist_values[time_meta["model_common_idx"], :, :]
                    obs_common = obs_values[time_meta["obs_common_idx"], :, :]

                    ntime, nN, nE = model_values.shape
                    cdft_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

                    blocks = [
                        (i0, min(i0 + block_size, nN))
                        for i0 in range(0, nN, block_size)
                    ]

                    results = Parallel(
                        n_jobs=n_jobs,
                        prefer="threads",
                    )(
                        delayed(process_block)(
                            i0,
                            i1,
                            hist_common,
                            model_values,
                            obs_common,
                            time_meta,
                        )
                        for i0, i1 in blocks
                    )

                    for i0, i1, block in results:
                        cdft_data[:, i0:i1, :] = block

                    cdft_data[np.isnan(model_values)] = np.nan

                    out_ds = model_ds_filtered.copy()
                    out_ds[model_var_name] = (model_da.dims, cdft_data)

                    output_path = os.path.join(
                        bias_corrected_dir,
                        f"{gcm}_{time}_{var}_CDFt.nc",
                    )

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    out_ds.to_netcdf(output_path)

                    print(f"Saved BCCDFt to {output_path}")
                    model_ds.close()

                except Exception as e:
                    print(f"Error for file {model_path}: {e}")

            hist_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()


if __name__ == "__main__":
    main()