import argparse
import importlib.util
import os
import warnings

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from SBCK import dOTC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

spec = importlib.util.spec_from_file_location(
    "config",
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/config.py",
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def get_doy(d):
    return (
        np.datetime64(d, "D")
        - np.datetime64(str(d)[:4] + "-01-01", "D")
    ).astype(int) + 1


def open_ds(path):


    ds = xr.open_dataset(path, decode_times=True, use_cftime=True)

    time_values = []
    valid_indices = []

    for idx, t in enumerate(ds["time"].values):
        try:
            dt = np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
            time_values.append(dt)
            valid_indices.append(idx)
        except ValueError:
            continue



    ds_filtered = ds.isel(time=valid_indices).copy()
    ds_filtered["time"] = np.array(time_values, dtype="datetime64[D]")
    ds.close()
    return ds_filtered


def metadata(hist_times, target_times, obs_times):
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")

    hist_times_np = np.asarray(hist_times, dtype="datetime64[D]")
    target_times_np = np.asarray(target_times, dtype="datetime64[D]")
    obs_times_np = np.asarray(obs_times, dtype="datetime64[D]")

    hist_times_calib = hist_times_np[
        (hist_times_np >= calib_start) & (hist_times_np <= calib_end)
    ]
    obs_times_calib = obs_times_np[
        (obs_times_np >= calib_start) & (obs_times_np <= calib_end)
    ]
    calib_dates = np.intersect1d(hist_times_calib, obs_times_calib)

    full_doys = np.fromiter(
        (get_doy(d) for d in target_times_np),
        dtype=np.int16,
        count=target_times_np.size,
    )

    full_indices_by_doy = [np.empty(0, dtype=np.int32) for _ in range(367)]
    for doy in range(1, 367):
        full_indices_by_doy[doy] = np.where(full_doys == doy)[0]

    if calib_dates.size == 0:
        return {
            "has_calibration": False,
            "calib_model_indices": np.empty(0, dtype=np.int32),
            "calib_obs_indices": np.empty(0, dtype=np.int32),
            "calib_window_masks": [np.zeros(0, dtype=bool) for _ in range(367)],
            "full_indices_by_doy": full_indices_by_doy,
        }

    calib_model_indices = np.where(np.isin(hist_times_np, calib_dates))[0]
    calib_obs_indices = np.where(np.isin(obs_times_np, calib_dates))[0]

    calib_doys = np.fromiter(
        (get_doy(d) for d in calib_dates),
        dtype=np.int16,
        count=calib_dates.size,
    )

    calib_window_masks = [np.zeros(calib_dates.size, dtype=bool) for _ in range(367)]
    for doy in range(1, 367):
        window_diffs = (calib_doys - doy + 366) % 366
        calib_window_masks[doy] = (window_diffs <= 45) | (window_diffs >= (366 - 45))

    return {
        "has_calibration": True,
        "calib_model_indices": calib_model_indices,
        "calib_obs_indices": calib_obs_indices,
        "calib_window_masks": calib_window_masks,
        "full_indices_by_doy": full_indices_by_doy,
    }


def correct_cell(
    hist_tas_calib_cell,
    hist_pr_calib_cell,
    target_tas_cell,
    target_pr_cell,
    obs_tas_calib_cell,
    obs_pr_calib_cell,
    time_meta,
):
    full_mod_stack = np.column_stack((target_tas_cell, target_pr_cell)).astype(
        np.float32, copy=False
    )
    full_corrected_stack = np.full_like(full_mod_stack, np.nan, dtype=np.float32)

    nan_fraction = np.isnan(full_mod_stack).sum() / full_mod_stack.size
    if nan_fraction > 0.7 or not time_meta["has_calibration"]:
        return full_corrected_stack

    calib_mod_stack = np.column_stack((hist_tas_calib_cell, hist_pr_calib_cell)).astype(
        np.float32, copy=False
    )
    calib_obs_stack = np.column_stack((obs_tas_calib_cell, obs_pr_calib_cell)).astype(
        np.float32, copy=False
    )

    for doy in range(1, 367):
        window_mask = time_meta["calib_window_masks"][doy]
        full_indices = time_meta["full_indices_by_doy"][doy]

        if full_indices.size == 0 or not np.any(window_mask):
            continue

        calib_mod_win = calib_mod_stack[window_mask]
        calib_obs_win = calib_obs_stack[window_mask]
        full_mod_win_for_pred = full_mod_stack[full_indices]

        if (
            calib_mod_win.shape[0] == 0
            or calib_obs_win.shape[0] == 0
            or full_mod_win_for_pred.shape[0] == 0
        ):
            continue

        valid_calib_mask = ~(
            np.isnan(calib_mod_win).any(axis=1) | np.isnan(calib_obs_win).any(axis=1)
        )
        valid_pred_mask = ~np.isnan(full_mod_win_for_pred).any(axis=1)

        calib_mod_win_clean = calib_mod_win[valid_calib_mask]
        calib_obs_win_clean = calib_obs_win[valid_calib_mask]
        full_mod_win_clean = full_mod_win_for_pred[valid_pred_mask]

        if calib_mod_win_clean.shape[0] < 10 or full_mod_win_clean.shape[0] == 0:
            continue


        dotc_model = dOTC(bin_width=None, bin_origin=None)
        dotc_model.fit(
                calib_obs_win_clean,
                    calib_mod_win_clean,
                    full_mod_win_clean,
                )
        corrected_full = dotc_model.predict(full_mod_win_clean)



        valid_indices = full_indices[valid_pred_mask]
        full_corrected_stack[valid_indices, :] = corrected_full

    return full_corrected_stack


def process_tile(
    row_start,
    row_end,
    hist_tas_calib_np,
    hist_pr_calib_np,
    target_tas_np,
    target_pr_np,
    obs_tas_calib_np,
    obs_pr_calib_np,
    time_meta,
):
    ntime = target_tas_np.shape[0]
    nE = target_tas_np.shape[2]
    nrows = row_end - row_start

    corrected_tas_tile = np.full((ntime, nrows, nE), np.nan, dtype=np.float32)
    corrected_pr_tile = np.full((ntime, nrows, nE), np.nan, dtype=np.float32)

    for local_i, i in enumerate(range(row_start, row_end)):
        for j in range(nE):
            cell_result = correct_cell(
                hist_tas_calib_np[:, i, j],
                hist_pr_calib_np[:, i, j],
                target_tas_np[:, i, j],
                target_pr_np[:, i, j],
                obs_tas_calib_np[:, i, j],
                obs_pr_calib_np[:, i, j],
                time_meta,
            )

            if not np.all(np.isnan(cell_result)):
                corrected_tas_tile[:, local_i, j] = cell_result[:, 0]
                corrected_pr_tile[:, local_i, j] = cell_result[:, 1]

    return row_start, row_end, corrected_tas_tile, corrected_pr_tile


def bivariate_dotc(
    historical_tas_path,
    historical_pr_path,
    target_tas_path,
    target_pr_path,
    obs_tas,
    obs_pr,
    out_tas_path,
    out_pr_path,
    n_jobs,
    tile_rows,
):
    print(f"Processing:\n  {target_tas_path}\n  {target_pr_path}")



    hist_tas_ds = open_ds(historical_tas_path)
    hist_pr_ds = open_ds(historical_pr_path)
    target_tas_ds = open_ds(target_tas_path)
    target_pr_ds = open_ds(target_pr_path)




    hist_tas = hist_tas_ds["tas"]
    hist_pr = hist_pr_ds["pr"]
    target_tas = target_tas_ds["tas"]
    target_pr = target_pr_ds["pr"]


    hist_tas_np = np.asarray(hist_tas.values, dtype=np.float32)
    hist_pr_np = np.asarray(hist_pr.values, dtype=np.float32)
    target_tas_np = np.asarray(target_tas.values, dtype=np.float32)
    target_pr_np = np.asarray(target_pr.values, dtype=np.float32)
    obs_tas_np = np.asarray(obs_tas.values, dtype=np.float32)
    obs_pr_np = np.asarray(obs_pr.values, dtype=np.float32)

    time_meta = metadata(
        hist_tas["time"].values,
        target_tas["time"].values,
        obs_tas["time"].values,
    )

    hist_tas_calib_np = hist_tas_np[time_meta["calib_model_indices"]]
    hist_pr_calib_np = hist_pr_np[time_meta["calib_model_indices"]]
    obs_tas_calib_np = obs_tas_np[time_meta["calib_obs_indices"]]
    obs_pr_calib_np = obs_pr_np[time_meta["calib_obs_indices"]]

    ntime, nN, nE = target_tas_np.shape
    corrected_tas = np.full((ntime, nN, nE), np.nan, dtype=np.float32)
    corrected_pr = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

    row_tiles = [
        (row_start, min(row_start + tile_rows, nN))
        for row_start in range(0, nN, tile_rows)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_tile)(
            row_start,
            row_end,
            hist_tas_calib_np,
            hist_pr_calib_np,
            target_tas_np,
            target_pr_np,
            obs_tas_calib_np,
            obs_pr_calib_np,
            time_meta,
        )
        for row_start, row_end in tqdm(row_tiles, desc="Processing row tiles")
    )


    for row_start, row_end, tas_tile, pr_tile in results:
        corrected_tas[:, row_start:row_end, :] = tas_tile
        corrected_pr[:, row_start:row_end, :] = pr_tile

    corrected_tas = np.where(np.isnan(target_tas_np), np.nan, corrected_tas).astype(
        np.float32
    )
    corrected_pr = np.where(np.isnan(target_pr_np), np.nan, corrected_pr).astype(
        np.float32
    )

    out_tas_ds = target_tas_ds.copy()
    out_tas_ds["tas"] = (target_tas.dims, corrected_tas)

    out_pr_ds = target_pr_ds.copy()
    out_pr_ds["pr"] = (target_pr.dims, corrected_pr)

    os.makedirs(os.path.dirname(out_tas_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_pr_path), exist_ok=True)

    out_tas_ds.to_netcdf(out_tas_path)
    out_pr_ds.to_netcdf(out_pr_path)

    hist_tas_ds.close()
    hist_pr_ds.close()
    target_tas_ds.close()
    target_pr_ds.close()
    out_tas_ds.close()
    out_pr_ds.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--tile_rows", type=int, default=4)
    args = parser.parse_args()

    print("dOTC all cells started")

    obs_tas_ds = open_ds(f"{config.TARGET_DIR}/TabsD_1971_2023.nc")
    obs_pr_ds = open_ds(f"{config.TARGET_DIR}/RhiresD_1971_2023.nc")

    obs_tas = obs_tas_ds["TabsD"]
    obs_pr = obs_pr_ds["RhiresD"]





    gcm_list = ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]

    for gcm in gcm_list:
        historical_tas_path = (
            f"{config.ALPFINE_DIR}/Bilinear/{gcm}/historical/"
            f"r1i1p1f1/RegCM5-0/v1-r1/day/tas/v20250415/"
            f"{gcm}_historical_tas_bilinear.nc"
        )
        historical_pr_path = (
            f"{config.ALPFINE_DIR}/Bilinear/{gcm}/historical/"
            f"r1i1p1f1/RegCM5-0/v1-r1/day/pr/v20250415/"
            f"{gcm}_historical_pr_bilinear.nc"
        )

        if not os.path.exists(historical_tas_path) or not os.path.exists(historical_pr_path):
            print(f"Missing historical calibration files for {gcm}")
            continue

        for time in ["historical", "ssp370"]:
            target_tas_path = (
                f"{config.ALPFINE_DIR}/Bilinear/{gcm}/{time}/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/tas/v20250415/"
                f"{gcm}_{time}_tas_bilinear.nc"
            )
            target_pr_path = (
                f"{config.ALPFINE_DIR}/Bilinear/{gcm}/{time}/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/pr/v20250415/"
                f"{gcm}_{time}_pr_bilinear.nc"
            )

            if not os.path.exists(target_tas_path) or not os.path.exists(target_pr_path):
                print(f"Missing target files for {gcm} {time}")
                continue

            out_tas_path = (
                f"{config.ALPFINE_DIR}/BC/dOTC/{gcm}/{time}/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/tas/v20250415/"
                f"{gcm}_{time}_tas_dOTC.nc"
            )
            out_pr_path = (
                f"{config.ALPFINE_DIR}/BC/dOTC/{gcm}/{time}/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/pr/v20250415/"
                f"{gcm}_{time}_pr_dOTC.nc"
            )

            if os.path.exists(out_tas_path) and os.path.exists(out_pr_path):
                print(f"Skipping existing outputs for {gcm} {time}")
                continue

            try:
                bivariate_dotc(
                    historical_tas_path,
                    historical_pr_path,
                    target_tas_path,
                    target_pr_path,
                    obs_tas,
                    obs_pr,
                    out_tas_path,
                    out_pr_path,
                    args.n_jobs,
                    args.tile_rows,
                )
                print(f"Saved dOTC outputs for {gcm} {time}")
            except Exception as e:
                print(f"Error for {gcm} {time}: {e}")

    obs_tas_ds.close()
    obs_pr_ds.close()

    print("dOTC all cells finished")


if __name__ == "__main__":
    main()