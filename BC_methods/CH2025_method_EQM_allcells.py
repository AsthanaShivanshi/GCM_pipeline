import os
import gc
import argparse
import importlib.util
import warnings
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

spec = importlib.util.spec_from_file_location(
    "config",
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/config.py",
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import xarray as xr
import numpy as np
from joblib import Parallel, delayed
from SBCK import QM
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=DeprecationWarning)

QUANTILES_INNER = np.linspace(0.01, 0.99, 99)
QUANTILES_101 = np.linspace(0, 1, 101)
CALIB_START = np.datetime64("1981-01-01")
CALIB_END = np.datetime64("2010-12-31")


def seconds_to_minutes(seconds):
    return seconds / 60.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def to_datetime64ns(values):
    out = []
    valid = []

    for idx, t in enumerate(values):
        try:
            out.append(np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}", "ns"))
            valid.append(idx)
        except Exception:
            continue

    return np.array(out, dtype="datetime64[ns]"), valid


def get_doys(dates):
    dates = np.array(dates, dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]")
    return (dates - years).astype(int) + 1


def build_time_context(calib_model_times, target_model_times, obs_times):
    calib_model_dates = np.array(calib_model_times, dtype="datetime64[D]")
    target_model_dates = np.array(target_model_times, dtype="datetime64[D]")
    obs_dates = np.array(obs_times, dtype="datetime64[D]")

    calib_dates = np.arange(
        CALIB_START,
        CALIB_END + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )

    model_calib_idx = np.isin(calib_model_dates, calib_dates)
    obs_calib_idx = np.isin(obs_dates, calib_dates)

    common_dates = np.intersect1d(
        calib_model_dates[model_calib_idx],
        obs_dates[obs_calib_idx],
    )

    model_common_idx = np.isin(calib_model_dates, common_dates)
    obs_common_idx = np.isin(obs_dates, common_dates)

    common_doys = get_doys(common_dates)
    target_doys = get_doys(target_model_dates)
    target_indices_by_doy = [np.flatnonzero(target_doys == doy) for doy in range(1, 367)]

    return {
        "common_dates": common_dates,
        "model_common_idx": model_common_idx,
        "obs_common_idx": obs_common_idx,
        "common_doys": common_doys,
        "target_indices_by_doy": target_indices_by_doy,
    }


def eqm_cell(calib_model_cell, target_model_cell, obs_cell, time_ctx):
    ntime = target_model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    nquant = 99
    corr_by_doy = np.full((366, nquant), np.nan, dtype=np.float32)

    correction_obtaining_seconds = 0.0
    correction_applying_seconds = 0.0

    if (
        ntime == 0
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(calib_model_cell))
        or np.all(np.isnan(obs_cell))
    ):
        return (
            qm_series,
            corr_by_doy,
            correction_obtaining_seconds,
            correction_applying_seconds,
        )

    common_dates = time_ctx["common_dates"]

    if len(common_dates) == 0:
        return (
            qm_series,
            corr_by_doy,
            correction_obtaining_seconds,
            correction_applying_seconds,
        )

    calib_mod_cell = calib_model_cell[time_ctx["model_common_idx"]]
    calib_obs_cell = obs_cell[time_ctx["obs_common_idx"]]

    joint_valid = (~np.isnan(calib_mod_cell)) & (~np.isnan(calib_obs_cell))
    calib_mod_cell = calib_mod_cell[joint_valid]
    calib_obs_cell = calib_obs_cell[joint_valid]
    calib_doys = time_ctx["common_doys"][joint_valid]

    if calib_mod_cell.size == 0 or calib_obs_cell.size == 0:
        return (
            qm_series,
            corr_by_doy,
            correction_obtaining_seconds,
            correction_applying_seconds,
        )

    for doy in range(1, 367):
        obtaining_start = time.perf_counter()

        window_doys = ((calib_doys - doy + 366) % 366)
        window_mask = (window_doys <= 45) | (window_doys >= (366 - 45))

        obs_window = calib_obs_cell[window_mask]
        mod_window = calib_mod_cell[window_mask]

        joint_window_valid = (~np.isnan(obs_window)) & (~np.isnan(mod_window))
        obs_window = obs_window[joint_window_valid]
        mod_window = mod_window[joint_window_valid]

        if obs_window.size == 0 or mod_window.size == 0:
            correction_obtaining_seconds += time.perf_counter() - obtaining_start
            continue

        mod_q_inner = np.quantile(mod_window, QUANTILES_INNER)
        obs_q_inner = np.quantile(obs_window, QUANTILES_INNER)

        eqm = QM()
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))

        correction_inner = (
            eqm.predict(mod_q_inner.reshape(-1, 1)).flatten() - mod_q_inner
        ).astype(np.float32)

        corr_by_doy[doy - 1, :] = correction_inner

        interp_corr = interp1d(
            QUANTILES_INNER,
            correction_inner,
            kind="linear",
            fill_value="extrapolate",
        )

        mod_q = np.quantile(mod_window, QUANTILES_101)

        correction_obtaining_seconds += time.perf_counter() - obtaining_start

        applying_start = time.perf_counter()

        indices = time_ctx["target_indices_by_doy"][doy - 1]

        if indices.size == 0:
            correction_applying_seconds += time.perf_counter() - applying_start
            continue

        values = target_model_cell[indices]
        valid_mask = ~np.isnan(values)

        if not np.any(valid_mask):
            correction_applying_seconds += time.perf_counter() - applying_start
            continue

        corrected = np.full(values.shape, np.nan, dtype=np.float32)

        value_quantiles = np.searchsorted(mod_q, values[valid_mask], side="right") / 100.0
        value_quantiles = np.clip(value_quantiles, 0.01, 0.99)

        corrected[valid_mask] = values[valid_mask] + interp_corr(value_quantiles)

        qm_series[indices] = corrected

        correction_applying_seconds += time.perf_counter() - applying_start

    return (
        qm_series,
        corr_by_doy,
        correction_obtaining_seconds,
        correction_applying_seconds,
    )


def main():
    args = parse_args()
    n_jobs = args.n_jobs

    script_start = time.perf_counter()

    print("EQM all cells started")

    RESUME = True

    gcms = ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]
    variables = ["tas", "pr"]
    periods = ["historical", "ssp370"]

    model_var_map = {
        "tas": "tas",
        "pr": "pr",
    }

    jobs_by_gcm_var = {}

    for gcm in gcms:
        for var in variables:
            historical_path = (
                f"{config.ALPFINE_DIR}/Bilinear/{gcm}/historical/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                f"{gcm}_historical_{var}_bilinear.nc"
            )

            if not os.path.exists(historical_path):
                print(f"Missing historical calibration file: {historical_path}")
                continue

            for period in periods:
                model_path = (
                    f"{config.ALPFINE_DIR}/Bilinear/{gcm}/{period}/"
                    f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                    f"{gcm}_{period}_{var}_bilinear.nc"
                )

                bias_corrected_dir = (
                    f"{config.ALPFINE_DIR}/BC/EQM/{gcm}/{period}/"
                    f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415"
                )

                output_path = os.path.join(
                    bias_corrected_dir,
                    f"{gcm}_{period}_{var}_EQM.nc",
                )

                corr_output_path = os.path.join(
                    bias_corrected_dir,
                    f"{gcm}_{period}_{var}_EQM_corrfx.nc",
                )

                if RESUME and os.path.exists(output_path) and os.path.exists(corr_output_path):
                    print(f"[resume] already done, skipping before processing: {gcm} | {var} | {period}")
                    continue

                if not os.path.exists(model_path):
                    print(f"Missing target file: {model_path}")
                    continue

                jobs_by_gcm_var.setdefault((gcm, var), []).append(
                    {
                        "period": period,
                        "historical_path": historical_path,
                        "model_path": model_path,
                        "bias_corrected_dir": bias_corrected_dir,
                        "output_path": output_path,
                        "corr_output_path": corr_output_path,
                    }
                )

    if not jobs_by_gcm_var:
        print("All requested EQM outputs already exist. Nothing to process.")
        print(
            f"Total script runtime: "
            f"{seconds_to_minutes(time.perf_counter() - script_start):.2f} min"
        )
        return

    obs_ds_tas = xr.open_dataset(f"{config.TARGET_DIR}/TabsD_1971_2023.nc")
    obs_ds_pr = xr.open_dataset(f"{config.TARGET_DIR}/RhiresD_1971_2023.nc")

    obs_map = {
        "tas": obs_ds_tas["TabsD"].load(),
        "pr": obs_ds_pr["RhiresD"].load(),
    }

    obs_vals_map = {
        "tas": obs_map["tas"].values,
        "pr": obs_map["pr"].values,
    }

    obs_times_map = {
        "tas": obs_map["tas"]["time"].values,
        "pr": obs_map["pr"]["time"].values,
    }

    obs_grid_shape_map = {
        "tas": obs_map["tas"].shape[1:],
        "pr": obs_map["pr"].shape[1:],
    }

    for gcm in gcms:
        for var in variables:
            period_jobs = jobs_by_gcm_var.get((gcm, var))

            if not period_jobs:
                continue

            model_var_name = model_var_map[var]
            historical_path = period_jobs[0]["historical_path"]

            hist_ds = xr.open_dataset(historical_path, decode_times=True, use_cftime=True)

            if model_var_name not in hist_ds.data_vars:
                print(f"Variable '{model_var_name}' not found in {historical_path}")
                hist_ds.close()
                continue

            hist_da_raw = hist_ds[model_var_name]
            hist_time_values, hist_valid_indices = to_datetime64ns(hist_da_raw["time"].values)

            if not hist_valid_indices:
                print(f"No valid historical times in {historical_path}")
                hist_ds.close()
                continue

            hist_ds_filtered = hist_ds.isel(time=hist_valid_indices).copy()
            hist_ds_filtered["time"] = hist_time_values

            hist_da = hist_ds_filtered[model_var_name].load()
            hist_vals = hist_da.values
            hist_times = hist_da["time"].values
            hist_grid_shape = hist_da.shape[1:]

            obs_vals = obs_vals_map[var]
            obs_times = obs_times_map[var]
            obs_grid_shape = obs_grid_shape_map[var]

            for job in period_jobs:
                period = job["period"]
                model_path = job["model_path"]
                bias_corrected_dir = job["bias_corrected_dir"]
                output_path = job["output_path"]
                corr_output_path = job["corr_output_path"]

                if RESUME and os.path.exists(output_path) and os.path.exists(corr_output_path):
                    print(f"[resume] already done, skipping: {gcm} | {var} | {period}")
                    continue

                print(f"Processing {model_path}")

                run_start = time.perf_counter()
                correction_obtaining_total_seconds = 0.0
                correction_applying_total_seconds = 0.0

                model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)

                if model_var_name not in model_ds.data_vars:
                    print(f"Variable '{model_var_name}' not found in {model_path}")
                    print(f"Available variables: {list(model_ds.data_vars)}")
                    model_ds.close()
                    continue

                model_da_raw = model_ds[model_var_name]
                time_values, valid_indices = to_datetime64ns(model_da_raw["time"].values)

                if not valid_indices:
                    print(f"No valid times in {model_path}")
                    model_ds.close()
                    continue

                model_ds_filtered = model_ds.isel(time=valid_indices).copy()
                model_ds_filtered["time"] = time_values

                model_da = model_ds_filtered[model_var_name].load()
                model_vals = model_da.values
                model_times = model_da["time"].values

                if not (
                    hist_grid_shape
                    == obs_grid_shape
                    == model_da.shape[1:]
                ):
                    model_ds_filtered.close()
                    model_ds.close()
                    raise ValueError(f"Grid mismatch for {gcm} | {var} | {period}")

                ntime, nN, nE = model_da.shape

                qm_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)
                corr_data = np.full((366, 99, nN, nE), np.nan, dtype=np.float32)

                time_ctx = build_time_context(
                    hist_times,
                    model_times,
                    obs_times,
                )

                valid_cell_mask = (
                    (~np.all(np.isnan(hist_vals), axis=0))
                    & (~np.all(np.isnan(model_vals), axis=0))
                    & (~np.all(np.isnan(obs_vals), axis=0))
                )

                valid_cols_by_row = [
                    np.flatnonzero(valid_cell_mask[i])
                    for i in range(nN)
                ]

                def process_row(i):
                    row_qm = np.full((ntime, nE), np.nan, dtype=np.float32)
                    row_corr = np.full((366, 99, nE), np.nan, dtype=np.float32)

                    row_correction_obtaining_seconds = 0.0
                    row_correction_applying_seconds = 0.0

                    for j in valid_cols_by_row[i]:
                        q, c, obtain_seconds, apply_seconds = eqm_cell(
                            hist_vals[:, i, j],
                            model_vals[:, i, j],
                            obs_vals[:, i, j],
                            time_ctx,
                        )

                        row_qm[:, j] = q
                        row_corr[:, :, j] = c

                        row_correction_obtaining_seconds += obtain_seconds
                        row_correction_applying_seconds += apply_seconds

                    return (
                        i,
                        row_qm,
                        row_corr,
                        row_correction_obtaining_seconds,
                        row_correction_applying_seconds,
                    )

                row_iter = Parallel(
                    n_jobs=n_jobs,
                    backend="loky",
                    batch_size="auto",
                    verbose=10,
                    return_as="generator",
                )(
                    delayed(process_row)(i) for i in range(nN)
                )

                for (
                    i,
                    row_qm,
                    row_corr,
                    row_correction_obtaining_seconds,
                    row_correction_applying_seconds,
                ) in row_iter:
                    qm_data[:, i, :] = row_qm
                    corr_data[:, :, i, :] = row_corr

                    correction_obtaining_total_seconds += row_correction_obtaining_seconds
                    correction_applying_total_seconds += row_correction_applying_seconds

                target_masked_qm = np.where(
                    np.isnan(model_vals),
                    np.nan,
                    qm_data,
                ).astype(np.float32)

                out_ds = model_ds_filtered.copy()
                out_ds[model_var_name] = (model_da.dims, target_masked_qm)

                os.makedirs(bias_corrected_dir, exist_ok=True)

                out_ds.to_netcdf(output_path)
                print(f"Saved BCEQM to {output_path}")

                y_dim, x_dim = model_da.dims[1], model_da.dims[2]

                corr_ds = xr.Dataset(
                    {
                        f"{model_var_name}_correction": (
                            ("doy", "quantile", y_dim, x_dim),
                            corr_data,
                        )
                    },
                    coords={
                        "doy": np.arange(1, 367),
                        "quantile": QUANTILES_INNER.astype(np.float32),
                        y_dim: model_da[y_dim],
                        x_dim: model_da[x_dim],
                    },
                )

                corr_ds.to_netcdf(corr_output_path)
                print(f"Saved corrfx to {corr_output_path}")

                run_total_seconds = time.perf_counter() - run_start

                print(
                    f"Timing summary for {gcm} | {var} | {period}:\n"
                    f"  Total runtime per run: "
                    f"{seconds_to_minutes(run_total_seconds):.2f} min\n"
                    f"  Total correction-function obtaining time: "
                    f"{seconds_to_minutes(correction_obtaining_total_seconds):.2f} min\n"
                    f"  Total correction-function applying time: "
                    f"{seconds_to_minutes(correction_applying_total_seconds):.2f} min"
                )

                corr_ds.close()
                out_ds.close()
                model_ds_filtered.close()
                model_ds.close()

                del (
                    qm_data,
                    corr_data,
                    model_vals,
                    model_da,
                    target_masked_qm,
                    valid_cell_mask,
                    valid_cols_by_row,
                    time_ctx,
                )

                gc.collect()

            hist_ds_filtered.close()
            hist_ds.close()

            del hist_vals, hist_da
            gc.collect()

    obs_ds_tas.close()
    obs_ds_pr.close()

    print(
        f"Total script runtime: "
        f"{seconds_to_minutes(time.perf_counter() - script_start):.2f} min"
    )
    print("EQM all cells finished")


if __name__ == "__main__":
    main()