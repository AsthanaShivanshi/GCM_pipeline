import importlib.util

spec = importlib.util.spec_from_file_location(
    "config",
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/config.py",
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import os
import warnings

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from SBCK import QM
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=DeprecationWarning)

CALIB_START = np.datetime64("1981-01-01")
CALIB_END = np.datetime64("2010-12-31")

EQM_QUANTILES = np.linspace(0.01, 0.99, 99, dtype=np.float32) #aSeparately initialised for debugging purppose.- 

QUANTILES_INNER = EQM_QUANTILES
QUANTILES_FULL = np.linspace(0.0, 1.0, 101)



def get_jobs():
    for env_var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.getenv(env_var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1) #Using 8 cores in SLURM,,, check slurm script


def get_block_size():
    value = os.getenv("EQM_BLOCK_SIZE")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return 4


def doy(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    return (dates - year_start).astype("timedelta64[D]").astype(np.int16) + 1



def valid_days(time_values):
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


def filtered_days(ds, var_name):
    da_raw = ds[var_name]
    valid_indices, valid_times = valid_days(da_raw["time"].values)

    if not valid_indices:
        return None, None

    ds_filtered = ds.isel(time=valid_indices).assign_coords(time=("time", valid_times))
    return ds_filtered, ds_filtered[var_name]




def get_tf_path(gcm, var):
    return os.path.join(
        config.ALPFINE_DIR,
        "BC/EQM",
        gcm,
        "calibration",
        var,
        "v20250415",
        f"{gcm}_{var}_EQM_transfer.nc",
    )



def eqm_cell(calib_model_common_cell, target_model_cell, obs_common_cell, ctx):
    ntime = target_model_cell.shape[0]
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    if (
        ntime == 0
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(calib_model_common_cell))
        or np.all(np.isnan(obs_common_cell))
    ):
        return qm_series

    joint_valid = (~np.isnan(calib_model_common_cell)) & (~np.isnan(obs_common_cell))
    if not np.any(joint_valid):
        return qm_series

    for doy, indices in ctx["target_indices_by_doy"].items():
        window_mask = ctx["window_masks_by_doy"][doy] & joint_valid
        if not np.any(window_mask):
            continue

        obs_window = obs_common_cell[window_mask]
        mod_window = calib_model_common_cell[window_mask]

        if obs_window.size == 0 or mod_window.size == 0:
            continue

        mod_q_inner = np.quantile(mod_window, QUANTILES_INNER)
        obs_q_inner = np.quantile(obs_window, QUANTILES_INNER)

        eqm = QM()
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))
        correction_inner = (
            eqm.predict(mod_q_inner.reshape(-1, 1)).ravel() - mod_q_inner
        )

        interp_corr = interp1d(
            QUANTILES_INNER,
            correction_inner,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )

        values = target_model_cell[indices]
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            continue

        corrected = np.full(values.shape, np.nan, dtype=np.float32)
        mod_q = np.quantile(mod_window, QUANTILES_FULL)
        value_quantiles = np.searchsorted(mod_q, values[valid_mask], side="right") / 100.0
        value_quantiles = np.clip(value_quantiles, 0.01, 0.99)

        corrected[valid_mask] = values[valid_mask] + interp_corr(value_quantiles)
        qm_series[indices] = corrected

    return qm_series


def process_block(i0, i1, hist_common, model_values, obs_common, ctx):
    ntime, _, nE = model_values.shape
    block = np.full((ntime, i1 - i0, nE), np.nan, dtype=np.float32)

    for local_i, i in enumerate(range(i0, i1)):
        for j in range(nE):
            block[:, local_i, j] = eqm_cell(
                hist_common[:, i, j],
                model_values[:, i, j],
                obs_common[:, i, j],
                ctx,
            )

    return i0, i1, block




def build_time_context(calib_model_times, target_model_times, obs_times, calib_start, calib_end):
    calib_model_dates = np.asarray(calib_model_times, dtype="datetime64[D]")
    target_model_dates = np.asarray(target_model_times, dtype="datetime64[D]")
    obs_dates = np.asarray(obs_times, dtype="datetime64[D]")

    model_calib_idx = (calib_model_dates >= calib_start) & (calib_model_dates <= calib_end)
    obs_calib_idx = (obs_dates >= calib_start) & (obs_dates <= calib_end)

    common_dates = np.intersect1d(
        calib_model_dates[model_calib_idx],
        obs_dates[obs_calib_idx],
    )

    if common_dates.size == 0:
        return None

    model_common_idx = np.isin(calib_model_dates, common_dates)
    obs_common_idx = np.isin(obs_dates, common_dates)

    calib_doys = doy(common_dates)
    target_doys = doy(target_model_dates)

    target_indices_by_doy = {
        int(day): np.where(target_doys == day)[0]
        for day in np.unique(target_doys)
    }

    window_masks_by_doy = {}
    for day in range(1, 367):
        delta = (calib_doys - day + 366) % 366
        window_masks_by_doy[day] = (delta <= 45) | (delta >= 321)

    return {
        "model_common_idx": model_common_idx,
        "obs_common_idx": obs_common_idx,
        "target_indices_by_doy": target_indices_by_doy,
        "window_masks_by_doy": window_masks_by_doy,
    }



def save_eqm_tf(hist_common, obs_common, ctx, y_coord, x_coord, out_path):
    _, nN, nE = hist_common.shape
    mod_q = np.full((366, EQM_QUANTILES.size, nN, nE), np.nan, dtype=np.float32)
    delta_q = np.full_like(mod_q, np.nan)

    for doy in range(1, 367):
        w = ctx["window_masks_by_doy"][doy]
        for i in range(nN):
            for j in range(nE):
                m = w & ~np.isnan(hist_common[:, i, j]) & ~np.isnan(obs_common[:, i, j])
                if m.sum() < 20:
                    continue

                mq = np.quantile(hist_common[m, i, j], EQM_QUANTILES)
                oq = np.quantile(obs_common[m, i, j], EQM_QUANTILES)

                mod_q[doy - 1, :, i, j] = mq
                delta_q[doy - 1, :, i, j] = oq - mq

    ds_tf = xr.Dataset(
        {
            "mod_q": (("doy", "quantile", "N", "E"), mod_q),
            "delta_q": (("doy", "quantile", "N", "E"), delta_q),
        },
        coords={
            "doy": np.arange(1, 367, dtype=np.int16),
            "quantile": EQM_QUANTILES,
            "N": y_coord,
            "E": x_coord,
        },
        attrs={
            "method": "EQM",
            "description": "Gridwise model tf stored as model quantiles and quantile deltas",
        },
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds_tf.to_netcdf(out_path)




def main():
    print("EQM all cells started")

    n_jobs = get_jobs()
    bs = get_block_size()
    print(f"Using n_jobs={n_jobs}, block_size={bs}")







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
            obs_values = obs.values.astype(np.float32, copy=False)
            obs_times = obs["time"].values
            model_var_name = model_var_map[var]

            historical_path = (
                f"{config.ALPFINE_DIR}/Bilinear/{gcm}/historical/"
                f"r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415/"
                f"{gcm}_historical_{var}_bilinear.nc"
            )

            if not os.path.exists(historical_path):
                print(f"Missing histcalib file: {historical_path}")
                continue

            hist_ds = xr.open_dataset(historical_path, decode_times=True, use_cftime=True)





            if model_var_name not in hist_ds.data_vars:
                print(f"Var '{model_var_name}' not found in {historical_path}")
                hist_ds.close()
                continue

            _, hist_da = filtered_days(hist_ds, model_var_name)


            if hist_da is None:
                print(f"No valid times in {historical_path}")
                hist_ds.close()
                continue



            hist_values = hist_da.values.astype(np.float32, copy=False)
            hist_times = hist_da["time"].values


            calib_ctx = build_time_context(
                hist_times,
                hist_times,
                obs_times,
                CALIB_START,
                CALIB_END,
            )


            if calib_ctx is None:
                print(f"No common calibration dates for {gcm} {var}")
                hist_ds.close()
                continue


            hist_common_calib = hist_values[calib_ctx["model_common_idx"], :, :]
            obs_common_calib = obs_values[calib_ctx["obs_common_idx"], :, :]


            out_tf_path = get_tf_path(gcm, var)


            if not os.path.exists(out_tf_path):

                save_eqm_tf(
                    hist_common_calib,
                    obs_common_calib,
                    calib_ctx,
                    hist_da["N"].values,
                    hist_da["E"].values,
                    out_tf_path,
                )
                print(f"Saved EQM tf to {out_tf_path}")



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

                model_ds = None

                try:
                    model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)

                    if model_var_name not in model_ds.data_vars:
                        print(f"Variable '{model_var_name}' not found in {model_path}")
                        print(f"Available variables: {list(model_ds.data_vars)}")
                        model_ds.close()
                        continue

                    model_ds_filtered, model_da = filtered_days(model_ds, model_var_name)


                    if model_da is None:
                        print(f"No valid times in {model_path}")
                        model_ds.close()
                        continue

                    model_values = model_da.values.astype(np.float32, copy=False)
                    model_times = model_da["time"].values

                    ctx = build_time_context(
                        hist_times,
                        model_times,
                        obs_times,
                        CALIB_START,
                        CALIB_END,
                    )

                    if ctx is None:
                        print(f"No common calibration dates for {model_path}")
                        model_ds.close()
                        continue

                    hist_common = hist_values[ctx["model_common_idx"], :, :]
                    obs_common = obs_values[ctx["obs_common_idx"], :, :]

                    ntime, nN, nE = model_values.shape
                    qm_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

                    blocks = [
                        (i0, min(i0 + bs, nN))
                        for i0 in range(0, nN, bs)
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
                            ctx,
                        )
                        for i0, i1 in blocks
                    )

                    for i0, i1, block in results:
                        qm_data[:, i0:i1, :] = block

                    qm_data[np.isnan(model_values)] = np.nan

                    out_ds = model_ds_filtered.copy()

                    out_ds[model_var_name] = (model_da.dims, qm_data)

                    output_path = os.path.join(
                        bias_corrected_dir,
                        f"{gcm}_{time}_{var}_EQM.nc",
                    )

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    out_ds.to_netcdf(output_path)

                    print(f"Saved BCEQM to {output_path}")
                    model_ds.close()

                except Exception as e:
                    print(f"Error for file {model_path}: {e}")
                    if model_ds is not None:
                        model_ds.close()

            hist_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()



if __name__ == "__main__":
    main()