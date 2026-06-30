import os
import argparse
from pathlib import Path
import importlib.util
import warnings

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

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from SBCK import CDFt

warnings.filterwarnings("ignore", category=DeprecationWarning)




def _to_valid_datetime64(ds, var_name):
    da_raw = ds[var_name]
    time_values, valid_indices = [], []
    for idx, t in enumerate(da_raw["time"].values):
        try:
            time_values.append(np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}", "ns"))
            valid_indices.append(idx)
        except Exception:
            continue
    if not valid_indices:
        return None, None
    dsf = ds.isel(time=valid_indices).copy()
    dsf["time"] = np.array(time_values, dtype="datetime64[ns]")
    return dsf, dsf[var_name]



def cdft_cell(
    calib_model_cell,
    target_model_cell,
    obs_cell,
    calib_start,
    calib_end,
    calib_model_times,
    target_model_times,
    obs_times,
):
    ntime = target_model_cell.shape[0]
    corrected = np.full(ntime, np.nan, dtype=np.float32)

    if (
        ntime == 0
        or np.all(np.isnan(calib_model_cell))
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(obs_cell))
    ):
        return corrected

    calib_model_dates = np.array(calib_model_times, dtype="datetime64[D]")
    target_model_dates = np.array(target_model_times, dtype="datetime64[D]")
    obs_dates = np.array(obs_times, dtype="datetime64[D]")

    calib_mask_m = (calib_model_dates >= calib_start) & (calib_model_dates <= calib_end)
    calib_mask_o = (obs_dates >= calib_start) & (obs_dates <= calib_end)
    common_calib_dates = np.intersect1d(calib_model_dates[calib_mask_m], obs_dates[calib_mask_o])

    if common_calib_dates.size < 30:
        return corrected

    m_idx = np.isin(calib_model_dates, common_calib_dates)
    o_idx = np.isin(obs_dates, common_calib_dates)

    calib_mod_data = calib_model_cell[m_idx]
    calib_obs_data = obs_cell[o_idx]

    valid_joint = ~(np.isnan(calib_mod_data) | np.isnan(calib_obs_data))
    calib_mod_data = calib_mod_data[valid_joint]
    calib_obs_data = calib_obs_data[valid_joint]
    common_calib_dates = common_calib_dates[valid_joint]

    if calib_mod_data.size < 10 or calib_obs_data.size < 10:
        return corrected

    def get_doy(d):
        return (np.datetime64(d, "D") - np.datetime64(str(d)[:4] + "-01-01", "D")).astype(int) + 1

    calib_doys = np.array([get_doy(d) for d in common_calib_dates])
    target_doys = np.array([get_doy(d) for d in target_model_dates])

    for doy in range(1, 367):
        window_diffs = (calib_doys - doy + 366) % 366
        window_mask = (window_diffs <= 45) | (window_diffs >= (366 - 45))

        calib_mod_win = calib_mod_data[window_mask]
        calib_obs_win = calib_obs_data[window_mask]

        full_mask = target_doys == doy
        full_mod_win = target_model_cell[full_mask]
        valid_full = np.isfinite(full_mod_win)
        valid_full_data = full_mod_win[valid_full]

        if (
            calib_mod_win.size < 10
            or calib_obs_win.size < 10
            or valid_full_data.size == 0
            or np.unique(calib_mod_win).size < 2
            or np.unique(calib_obs_win).size < 2
            or np.unique(valid_full_data).size < 2
        ):
            continue

        model = CDFt()
        try:
            model.fit(
                calib_obs_win.reshape(-1, 1),
                calib_mod_win.reshape(-1, 1),
                valid_full_data.reshape(-1, 1),
            )
            corrected_full = model.predict(
                valid_full_data.reshape(-1, 1)
            ).flatten().astype(np.float32)
        except Exception:
            continue

        doy_idx = np.where(full_mask)[0]
        corrected[doy_idx[valid_full]] = corrected_full

    return corrected



def _resolve_model_file(base_dir, gcm, period, var):
    p = (
        base_dir / gcm / period / "r1i1p1f1" / "RegCM5-0" / "v1-r1" / "day" / var / "v20250415"
    )
    if not p.exists():
        return None
    exact = p / f"{gcm}_{period}_{var}_Swiss.nc"
    if exact.exists():
        return exact
    cands = sorted(p.glob(f"{gcm}_{period}_{var}*.nc"))
    return cands[0] if cands else None



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()
    n_jobs = max(1, args.n_jobs)

    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")
    resume = True

    root = Path(__file__).resolve().parents[3]
    obs_tas_path = root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc"
    obs_pr_path = root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc"
    model_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/Swiss"
    out_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/CDFt_Coarse"

    obs_ds_tas = xr.open_dataset(obs_tas_path)
    obs_ds_pr = xr.open_dataset(obs_pr_path)
    obs_map = {"tas": obs_ds_tas["TabsD"], "pr": obs_ds_pr["RhiresD"]}
    model_var_map = {"tas": "tas", "pr": "pr"}

    for gcm in ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]:
        for var in ["tas", "pr"]:
            obs = obs_map[var]
            model_var = model_var_map[var]

            hist_path = _resolve_model_file(model_root, gcm, "historical", var)
            if hist_path is None:
                continue

            hist_ds = xr.open_dataset(hist_path, decode_times=True, use_cftime=True)
            if model_var not in hist_ds.data_vars:
                hist_ds.close()
                continue
            hist_dsf, hist_da = _to_valid_datetime64(hist_ds, model_var)
            if hist_dsf is None:
                hist_ds.close()
                continue

            for period in ["historical", "ssp370"]:
                model_path = _resolve_model_file(model_root, gcm, period, var)
                if model_path is None:
                    continue

                rel_dir = model_path.parent.relative_to(model_root)
                out_dir = out_root / rel_dir
                src_stem = model_path.stem
                out_file = out_dir / f"{src_stem}_CDFt.nc"

                if resume and out_file.exists():
                    continue

                model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)
                if model_var not in model_ds.data_vars:
                    model_ds.close()
                    continue
                model_dsf, model_da = _to_valid_datetime64(model_ds, model_var)
                if model_dsf is None:
                    model_ds.close()
                    continue

                hist_vals = hist_da.values
                model_vals = model_da.values
                obs_vals = obs.values

                if obs_vals.shape[1:] != model_vals.shape[1:] or hist_vals.shape[1:] != model_vals.shape[1:]:
                    raise ValueError(f"Grid mismatch: obs{obs_vals.shape[1:]}, hist{hist_vals.shape[1:]}, model{model_vals.shape[1:]}")

                ntime, nN, nE = model_da.shape
                out_vals = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

                valid_cell_mask = (
                    (~np.all(np.isnan(obs_vals), axis=0))
                    & (~np.all(np.isnan(model_vals), axis=0))
                )

                hist_times = hist_da["time"].values
                model_times = model_da["time"].values
                obs_times = obs["time"].values

                def process_row(i):
                    row = np.full((ntime, nE), np.nan, dtype=np.float32)
                    for j in range(nE):
                        if not valid_cell_mask[i, j]:
                            continue
                        row[:, j] = cdft_cell(
                            hist_vals[:, i, j],
                            model_vals[:, i, j],
                            obs_vals[:, i, j],
                            calib_start,
                            calib_end,
                            hist_times,
                            model_times,
                            obs_times,
                        )
                    return i, row

                row_iter = Parallel(n_jobs=n_jobs, backend="threading", batch_size="auto", verbose=10,
                                    return_as="generator")(
                    delayed(process_row)(i) for i in range(nN)
                )
                for i, row in row_iter:
                    out_vals[:, i, :] = row

                out_vals = np.where(valid_cell_mask[np.newaxis, :, :], out_vals, np.nan).astype(np.float32)
                
                
                out_ds = model_dsf.copy()
                out_ds[model_var] = (model_da.dims, out_vals)

                os.makedirs(out_dir, exist_ok=True)


                out_ds.to_netcdf(out_file)
                out_ds.close()

                print(f"Saved: {out_file}")

                model_ds.close()

            hist_dsf.close()
            hist_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()


if __name__ == "__main__":
    main()