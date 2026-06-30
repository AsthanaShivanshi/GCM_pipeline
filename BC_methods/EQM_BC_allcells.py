import os
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
from SBCK import QM
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=DeprecationWarning)


def eqm_cell(
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
    qm_series = np.full(ntime, np.nan, dtype=np.float32)

    nquant = 99
    corr_by_doy = np.full((366, nquant), np.nan, dtype=np.float32)

    if (
        ntime == 0
        or np.all(np.isnan(target_model_cell))
        or np.all(np.isnan(calib_model_cell))
        or np.all(np.isnan(obs_cell))
    ):
        return qm_series, corr_by_doy

    calib_model_dates = np.array(calib_model_times, dtype="datetime64[D]")
    target_model_dates = np.array(target_model_times, dtype="datetime64[D]")
    obs_dates = np.array(obs_times, dtype="datetime64[D]")

    calib_dates = np.arange(
        calib_start, calib_end + np.timedelta64(1, "D"), dtype="datetime64[D]"
    )

    model_calib_idx = np.isin(calib_model_dates, calib_dates)
    obs_calib_idx = np.isin(obs_dates, calib_dates)

    common_dates = np.intersect1d(
        calib_model_dates[model_calib_idx], obs_dates[obs_calib_idx]
    )
    if len(common_dates) == 0:
        return qm_series, corr_by_doy

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
            np.datetime64(d, "D") - np.datetime64(str(d)[:4] + "-01-01", "D")
        ).astype(int) + 1

    calib_doys = np.array([get_doy(d) for d in common_dates])
    target_doys = np.array([get_doy(d) for d in target_model_dates])

    quantiles_inner = np.linspace(0.01, 0.99, 99)

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

        mod_q_inner = np.quantile(mod_window, quantiles_inner)
        obs_q_inner = np.quantile(obs_window, quantiles_inner)

        eqm = QM()
        eqm.fit(obs_q_inner.reshape(-1, 1), mod_q_inner.reshape(-1, 1))

        correction_inner = (
            eqm.predict(mod_q_inner.reshape(-1, 1)).flatten() - mod_q_inner
        ).astype(np.float32)

        corr_by_doy[doy - 1, :] = correction_inner

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
        value_quantiles = (
            np.searchsorted(mod_q, values[valid_mask], side="right") / 100.0
        )
        value_quantiles = np.clip(value_quantiles, 0.01, 0.99)
        corrected[valid_mask] = values[valid_mask] + interp_corr(value_quantiles)

        qm_series[indices] = corrected

    return qm_series, corr_by_doy


def _to_valid_datetime64(ds, var_name):
    da_raw = ds[var_name]
    time_values, valid_indices = [], []

    for idx, t in enumerate(da_raw["time"].values):
        try:
            dt = np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
            time_values.append(dt)
            valid_indices.append(idx)
        except Exception:
            continue

    if not valid_indices:
        return None, None

    ds_filtered = ds.isel(time=valid_indices).copy()
    ds_filtered["time"] = np.array(time_values, dtype="datetime64[D]")
    return ds_filtered, ds_filtered[var_name]


def _resolve_model_file(base_dir, gcm, period, var):
    preferred = (
        base_dir
        / gcm
        / period
        / "r1i1p1f1"
        / "RegCM5-0"
        / "v1-r1"
        / "day"
        / var
        / "v20250415"
    )


    if not preferred.exists():
        return None

    exact = preferred / f"{gcm}_{period}_{var}_Swiss.nc"
    
    
    if exact.exists():
        return exact

    nc_files = sorted(preferred.glob(f"{gcm}_{period}_{var}*.nc"))

    if nc_files:
        return nc_files[0]

    nc_files = sorted(preferred.glob("*.nc"))
    return nc_files[0] if nc_files else None


def main():
    print("EQM coarse-scale all cells started")

    RESUME = True
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")

    workspace_root = Path(__file__).resolve().parents[3]

    obs_tas_path = workspace_root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc"
    obs_pr_path = workspace_root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc"

    model_root = workspace_root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/Swiss"
    out_root = workspace_root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/EQM_Coarse"

    obs_ds_tas = xr.open_dataset(obs_tas_path)
    obs_ds_pr = xr.open_dataset(obs_pr_path)

    obs_map = {"tas": obs_ds_tas["TabsD"], "pr": obs_ds_pr["RhiresD"]}
    model_var_map = {"tas": "tas", "pr": "pr"}

    for gcm in ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]:


        for var in ["tas", "pr"]:
            obs = obs_map[var]
            model_var_name = model_var_map[var]

            historical_path = _resolve_model_file(model_root, gcm, "historical", var)
            
            
            if historical_path is None:
                print(f"Missing historical calibration file for {gcm} {var}")
                continue

            hist_ds = xr.open_dataset(historical_path, decode_times=True, use_cftime=True)
            if model_var_name not in hist_ds.data_vars:
                print(f"Variable '{model_var_name}' not found in {historical_path}")
                hist_ds.close()
                continue

            hist_ds_filtered, hist_da = _to_valid_datetime64(hist_ds, model_var_name)
            if hist_ds_filtered is None:
                print(f"No valid historical times in {historical_path}")
                hist_ds.close()
                continue

            for period in ["historical", "ssp370"]:
                model_path = _resolve_model_file(model_root, gcm, period, var)
                if model_path is None:
                    print(f"Missing target file for {gcm} {period} {var}")
                    continue

                rel_dir = model_path.parent.relative_to(model_root)
                out_dir = out_root / rel_dir



                src_stem = model_path.stem
                output_path = out_dir / f"{src_stem}_EQM.nc"
                corr_output_path = out_dir / f"{src_stem}_EQM_corrfx.nc"



                if RESUME and output_path.exists() and corr_output_path.exists():
                    print(f"[resume] already done, skipping: {gcm} | {var} | {period}")
                    continue

                print(f"Processing {model_path}")
                model_ds = xr.open_dataset(model_path, decode_times=True, use_cftime=True)
                if model_var_name not in model_ds.data_vars:
                    print(f"Variable '{model_var_name}' not found in {model_path}")
                    model_ds.close()
                    continue

                model_ds_filtered, model_da = _to_valid_datetime64(model_ds, model_var_name)
                if model_ds_filtered is None:
                    print(f"No valid times in {model_path}")
                    model_ds.close()
                    continue

                ntime, nN, nE = model_da.shape
                qm_data = np.full((ntime, nN, nE), np.nan, dtype=np.float32)
                corr_data = np.full((366, 99, nN, nE), np.nan, dtype=np.float32)

                hist_vals = hist_da.values
                model_vals = model_da.values
                obs_vals = obs.values
                hist_times = hist_da["time"].values
                model_times = model_da["time"].values
                obs_times = obs["time"].values

                if obs_vals.shape[1:] != model_vals.shape[1:] or hist_vals.shape[1:] != model_vals.shape[1:]:
                    raise ValueError(f"Grid mismatch: obs{obs_vals.shape[1:]}, hist{hist_vals.shape[1:]}, model{model_vals.shape[1:]}")

                valid_cell_mask = (
                    (~np.all(np.isnan(obs_vals), axis=0))
                    & (~np.all(np.isnan(model_vals), axis=0))
                )


                def process_row(i):
                    row_qm = np.full((ntime, nE), np.nan, dtype=np.float32)
                    row_corr = np.full((366, 99, nE), np.nan, dtype=np.float32)
                    for j in range(nE):
                        if not valid_cell_mask[i, j]:
                            continue
                        q, c = eqm_cell(
                            hist_vals[:, i, j],
                            model_vals[:, i, j],
                            obs_vals[:, i, j],
                            calib_start,
                            calib_end,
                            hist_times,
                            model_times,
                            obs_times,
                        )
                        row_qm[:, j] = q
                        row_corr[:, :, j] = c
                    return i, row_qm, row_corr

                rows = Parallel(
                    n_jobs=-1,
                    backend="loky",
                    batch_size="auto",
                    verbose=10,
                )(delayed(process_row)(i) for i in range(nN))

                for i, row_qm, row_corr in rows:
                    qm_data[:, i, :] = row_qm
                    corr_data[:, :, i, :] = row_corr

                out_ds = model_ds_filtered.copy()
                target_masked_qm = np.where(valid_cell_mask[None, :, :], qm_data, np.nan).astype(np.float32)
                out_ds[model_var_name] = (model_da.dims, target_masked_qm)



                os.makedirs(out_dir, exist_ok=True)
                out_ds.to_netcdf(output_path)
                print(f"Saved BC EQM to {output_path}")

                y_dim, x_dim = model_da.dims[1], model_da.dims[2]

                corr_data = np.where(valid_cell_mask[None, None, :, :], corr_data, np.nan).astype(np.float32)

                corr_ds = xr.Dataset(
                    {
                        f"{model_var_name}_correction": (
                            ("doy", "quantile", y_dim, x_dim),
                            corr_data,
                        )
                    },
                    coords={
                        "doy": np.arange(1, 367),
                        "quantile": np.linspace(0.01, 0.99, 99, dtype=np.float32),
                        y_dim: model_da[y_dim],
                        x_dim: model_da[x_dim],
                    },
                )
                corr_ds.to_netcdf(corr_output_path)
                print(f"Saved corrfx to {corr_output_path}")

                model_ds.close()

            hist_ds_filtered.close()
            hist_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()


if __name__ == "__main__":
    main()