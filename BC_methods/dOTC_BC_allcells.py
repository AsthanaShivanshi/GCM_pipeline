import os
from pathlib import Path
import importlib.util
import warnings
import argparse

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
from SBCK import dOTC

warnings.filterwarnings("ignore", category=DeprecationWarning)




def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser.parse_args()


def get_doy(d):
    return (np.datetime64(d, "D") - np.datetime64(str(d)[:4] + "-01-01", "D")).astype(int) + 1


def _to_valid_datetime64(ds, var_name):
    da = ds[var_name]
    tvals, idx = [], []
    for i, t in enumerate(da["time"].values):
        try:
            tvals.append(np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}"))
            idx.append(i)
        except Exception:
            continue
    if not idx:
        return None, None
    out = ds.isel(time=idx).copy()
    out["time"] = np.array(tvals, dtype="datetime64[D]")
    return out, out[var_name]


def _resolve_model_file(base_dir, gcm, period, var):
    p = base_dir / gcm / period / "r1i1p1f1" / "RegCM5-0" / "v1-r1" / "day" / var / "v20250415"
    if not p.exists():
        return None
    exact = p / f"{gcm}_{period}_{var}_Swiss.nc"
    if exact.exists():
        return exact
    cands = sorted(p.glob(f"{gcm}_{period}_{var}*.nc"))
    return cands[0] if cands else None


def dotc_cell(
    calib_tas, target_tas, obs_tas,
    calib_pr, target_pr, obs_pr,
    calib_start, calib_end,
    calib_times, target_times, obs_times
):
    ntime = target_tas.shape[0]
    out = np.full((ntime, 2), np.nan, dtype=np.float32)

    full_mod = np.stack([target_tas, target_pr], axis=1)
    if np.isnan(full_mod).sum() / full_mod.size > 0.7:
        return out[:, 0], out[:, 1]

    calib_times_np = np.array(calib_times, dtype="datetime64[D]")
    target_times_np = np.array(target_times, dtype="datetime64[D]")
    obs_times_np = np.array(obs_times, dtype="datetime64[D]")

    m_cal = calib_times_np[(calib_times_np >= calib_start) & (calib_times_np <= calib_end)]
    o_cal = obs_times_np[(obs_times_np >= calib_start) & (obs_times_np <= calib_end)]
    calib_dates = np.intersect1d(m_cal, o_cal)
    if calib_dates.size < 30:
        return out[:, 0], out[:, 1]

    m_mask = np.isin(calib_times_np, calib_dates)
    o_mask = np.isin(obs_times_np, calib_dates)

    calib_mod = np.stack([calib_tas[m_mask], calib_pr[m_mask]], axis=1)
    calib_obs = np.stack([obs_tas[o_mask], obs_pr[o_mask]], axis=1)

    calib_doys = np.array([get_doy(d) for d in calib_dates])
    full_doys = np.array([get_doy(d) for d in target_times_np])

    for doy in range(1, 367):
        wd = (calib_doys - doy + 366) % 366
        wmask = (wd <= 45) | (wd >= (366 - 45))

        cm = calib_mod[wmask]
        co = calib_obs[wmask]
        fmask = full_doys == doy
        fm = full_mod[fmask]

        if cm.shape[0] == 0 or co.shape[0] == 0 or fm.shape[0] == 0:
            continue

        valid_cal = ~(np.isnan(cm).any(axis=1) | np.isnan(co).any(axis=1))
        valid_pred = ~np.isnan(fm).any(axis=1)

        cm = cm[valid_cal]
        co = co[valid_cal]
        fm_clean = fm[valid_pred]

        if cm.shape[0] < 10 or fm_clean.shape[0] == 0:
            continue

        if (np.all(cm == cm[0], axis=0).any() or np.all(co == co[0], axis=0).any()):
            corrected = fm_clean + (np.nanmean(co, axis=0) - np.nanmean(cm, axis=0))
        else:
            try:
                model = dOTC(bin_width=None, bin_origin=None)
                model.fit(co, cm, fm_clean)
                corrected = model.predict(fm_clean)
                if corrected is None or np.all(np.isnan(corrected)) or corrected.shape != fm_clean.shape:
                    raise ValueError("invalid dOTC output")
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                corrected = fm_clean + (np.nanmean(co, axis=0) - np.nanmean(cm, axis=0))

        full_idx = np.where(fmask)[0]
        valid_idx = full_idx[valid_pred]
        out[valid_idx, :] = corrected

    return out[:, 0], out[:, 1]


def main():
    print("dOTC coarse all cells started")

    args = parse_args()

    RESUME = True
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")
    n_jobs = max(1, args.n_jobs)

    root = Path(__file__).resolve().parents[3]
    obs_tas_path = root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc"
    obs_pr_path = root / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step2_coarse.nc"
    model_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/Swiss"
    out_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/dOTC_Coarse"

    obs_ds_tas = xr.open_dataset(obs_tas_path)
    obs_ds_pr = xr.open_dataset(obs_pr_path)
    obs_tas = obs_ds_tas["TabsD"]
    obs_pr = obs_ds_pr["RhiresD"]

    gcms = ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]

    for gcm in gcms:
        hist_tas_path = _resolve_model_file(model_root, gcm, "historical", "tas")
        hist_pr_path = _resolve_model_file(model_root, gcm, "historical", "pr")
        if hist_tas_path is None or hist_pr_path is None:
            continue

        htas_ds = xr.open_dataset(hist_tas_path, decode_times=True, use_cftime=True)
        hpr_ds = xr.open_dataset(hist_pr_path, decode_times=True, use_cftime=True)
        htas_dsf, htas = _to_valid_datetime64(htas_ds, "tas")
        hpr_dsf, hpr = _to_valid_datetime64(hpr_ds, "pr")
        if htas_dsf is None or hpr_dsf is None:
            htas_ds.close(); hpr_ds.close()
            continue

        for period in ["historical", "ssp370"]:
            tas_path = _resolve_model_file(model_root, gcm, period, "tas")
            pr_path = _resolve_model_file(model_root, gcm, period, "pr")
            if tas_path is None or pr_path is None:
                continue

            tas_rel = tas_path.parent.relative_to(model_root)
            pr_rel = pr_path.parent.relative_to(model_root)

            tas_out_dir = out_root / tas_rel
            pr_out_dir = out_root / pr_rel
            tas_out = tas_out_dir / f"{tas_path.stem}_dOTC.nc"
            pr_out = pr_out_dir / f"{pr_path.stem}_dOTC.nc"

            if RESUME and tas_out.exists() and pr_out.exists():
                continue

            tds = xr.open_dataset(tas_path, decode_times=True, use_cftime=True)
            pds = xr.open_dataset(pr_path, decode_times=True, use_cftime=True)
            tdsf, tas = _to_valid_datetime64(tds, "tas")
            pdsf, pr = _to_valid_datetime64(pds, "pr")
            if tdsf is None or pdsf is None:
                tds.close(); pds.close()
                continue

            if not (
                obs_tas.shape[1:] == obs_pr.shape[1:] == tas.shape[1:] == pr.shape[1:] == htas.shape[1:] == hpr.shape[1:]
            ):
                raise ValueError(f"Grid mismatch for {gcm} {period}")

            ntime, nN, nE = tas.shape
            tas_out_vals = np.full((ntime, nN, nE), np.nan, dtype=np.float32)
            pr_out_vals = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

            obs_tas_vals = obs_tas.values
            obs_pr_vals = obs_pr.values
            tas_vals = tas.values
            pr_vals = pr.values
            htas_vals = htas.values
            hpr_vals = hpr.values

            valid_cell_mask = (
                (~np.all(np.isnan(obs_tas_vals), axis=0))
                & (~np.all(np.isnan(obs_pr_vals), axis=0))
                & (~np.all(np.isnan(tas_vals), axis=0))
                & (~np.all(np.isnan(pr_vals), axis=0))
            )

            hist_times = htas["time"].values
            target_times = tas["time"].values
            obs_times = obs_tas["time"].values

            def process_row(i):
                row_t = np.full((ntime, nE), np.nan, dtype=np.float32)
                row_p = np.full((ntime, nE), np.nan, dtype=np.float32)
                for j in range(nE):
                    if not valid_cell_mask[i, j]:
                        continue
                    ct, cp = dotc_cell(
                        htas_vals[:, i, j], tas_vals[:, i, j], obs_tas_vals[:, i, j],
                        hpr_vals[:, i, j], pr_vals[:, i, j], obs_pr_vals[:, i, j],
                        calib_start, calib_end, hist_times, target_times, obs_times
                    )
                    row_t[:, j] = ct
                    row_p[:, j] = cp
                return i, row_t, row_p

            row_iter = Parallel(n_jobs=n_jobs, 
                                backend="threading", 
                                batch_size="auto", 
                                verbose=10,
                                return_as="generator")(
                delayed(process_row)(i) for i in range(nN)
            )
            for i, rt, rp in row_iter:
                tas_out_vals[:, i, :] = rt
                pr_out_vals[:, i, :] = rp

            tas_out_vals = np.where(valid_cell_mask[None, :, :], tas_out_vals, np.nan).astype(np.float32)
            pr_out_vals = np.where(valid_cell_mask[None, :, :], pr_out_vals, np.nan).astype(np.float32)
            
            
            os.makedirs(tas_out_dir, exist_ok=True)
            os.makedirs(pr_out_dir, exist_ok=True)

            t_out_ds = tdsf.copy()
            p_out_ds = pdsf.copy()
            t_out_ds["tas"] = (tas.dims, tas_out_vals)
            p_out_ds["pr"] = (pr.dims, pr_out_vals)



            t_out_ds.to_netcdf(tas_out)
            p_out_ds.to_netcdf(pr_out)


            t_out_ds.close()
            p_out_ds.close()
            
            print(f"Saved: {tas_out}")
            print(f"Saved: {pr_out}")

            tds.close(); pds.close()

        htas_dsf.close(); hpr_dsf.close()
        htas_ds.close(); hpr_ds.close()

    obs_ds_tas.close()
    obs_ds_pr.close()


if __name__ == "__main__":
    main()