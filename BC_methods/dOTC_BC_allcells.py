import os
from pathlib import Path
import warnings
import argparse
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from SBCK import dOTC

warnings.filterwarnings("ignore", category=DeprecationWarning)


def seconds_to_minutes(seconds):
    return seconds / 60.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def get_doy(d):
    return (
        np.datetime64(d, "D")
        - np.datetime64(str(d)[:4] + "-01-01", "D")
    ).astype(int) + 1


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
    out["time"] = np.array(tvals, dtype="datetime64[ns]")
    return out, out[var_name]


def _resolve_model_file(base_dir, gcm, period, var):
    p = (
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

    if not p.exists():
        return None

    exact = p / f"{gcm}_{period}_{var}_Swiss.nc"

    if exact.exists():
        return exact

    cands = sorted(p.glob(f"{gcm}_{period}_{var}*.nc"))
    return cands[0] if cands else None


def dotc_cell(
    calib_tas,
    target_tas,
    obs_tas,
    calib_pr,
    target_pr,
    obs_pr,
    calib_start,
    calib_end,
    calib_times,
    target_times,
    obs_times,
):
    ntime = target_tas.shape[0]
    out = np.full((ntime, 2), np.nan, dtype=np.float32)

    correction_obtaining_seconds = 0.0
    correction_applying_seconds = 0.0

    full_mod = np.stack([target_tas, target_pr], axis=1)

    if np.isnan(full_mod).sum() / full_mod.size > 0.7:
        return (
            out[:, 0],
            out[:, 1],
            correction_obtaining_seconds,
            correction_applying_seconds,
        )

    calib_times_np = np.array(calib_times, dtype="datetime64[D]")
    target_times_np = np.array(target_times, dtype="datetime64[D]")
    obs_times_np = np.array(obs_times, dtype="datetime64[D]")

    m_cal = calib_times_np[
        (calib_times_np >= calib_start) & (calib_times_np <= calib_end)
    ]
    o_cal = obs_times_np[
        (obs_times_np >= calib_start) & (obs_times_np <= calib_end)
    ]

    calib_dates = np.intersect1d(m_cal, o_cal)

    if calib_dates.size < 30:
        return (
            out[:, 0],
            out[:, 1],
            correction_obtaining_seconds,
            correction_applying_seconds,
        )

    m_mask = np.isin(calib_times_np, calib_dates)
    o_mask = np.isin(obs_times_np, calib_dates)

    calib_mod = np.stack([calib_tas[m_mask], calib_pr[m_mask]], axis=1)
    calib_obs = np.stack([obs_tas[o_mask], obs_pr[o_mask]], axis=1)

    calib_doys = np.array([get_doy(d) for d in calib_dates])
    full_doys = np.array([get_doy(d) for d in target_times_np])

    for doy in range(1, 367):
        obtaining_start = time.perf_counter()

        wd = (calib_doys - doy + 366) % 366
        wmask = (wd <= 45) | (wd >= (366 - 45))

        cm = calib_mod[wmask]
        co = calib_obs[wmask]

        fmask = full_doys == doy
        fm = full_mod[fmask]

        if cm.shape[0] == 0 or co.shape[0] == 0 or fm.shape[0] == 0:
            correction_obtaining_seconds += time.perf_counter() - obtaining_start
            continue

        valid_cal = ~(np.isnan(cm).any(axis=1) | np.isnan(co).any(axis=1))
        valid_pred = ~np.isnan(fm).any(axis=1)

        cm = cm[valid_cal]
        co = co[valid_cal]
        fm_clean = fm[valid_pred]

        if cm.shape[0] < 10 or fm_clean.shape[0] == 0:
            correction_obtaining_seconds += time.perf_counter() - obtaining_start
            continue

        if (
            np.unique(cm[:, 0]).size < 2
            or np.unique(cm[:, 1]).size < 2
            or np.unique(co[:, 0]).size < 2
            or np.unique(co[:, 1]).size < 2
            or np.unique(fm_clean[:, 0]).size < 2
            or np.unique(fm_clean[:, 1]).size < 2
        ):
            correction_obtaining_seconds += time.perf_counter() - obtaining_start
            continue

        try:
            model = dOTC(bin_width=None, bin_origin=None)
            model.fit(co, cm, fm_clean)
        except Exception:
            correction_obtaining_seconds += time.perf_counter() - obtaining_start
            continue

        correction_obtaining_seconds += time.perf_counter() - obtaining_start

        applying_start = time.perf_counter()

        try:
            corrected = model.predict(fm_clean)

            if (
                corrected is None
                or corrected.shape != fm_clean.shape
                or np.all(np.isnan(corrected))
            ):
                raise RuntimeError("invalid dOTC output")

            corrected = corrected.astype(np.float32, copy=False)

            out_idx = np.flatnonzero(fmask)
            out[out_idx[valid_pred], :] = corrected

        except Exception:
            correction_applying_seconds += time.perf_counter() - applying_start
            continue

        correction_applying_seconds += time.perf_counter() - applying_start

    return (
        out[:, 0],
        out[:, 1],
        correction_obtaining_seconds,
        correction_applying_seconds,
    )


def main():
    script_start = time.perf_counter()

    print("dOTC coarse all cells started")

    args = parse_args()

    RESUME = True
    calib_start = np.datetime64("1981-01-01")
    calib_end = np.datetime64("2010-12-31")
    n_jobs = args.n_jobs

    root = Path(__file__).resolve().parents[3]

    obs_tas_path = (
        root
        / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/"
        / "TabsD_step2_coarse.nc"
    )
    obs_pr_path = (
        root
        / "Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/"
        / "RhiresD_step2_coarse.nc"
    )

    model_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/Swiss"
    out_root = root / "Downscaling/GCM_pipeline/ALP-FINEv1.0/BC/dOTC"

    gcms = ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]
    periods = ["historical", "ssp370"]

    jobs_by_gcm = {}

    for gcm in gcms:
        for period in periods:
            tas_path = _resolve_model_file(model_root, gcm, period, "tas")
            pr_path = _resolve_model_file(model_root, gcm, period, "pr")

            if tas_path is None or pr_path is None:
                print(f"Missing input, skipping: {gcm} {period}")
                continue

            tas_rel = tas_path.parent.relative_to(model_root)
            pr_rel = pr_path.parent.relative_to(model_root)

            tas_out_dir = out_root / tas_rel
            pr_out_dir = out_root / pr_rel

            tas_out = tas_out_dir / f"{tas_path.stem}_dOTC.nc"
            pr_out = pr_out_dir / f"{pr_path.stem}_dOTC.nc"

            write_tas = not tas_out.exists()
            write_pr = not pr_out.exists()

            if RESUME and not write_tas and not write_pr:
                print(f"Already exists, skipping before processing: {gcm} {period}")
                continue

            jobs_by_gcm.setdefault(gcm, []).append(
                {
                    "period": period,
                    "tas_path": tas_path,
                    "pr_path": pr_path,
                    "tas_out_dir": tas_out_dir,
                    "pr_out_dir": pr_out_dir,
                    "tas_out": tas_out,
                    "pr_out": pr_out,
                    "write_tas": write_tas,
                    "write_pr": write_pr,
                }
            )

    if not jobs_by_gcm:
        print("All requested outputs already exist. Nothing to process.")
        print(
            f"Total script runtime: "
            f"{seconds_to_minutes(time.perf_counter() - script_start):.2f} min"
        )
        return

    obs_ds_tas = xr.open_dataset(obs_tas_path)
    obs_ds_pr = xr.open_dataset(obs_pr_path)

    obs_tas = obs_ds_tas["TabsD"].load()
    obs_pr = obs_ds_pr["RhiresD"].load()

    obs_tas_vals = obs_tas.values
    obs_pr_vals = obs_pr.values
    obs_times = obs_tas["time"].values
    obs_grid_shape = obs_tas.shape[1:]

    for gcm in gcms:
        period_jobs = jobs_by_gcm.get(gcm)

        if not period_jobs:
            continue

        hist_tas_path = _resolve_model_file(model_root, gcm, "historical", "tas")
        hist_pr_path = _resolve_model_file(model_root, gcm, "historical", "pr")

        if hist_tas_path is None or hist_pr_path is None:
            print(f"Missing historical input, skipping GCM: {gcm}")
            continue

        htas_ds = xr.open_dataset(hist_tas_path, decode_times=True, use_cftime=True)
        hpr_ds = xr.open_dataset(hist_pr_path, decode_times=True, use_cftime=True)

        htas_dsf, htas = _to_valid_datetime64(htas_ds, "tas")
        hpr_dsf, hpr = _to_valid_datetime64(hpr_ds, "pr")

        if htas_dsf is None or hpr_dsf is None:
            htas_ds.close()
            hpr_ds.close()
            continue

        htas = htas.load()
        hpr = hpr.load()

        htas_vals = htas.values
        hpr_vals = hpr.values
        hist_times = htas["time"].values
        hist_grid_shape = htas.shape[1:]

        for job in period_jobs:
            period = job["period"]
            tas_path = job["tas_path"]
            pr_path = job["pr_path"]
            tas_out_dir = job["tas_out_dir"]
            pr_out_dir = job["pr_out_dir"]
            tas_out = job["tas_out"]
            pr_out = job["pr_out"]

            write_tas = not tas_out.exists()
            write_pr = not pr_out.exists()

            if RESUME and not write_tas and not write_pr:
                print(f"Already exists, skipping: {gcm} {period}")
                continue

            print(f"Processing: {gcm} {period}")

            run_start = time.perf_counter()
            correction_obtaining_total_seconds = 0.0
            correction_applying_total_seconds = 0.0

            tds = xr.open_dataset(tas_path, decode_times=True, use_cftime=True)
            pds = xr.open_dataset(pr_path, decode_times=True, use_cftime=True)

            tdsf, tas = _to_valid_datetime64(tds, "tas")
            pdsf, pr = _to_valid_datetime64(pds, "pr")

            if tdsf is None or pdsf is None:
                tds.close()
                pds.close()
                continue

            tas = tas.load()
            pr = pr.load()

            tas_vals = tas.values
            pr_vals = pr.values
            target_times = tas["time"].values

            if not (
                obs_grid_shape
                == obs_pr.shape[1:]
                == tas.shape[1:]
                == pr.shape[1:]
                == hist_grid_shape
                == hpr.shape[1:]
            ):
                raise ValueError(f"Grid mismatch for {gcm} {period}")

            ntime, nN, nE = tas.shape

            tas_out_vals = np.full((ntime, nN, nE), np.nan, dtype=np.float32)
            pr_out_vals = np.full((ntime, nN, nE), np.nan, dtype=np.float32)

            valid_cell_mask = (
                (~np.all(np.isnan(obs_tas_vals), axis=0))
                & (~np.all(np.isnan(obs_pr_vals), axis=0))
                & (~np.all(np.isnan(htas_vals), axis=0))
                & (~np.all(np.isnan(hpr_vals), axis=0))
                & (~np.all(np.isnan(tas_vals), axis=0))
                & (~np.all(np.isnan(pr_vals), axis=0))
            )

            valid_cols_by_row = [
                np.flatnonzero(valid_cell_mask[i]) for i in range(nN)
            ]

            def process_row(i):
                row_t = np.full((ntime, nE), np.nan, dtype=np.float32)
                row_p = np.full((ntime, nE), np.nan, dtype=np.float32)

                row_correction_obtaining_seconds = 0.0
                row_correction_applying_seconds = 0.0

                for j in valid_cols_by_row[i]:
                    ct, cp, obtain_seconds, apply_seconds = dotc_cell(
                        htas_vals[:, i, j],
                        tas_vals[:, i, j],
                        obs_tas_vals[:, i, j],
                        hpr_vals[:, i, j],
                        pr_vals[:, i, j],
                        obs_pr_vals[:, i, j],
                        calib_start,
                        calib_end,
                        hist_times,
                        target_times,
                        obs_times,
                    )

                    row_t[:, j] = ct
                    row_p[:, j] = cp

                    row_correction_obtaining_seconds += obtain_seconds
                    row_correction_applying_seconds += apply_seconds

                return (
                    i,
                    row_t,
                    row_p,
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
                rt,
                rp,
                row_correction_obtaining_seconds,
                row_correction_applying_seconds,
            ) in row_iter:
                tas_out_vals[:, i, :] = rt
                pr_out_vals[:, i, :] = rp

                correction_obtaining_total_seconds += (
                    row_correction_obtaining_seconds
                )
                correction_applying_total_seconds += (
                    row_correction_applying_seconds
                )

            tas_out_vals = np.where(
                valid_cell_mask[None, :, :],
                tas_out_vals,
                np.nan,
            ).astype(np.float32)

            pr_out_vals = np.where(
                valid_cell_mask[None, :, :],
                pr_out_vals,
                np.nan,
            ).astype(np.float32)

            os.makedirs(tas_out_dir, exist_ok=True)
            os.makedirs(pr_out_dir, exist_ok=True)

            if write_tas:
                t_out_ds = tdsf.copy()
                t_out_ds["tas"] = (tas.dims, tas_out_vals)
                t_out_ds.to_netcdf(tas_out)
                t_out_ds.close()
                print(f"Saved: {tas_out}")
            else:
                print(f"tas already exists, not overwritten: {tas_out}")

            if write_pr:
                p_out_ds = pdsf.copy()
                p_out_ds["pr"] = (pr.dims, pr_out_vals)
                p_out_ds.to_netcdf(pr_out)
                p_out_ds.close()
                print(f"Saved: {pr_out}")
            else:
                print(f"pr already exists, not overwritten: {pr_out}")

            run_total_seconds = time.perf_counter() - run_start

            print(
                f"Timing summary for {gcm} | tas+pr | {period}:\n"
                f"  Total runtime per run: "
                f"{seconds_to_minutes(run_total_seconds):.2f} min\n"
                f"  Total correction-function obtaining time: "
                f"{seconds_to_minutes(correction_obtaining_total_seconds):.2f} min\n"
                f"  Total correction-function applying time: "
                f"{seconds_to_minutes(correction_applying_total_seconds):.2f} min"
            )

            tdsf.close()
            pdsf.close()
            tds.close()
            pds.close()

            del tas_vals, pr_vals, tas_out_vals, pr_out_vals, tas, pr

        htas_dsf.close()
        hpr_dsf.close()
        htas_ds.close()
        hpr_ds.close()

        del htas_vals, hpr_vals, htas, hpr

    obs_ds_tas.close()
    obs_ds_pr.close()

    print(
        f"Total script runtime: "
        f"{seconds_to_minutes(time.perf_counter() - script_start):.2f} min"
    )
    print("dOTC coarse all cells finished")


if __name__ == "__main__":
    main()