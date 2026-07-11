import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

import config
import numpy as np
import rasterio
import torch
import xarray as xr
from tqdm import tqdm

sys.path.append(config.DDIM_PROJ_PATH)
sys.path.append(config.LDM_PROJ_PATH)
sys.path.append(config.DM_DIR)

from Downscaling_Dataset_Prep import DownscalingDataset
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.denoiser.unet import UNetModel
from models.diff_module import DDIMResidualContextual
from models.unet_module import DownscalingUnetLightning


NUM_SAMPLES = 10
ETA = 0.0
DDIM_STEPS = 30
SEED = 42


def get_period(path):
    if "/historical/" in path:
        return "historical"
    if "/ssp370/" in path:
        return "ssp370"
    return "unknown"


def get_id(path, var_name):
    fname = os.path.basename(path)
    while fname.endswith(".nc"):
        fname = fname[:-3]
    if fname.startswith(f"{var_name}_day_"):
        return fname.replace(f"{var_name}_day_", "", 1)
    return fname.replace(f"_{var_name}_", "_", 1)


def add_year_suffix(path, y0, y1):
    root, ext = os.path.splitext(path)
    return f"{root}_{y0}-{y1}{ext}"


def year_windows(start_year, end_year):
    windows = []
    if start_year <= 2014:
        windows.append(("historical", start_year, min(end_year, 2014)))
    if end_year >= 2015:
        windows.append(("ssp370", max(start_year, 2015), end_year))
    return windows


def load_grid():
    ref_grid = f"{config.DATASETS_TRAINING_DIR}/Swiss_Mask_HR.nc"
    with xr.open_dataset(ref_grid) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        mask = np.isnan(ds["TabsD"].values)
    return ref_grid, lat, lon, mask


def latlon_2d(lat, lon):
    if lat.ndim == 2 and lon.ndim == 2:
        return lat, lon
    if lat.ndim == 1 and lon.ndim == 1:
        return np.meshgrid(lat, lon, indexing="ij")
    return None, None


def run_cdo_remap(ref_grid, src, dst, y0, y1):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(
        os.path.dirname(dst),
        f".{os.path.basename(dst)}.{os.getpid()}.{uuid.uuid4().hex}.tmp.nc",
    )

    cmd = [
        "cdo",
        "-O",
        f"remapbil,{ref_grid}",
        f"-selyear,{y0}/{y1}",
        src,
        tmp,
    ]

    subprocess.run(cmd, check=True)
    os.replace(tmp, dst)


def apply_mask(path, var_name, mask):
    tmp = f"{path}.{os.getpid()}.tmp.nc"

    ds = xr.open_dataset(path).load()
    arr = ds[var_name].values

    m = mask
    if m.shape != arr.shape[-2:] and m.T.shape == arr.shape[-2:]:
        m = m.T

    arr[..., m] = np.nan
    ds[var_name].values[:] = arr

    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, path)


def load_scaling():
    with open(f"{config.DATASETS_TRAINING_DIR}/RhiresD_bilinear_scaling_params.json") as f:
        pr_params = json.load(f)
    with open(f"{config.DATASETS_TRAINING_DIR}/TabsD_bilinear_scaling_params.json") as f:
        tas_params = json.load(f)
    return pr_params, tas_params


def norm_pr(x, params):
    return (np.log(np.clip(x, 0, None) + params["epsilon"]) - params["mean"]) / params["std"]


def norm_tas(x, params):
    return (x - params["mean"]) / params["std"]


def denorm_pr(x, params):
    return np.exp(x * params["std"] + params["mean"]) - params["epsilon"]


def denorm_tas(x, params):
    return x * params["std"] + params["mean"]


def load_elevation():
    elevation_path = f"{config.BASE_DIR}/sasthana/Downscaling/GCM_pipeline/elevation.tif"
    with rasterio.open(elevation_path) as src:
        return src.read(1).astype(np.float32)


def load_unet(device):
    model = DownscalingUnetLightning(
        in_ch=3,
        out_ch=2,
        features=[64, 128, 256, 512],
        channel_names=["precip", "temp"],
        precip_scaling_json=f"{config.DATASETS_TRAINING_DIR}/RhiresD_bilinear_scaling_params.json",
    )

    ckpt = torch.load(
        f"{config.LDM_PROJ_PATH}/LDM_conditional/trained_ckpts/12km/BILINEAR_LDM_conditional.models.unet_module.DownscalingUnetLightning_bs32_lr0.001_delta1.0_factor0.5_pat3.ckpt",
        map_location="cpu",
        weights_only=False,
    )["state_dict"]

    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


def load_ddim(device):
    denoiser = UNetModel(
        model_channels=32,
        in_channels=2,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=[1, 2, 4],
        context_ch=[32, 64, 128],
        channel_mult=[1, 2, 4],
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=2,
    )

    conditioner = AFNOConditionerNetCascade(
        autoencoder=None,
        input_channels=[2],
        embed_dim=[32, 64, 128],
        analysis_depth=3,
        cascade_depth=3,
        context_ch=[32, 64, 128],
    )

    ddim = DDIMResidualContextual(
        denoiser=denoiser,
        context_encoder=conditioner,
        timesteps=1000,
        parameterization="v",
        loss_type="l1",
        beta_schedule="cosine",
        cosine_s=8e-3,
        use_ema=True,
        ema_decay=0.9999,
        lr=1e-4,
    )

    ckpt = torch.load(
        f"{config.DDIM_PROJ_PATH}/trained_ckpts/12km/BILINEAR_DDIM_L1_cosine_loss_parameterisation_v.ckpt",
        map_location=device,
    )

    ddim.load_state_dict(ckpt["state_dict"], strict=False)
    ddim.to(device)
    ddim.eval()
    return DDIMSampler(ddim, device=device)


def collect_bc_pairs(bc_root, model, start_year, end_year):
    pairs = []

    for period, y0, y1 in year_windows(start_year, end_year):
        tas_files = glob.glob(
            os.path.join(bc_root, model, period, "*", "*", "*", "day", "tas", "*", "*.nc")
        )
        pr_files = glob.glob(
            os.path.join(bc_root, model, period, "*", "*", "*", "day", "pr", "*", "*.nc")
        )

        tas_files = [f for f in tas_files if "corrfx" not in os.path.basename(f)]
        pr_files = [f for f in pr_files if "corrfx" not in os.path.basename(f)]

        pr_dict = {get_id(f, "pr"): f for f in pr_files}

        for tas_path in tas_files:
            tas_id = get_id(tas_path, "tas")
            pr_path = pr_dict.get(tas_id)
            if pr_path:
                pairs.append((tas_path, pr_path, period, y0, y1))

    return pairs


def collect_bilinear_pairs(bilinear_dir, model, start_year, end_year):
    pairs = []

    for period, y0, y1 in year_windows(start_year, end_year):
        suffix = f"_{y0}-{y1}.nc"

        tas_files = glob.glob(
            os.path.join(bilinear_dir, model, period, "**", "day", "tas", "*", f"*{suffix}"),
            recursive=True,
        )
        pr_files = glob.glob(
            os.path.join(bilinear_dir, model, period, "**", "day", "pr", "*", f"*{suffix}"),
            recursive=True,
        )

        pr_dict = {get_id(f, "pr"): f for f in pr_files}

        for tas_path in tas_files:
            tas_id = get_id(tas_path, "tas")
            pr_path = pr_dict.get(tas_id)
            if pr_path:
                pairs.append((tas_path, pr_path, period, y0, y1))

    return pairs


def run_bilinear(args, models, bc_root, bilinear_dir, ref_grid, mask):
    for model in models:
        for tas_path, pr_path, period, y0, y1 in collect_bc_pairs(
            bc_root, model, args.start_year, args.end_year
        ):
            t0 = time.perf_counter()

            tas_out = add_year_suffix(
                os.path.join(bilinear_dir, os.path.relpath(tas_path, bc_root)),
                y0,
                y1,
            )
            pr_out = add_year_suffix(
                os.path.join(bilinear_dir, os.path.relpath(pr_path, bc_root)),
                y0,
                y1,
            )

            if not os.path.exists(tas_out):
                print(f"[bilinear] tas {y0}-{y1}: {tas_out}")
                run_cdo_remap(ref_grid, tas_path, tas_out, y0, y1)
                apply_mask(tas_out, "tas", mask)

            if not os.path.exists(pr_out):
                print(f"[bilinear] pr {y0}-{y1}: {pr_out}")
                run_cdo_remap(ref_grid, pr_path, pr_out, y0, y1)
                apply_mask(pr_out, "pr", mask)

            print(f"[bilinear] done in {time.perf_counter() - t0:.1f}s")


def run_unet(args, models, bilinear_dir, unet_dir, lat, lon, mask, device):
    pr_params, tas_params = load_scaling()
    elevation = load_elevation()
    model_unet = load_unet(device)

    config_dict = {
        "variables": {
            "input": {"precip": "pr", "temp": "tas"},
            "target": {"precip": "pr", "temp": "tas"},
        },
        "preprocessing": {"nan_to_num": True, "nan_value": 0.0},
    }

    lat2d, lon2d = latlon_2d(lat, lon)
    batch_size = 16

    for model in models:
        for tas_path, pr_path, period, y0, y1 in collect_bilinear_pairs(
            bilinear_dir, model, args.start_year, args.end_year
        ):
            t0 = time.perf_counter()
            run_id = get_id(tas_path, "tas")

            rel_dir = os.path.dirname(os.path.relpath(tas_path, bilinear_dir))
            out_path = os.path.join(
                unet_dir,
                rel_dir,
                f"UNet_{period}_{y0}-{y1}_tas_{run_id}.nc",
            )

            if os.path.exists(out_path):
                print(f"[unet] exists: {out_path}")
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            input_ds = {
                "precip": xr.open_dataset(pr_path).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")),
                "temp": xr.open_dataset(tas_path).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")),
            }
            target_ds = {
                "precip": input_ds["precip"],
                "temp": input_ds["temp"],
            }

            ds = DownscalingDataset(input_ds, target_ds, config_dict, elevation_path=elevation)
            n = len(ds)
            if n == 0:
                continue

            sample, _ = ds[0]
            spatial_shape = sample.shape[1:]
            times = input_ds["temp"]["time"].values
            preds = np.empty((n, 2, *spatial_shape), dtype=np.float32)

            for start in tqdm(range(0, n, batch_size), desc=f"UNet {run_id}"):
                end = min(start + batch_size, n)
                batch = []

                for idx in range(start, end):
                    x, _ = ds[idx]
                    x = x.numpy()
                    x[0] = norm_pr(x[0], pr_params)
                    x[1] = norm_tas(x[1], tas_params)
                    batch.append(torch.from_numpy(x))

                batch = torch.stack(batch).to(device)

                with torch.no_grad():
                    y = model_unet(batch).cpu().numpy()

                y_denorm = np.empty_like(y)
                y_denorm[:, 0] = denorm_pr(y[:, 0], pr_params)
                y_denorm[:, 1] = denorm_tas(y[:, 1], tas_params)
                preds[start:end] = y_denorm

            preds = np.transpose(preds, (0, 2, 3, 1))
            preds[:, :, :, 0] = np.where(mask[None, :, :], np.nan, preds[:, :, :, 0])
            preds[:, :, :, 1] = np.where(mask[None, :, :], np.nan, preds[:, :, :, 1])

            coords = {"time": times}
            if lat2d is not None:
                coords["lat"] = (("N", "E"), lat2d)
                coords["lon"] = (("N", "E"), lon2d)

            out_ds = xr.Dataset(
                {
                    "precip": (("time", "N", "E"), preds[:, :, :, 0]),
                    "temp": (("time", "N", "E"), preds[:, :, :, 1]),
                },
                coords=coords,
            )

            out_ds.to_netcdf(out_path)
            out_ds.close()

            for d in input_ds.values():
                d.close()

            print(f"[unet] saved {out_path} in {time.perf_counter() - t0:.1f}s")



def run_ddim(args, models, unet_dir, ddim_dir, lat, lon, mask, device):
    pr_params, tas_params = load_scaling()
    sampler = load_ddim(device)

    config_dict = {
        "variables": {
            "input": {"precip": "precip", "temp": "temp"},
            "target": {"precip": "precip", "temp": "temp"},
        },
        "preprocessing": {"nan_to_num": True, "nan_value": 0.0},
    }

    lat2d, lon2d = latlon_2d(lat, lon)
    batch_size = 16

    for model in models:
        for period, y0, y1 in year_windows(args.start_year, args.end_year):
            unet_files = glob.glob(
                os.path.join(unet_dir, model, period, "**", f"UNet_{period}_{y0}-{y1}_tas_*.nc"),
                recursive=True,
            )

            for unet_file in unet_files:
                t0 = time.perf_counter()

                rel_dir = os.path.dirname(os.path.relpath(unet_file, unet_dir))
                out_path = os.path.join(
                    ddim_dir,
                    rel_dir,
                    f"DDIM_{NUM_SAMPLES}samples_{period}_{y0}-{y1}_{os.path.basename(unet_file)}",
                )

                if os.path.exists(out_path):
                    print(f"[ddim] exists: {out_path}")
                    continue

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                input_ds = xr.open_dataset(unet_file).sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
                target_ds = input_ds

                ds = DownscalingDataset(
                    {"precip": input_ds, "temp": input_ds},
                    {"precip": target_ds, "temp": target_ds},
                    config_dict,
                )


                n = len(ds)
                if n == 0:
                    continue

                sample, _ = ds[0]
                spatial_shape = sample.shape[1:]
                times = input_ds["time"].values
                batches = []

                for start in tqdm(range(0, n, batch_size), desc=f"DDIM {os.path.basename(unet_file)}"):
                    end = min(start + batch_size, n)
                    batch = []

                    for idx in range(start, end):
                        x, _ = ds[idx]
                        x = x.numpy()
                        x[0] = norm_pr(x[0], pr_params)
                        x[1] = norm_tas(x[1], tas_params)
                        batch.append(torch.from_numpy(x))

                    batch = torch.stack(batch).to(device)

                    with torch.no_grad():
                        unet_pred = batch
                        context = [(unet_pred, None)]
                        sample_shape = unet_pred.shape
                        batch_preds = np.empty(
                            (end - start, NUM_SAMPLES, *sample_shape[1:]),
                            dtype=np.float32,
                        )

                        for s in range(NUM_SAMPLES):
                            torch.manual_seed(SEED + s)
                            np.random.seed(SEED + s)

                            z = torch.randn((end - start, *sample_shape[1:]), device=device)
                            residual, _ = sampler.sample(
                                S=DDIM_STEPS,
                                batch_size=end - start,
                                shape=sample_shape[1:],
                                conditioning=context,
                                eta=ETA,
                                verbose=False,
                                x_T=z,
                                schedule="cosine",
                            )

                            y = (unet_pred + residual).cpu().numpy()
                            y_denorm = np.empty_like(y)
                            y_denorm[:, 0] = denorm_pr(y[:, 0], pr_params)
                            y_denorm[:, 1] = denorm_tas(y[:, 1], tas_params)
                            batch_preds[:, s] = y_denorm

                    batch_preds = np.transpose(batch_preds, (0, 1, 3, 4, 2))
                    batch_preds[:, :, :, :, 0] = np.where(
                        mask[None, None, :, :], np.nan, batch_preds[:, :, :, :, 0]
                    )
                    batch_preds[:, :, :, :, 1] = np.where(
                        mask[None, None, :, :], np.nan, batch_preds[:, :, :, :, 1]
                    )

                    coords = {
                        "time": times[start:end],
                        "sample": np.arange(NUM_SAMPLES),
                    }
                    if lat2d is not None:
                        coords["lat"] = (("N", "E"), lat2d)
                        coords["lon"] = (("N", "E"), lon2d)

                    batches.append(
                        xr.Dataset(
                            {
                                "precip": (("time", "sample", "N", "E"), batch_preds[:, :, :, :, 0]),
                                "temp": (("time", "sample", "N", "E"), batch_preds[:, :, :, :, 1]),
                            },
                            coords=coords,
                        )
                    )

                out_ds = xr.concat(batches, dim="time")
                out_ds.to_netcdf(out_path)
                out_ds.close()
                input_ds.close()

                print(f"[ddim] saved {out_path} in {time.perf_counter() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--mode", choices=["bilinear", "unet", "ddim"], required=True)
    parser.add_argument("--ensemble", choices=["EQM_C", "dOTC", "CDF-t"], required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_grid, lat, lon, mask = load_grid()

    bc_root = f"{config.BIAS_CORRECTED_DIR_SSP370}/{args.ensemble}"
    out_root = os.path.join(os.path.dirname(config.BIAS_CORRECTED_DIR_SSP370), "BC+SR")

    bilinear_dir = os.path.join(out_root, "Bilinear", args.ensemble)
    unet_dir = os.path.join(out_root, "Bilinear_UNet", args.ensemble)
    ddim_dir = os.path.join(out_root, "Bilinear_UNet_DDIM", args.ensemble)

    models = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(bc_root, "*"))
        if os.path.isdir(p)
    )

    if args.mode == "bilinear":
        run_bilinear(args, models, bc_root, bilinear_dir, ref_grid, mask)
    elif args.mode == "unet":
        run_unet(args, models, bilinear_dir, unet_dir, lat, lon, mask, device)
    elif args.mode == "ddim":
        run_ddim(args, models, unet_dir, ddim_dir, lat, lon, mask, device)

    print(f"[done] {args.mode} {args.ensemble} {args.start_year}-{args.end_year} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()