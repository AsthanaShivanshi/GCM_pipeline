import sys
import config
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import xarray as xr
import argparse
import rasterio
import glob
import subprocess
import concurrent.futures

import re

sys.path.append(config.DDIM_PROJ_PATH)
sys.path.append(config.LDM_PROJ_PATH)
sys.path.append(config.DM_DIR)

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from Downscaling_Dataset_Prep import DownscalingDataset
from models.components.diff.denoiser.unet import UNetModel
from models.components.diff.denoiser.ddim import DDIMSampler
from models.components.diff.conditioner import AFNOConditionerNetCascade
from models.diff_module import DDIMResidualContextual

#Purpose : Bili. interp. and then UNet,,,+ DDIM tbs to GCM_pipeline/ALP-FINEv1.0/BC+SR
#----------------------------------------------------------------------#

def run_cdo(cmd):
    subprocess.run(cmd, check=True)

def cat_file(pattern, out_path, dim="time"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    print(f"Concatenating {len(files)} files to {out_path}")
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds.to_netcdf(out_path)
    ds.close()
    for f in files:
        os.remove(f)
    print(f"Saved {out_path} and deleted intermediates.")

#----------------------------------------------------------------------#

num_samples = 10 # Deterministic for fixed random seed
eta = 0.0       # DDIM
S = 30         # Number of DDIM steps
manual_seed=42
#----------------------------------------------------------------------#


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#----------------------------------------------------------------------#


ref_grid_pr = f"{config.DATASETS_TRAINING_DIR}/Swiss_Mask_HR.nc"
ref_grid_tas = f"{config.DATASETS_TRAINING_DIR}/Swiss_Mask_HR.nc"

ref_grid_pr_ds = xr.open_dataset(ref_grid_pr)
ref_grid_tas_ds = xr.open_dataset(ref_grid_tas)

ref_lat = ref_grid_pr_ds["lat"].values
ref_lon = ref_grid_pr_ds["lon"].values

mask_pr = np.isnan(ref_grid_pr_ds["TabsD"].values)
mask_tas = np.isnan(ref_grid_tas_ds["TabsD"].values)

ref_grid_pr_ds.close()
ref_grid_tas_ds.close()


#Unique identifier for model runs. Same scenario same id runs fed together. 


def get_id(path, var_prefix):
    fname = os.path.basename(path)

    if fname.startswith(f"{var_prefix}_day_"):
        return fname.replace(f"{var_prefix}_day_", "")

    return fname.replace(f"_{var_prefix}_", "_")

def apply_mask_to_file(nc_path, var_name, mask):
    ds = xr.open_dataset(nc_path).load()

    data_var = var_name if var_name in ds.data_vars else list(ds.data_vars)[0]
    mask_da = xr.DataArray(mask, dims=ds[data_var].dims[-2:])

    ds[data_var] = ds[data_var].where(~mask_da)

    tmp_path = nc_path + ".tmp"
    ds.to_netcdf(tmp_path)
    ds.close()

    os.replace(tmp_path, nc_path)



config_dict = {
    'variables': {
        'input': {'precip': 'pr', 'temp': 'tas'},
        'target': {'precip': 'pr', 'temp': 'tas'}
    },
    'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
}


#.---#

#For ddim case only 


def find_unet_file(unet_dir, pr_path, target_year):
    model_id = get_id(pr_path, "pr")
    pattern = re.compile(
        r"UNet_ssp370_(\d{4})-(\d{4})_tas_" + re.escape(model_id) + r"(\.nc)?$"
    )

    for path in glob.glob(os.path.join(unet_dir, "**", "*"), recursive=True):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start <= target_year <= end:
                return path

    return None

#----------------------------------------------------------------------#

with open(f"{config.DATASETS_TRAINING_DIR}/RhiresD_bilinear_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open(f"{config.DATASETS_TRAINING_DIR}/TabsD_bilinear_scaling_params.json", 'r') as f:
    temp_params = json.load(f)

def norm_pr(x, pr_params):
    x_safe = np.clip(x, 0, None) #For model non negativity of precip
    
    return (np.log(x_safe + pr_params['epsilon']) - pr_params['mean']) / pr_params['std']

def norm_temp(x, params):
    return (x - params['mean']) / params['std']

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']

#----------------------------------------------------------------------#

elevation_path = f"{config.BASE_DIR}/sasthana/Downscaling/GCM_pipeline/elevation.tif"
with rasterio.open(elevation_path) as src:
    elevation_array = src.read(1).astype(np.float32)

#----------------------------------------------------------------------#


unet_regr = DownscalingUnetLightning(
    in_ch=3,
    out_ch=2,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp"],
    precip_scaling_json=f"{config.DATASETS_TRAINING_DIR}/RhiresD_bilinear_scaling_params.json",
)
unet_regr_ckpt = torch.load(
    f"{config.LDM_PROJ_PATH}/LDM_conditional/trained_ckpts/12km/BILINEAR_LDM_conditional.models.unet_module.DownscalingUnetLightning_bs32_lr0.001_delta1.0_factor0.5_pat3.ckpt",
    map_location="cpu",
    weights_only=False
)["state_dict"]
unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
unet_regr = unet_regr.to(device)
unet_regr.eval()




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
    num_heads=2
)





conditioner = AFNOConditionerNetCascade(
    autoencoder=None,
    input_channels=[2],
    embed_dim=[32, 64, 128],
    analysis_depth=3,
    cascade_depth=3,
    context_ch=[32, 64, 128]
)






ddim = DDIMResidualContextual(
    denoiser=denoiser,
    context_encoder=conditioner,
    timesteps=1000,                
    parameterization="v",
    loss_type="l1",
    beta_schedule="cosine",
    #linear_start=1e-4,
    #linear_end=2e-2,
    cosine_s=8e-3,
    use_ema=True,
    ema_decay=0.9999,
    lr=1e-4
)





ddim_ckpt = torch.load(
    f"{config.DDIM_PROJ_PATH}/trained_ckpts/12km/BILINEAR_DDIM_L1_cosine_loss_parameterisation_v.ckpt",
    map_location=device
)


ddim.load_state_dict(ddim_ckpt["state_dict"], strict=False)
ddim = ddim.to(device)
ddim.eval()
sampler = DDIMSampler(ddim, device=device)


#----------------------------------------------------------------------#


if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True, help="Start year")
    parser.add_argument("--end_year", type=int, required=True, help="End year")
    parser.add_argument("--mode", type=str, choices=["bilinear", "unet", "ddim"], required=True) #modes for models
    parser.add_argument("--ensemble", type=str, choices=["EQM_C", "dOTC", "CDF-t"], required=True, help="BC name (EQM, dOTC, or CDF-t at 12.5 kms)")
    args = parser.parse_args()

#----------------------------------------------------------------------#

bc_root = f"{config.BIAS_CORRECTED_DIR_SSP370}/{args.ensemble}"

models = [
    os.path.basename(p)
    for p in glob.glob(os.path.join(bc_root, "*"))
    if os.path.isdir(p)
]

periods = ["historical", "ssp370"]

for model in models:
    tas_files = []
    pr_files = []

    for period in periods:
        tas_pattern = os.path.join(
            bc_root, model, period, "*", "*", "*", "day", "tas", "*", "*.nc"
        )
        pr_pattern = os.path.join(
            bc_root, model, period, "*", "*", "*", "day", "pr", "*", "*.nc"
        )

        tas_files.extend(glob.glob(tas_pattern))
        pr_files.extend(glob.glob(pr_pattern))

    bilinear_dir = os.path.join(
    os.path.dirname(config.BIAS_CORRECTED_DIR_SSP370),
    "BC+SR",
    "Bilinear",
    args.ensemble
)
    os.makedirs(bilinear_dir, exist_ok=True)

    pr_dict = {get_id(f, "pr"): f for f in pr_files}



#----------------------------------------------------------------------#

    if args.mode == "bilinear": 
        for tas_path in tas_files: 
            tas_id= get_id(tas_path,"tas")
            pr_path= pr_dict.get(tas_id, None) #unique id for corr pr file


            #CHECK . 

            if pr_path is None:
                print("Need matching pr file to proceed")
                continue


            tas_rel_path = os.path.relpath(tas_path, bc_root)
            pr_rel_path = os.path.relpath(pr_path, bc_root)

            bilinear_tas_path = os.path.join(bilinear_dir, tas_rel_path)
            bilinear_pr_path = os.path.join(bilinear_dir, pr_rel_path)

            os.makedirs(os.path.dirname(bilinear_tas_path), exist_ok=True)
            os.makedirs(os.path.dirname(bilinear_pr_path), exist_ok=True)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                tasks=[]
                if not os.path.exists(bilinear_tas_path):
                    print(f"bilinearly interpolating {tas_path}")

                    tasks.append(executor.submit(run_cdo, ["cdo", "remapbil," + ref_grid_tas, 
                                                           tas_path, 
                                                           bilinear_tas_path]))
                    
                else:
                    print(f"{bilinear_tas_path} exists already")


                if not os.path.exists(bilinear_pr_path):
                    print(f"Bilinearly interpolating {pr_path}")

                    tasks.append(executor.submit(run_cdo, ["cdo", "remapbil," + ref_grid_pr, 
                                                           pr_path, 
                                                           bilinear_pr_path]))
                    

                else:
                    print(f"{bilinear_pr_path} exists already")
                
                for task in tasks:
                    task.result()

                apply_mask_to_file(bilinear_tas_path, "tas", mask_tas)
                apply_mask_to_file(bilinear_pr_path, "pr", mask_pr)

    
#----------------------------------------------------------------------#


    elif args.mode == "unet":
        batch_size = 16

        bilinear_tas_files = glob.glob(
    os.path.join(bilinear_dir, model, "**", "day", "tas", "*", "*.nc"),
    recursive=True
)
        bilinear_pr_files = glob.glob(
    os.path.join(bilinear_dir, model, "**", "day", "pr", "*", "*.nc"),
    recursive=True
)

        bilinear_pr_dict = {get_id(f, "pr"): f for f in bilinear_pr_files}

        for bilinear_tas_path in bilinear_tas_files:
            model_id = get_id(bilinear_tas_path, "tas")
            bilinear_pr_path = bilinear_pr_dict.get(model_id)

            if bilinear_pr_path is None:
                print(f"No matching bilinear pr file for {bilinear_tas_path}")
                continue

            print(f"Processing run: {model_id}")

            input_ds = {
                "precip": xr.open_dataset(bilinear_pr_path),
                "temp": xr.open_dataset(bilinear_tas_path)
            }
            target_ds = {
                "precip": xr.open_dataset(bilinear_pr_path),
                "temp": xr.open_dataset(bilinear_tas_path)
            }

            time_start = f"{args.start_year}-01-01"
            time_end = f"{args.end_year}-12-31"


            for var in input_ds:
                input_ds[var] = input_ds[var].sel(time=slice(time_start, time_end))
            for var in target_ds:
                target_ds[var] = target_ds[var].sel(time=slice(time_start, time_end))

            ds = DownscalingDataset(input_ds, target_ds, config_dict, elevation_path=elevation_array)
            N = len(ds)


            print(f"Model: {model_id} | Dataloader prepared with shape: {N}")

            input_tensor, _ = ds[0]
            spatial_shape = input_tensor.shape[1:]
            print("Spatial shape: ", spatial_shape)
            times = input_ds["temp"]["time"].values

            unet_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)
            var_names = ["precip", "temp"]

            lat2d, lon2d = None, None
            if ref_lat.ndim == 2 and ref_lon.ndim == 2:
                lat2d, lon2d = ref_lat, ref_lon
            elif ref_lat.ndim == 1 and ref_lon.ndim == 1:
                lat2d, lon2d = np.meshgrid(ref_lat, ref_lon, indexing="ij")
            encoding = {}

            for batch_start in tqdm(range(0, N, batch_size), desc=f"UNet Inference for {model_id}"):


                batch_end = min(batch_start + batch_size, N)
                batch_inputs = []
                for idx in range(batch_start, batch_end):
                    input_tensor, _ = ds[idx]


                    input_tensor_np = input_tensor.numpy()
                    input_tensor_np[0] = norm_pr(input_tensor_np[0], pr_params)
                    input_tensor_np[1] = norm_temp(input_tensor_np[1], temp_params)
                    batch_inputs.append(torch.from_numpy(input_tensor_np))


                batch_tensor = torch.stack(batch_inputs).to(device)
                with torch.no_grad():
                    unet_pred = unet_regr(batch_tensor)
                    unet_pred_np = unet_pred.cpu().numpy()
                    unet_pred_denorm = np.empty_like(unet_pred_np)
                    for varindex, var in enumerate(var_names):
                        if var == "precip":
                            unet_pred_denorm[:, varindex] = denorm_pr(unet_pred_np[:, varindex], pr_params)
                        elif var == "temp":
                            unet_pred_denorm[:, varindex] = denorm_temp(unet_pred_np[:, varindex], temp_params)
                        else:
                            raise ValueError(f"Unknown variable name: {var}")
                    unet_all[batch_start:batch_end] = unet_pred_denorm

            unet_preds_np = np.transpose(unet_all, (0, 2, 3, 1))  # [time, N, E, var]
            
            for i, var in enumerate(var_names):
                if var == "precip":
                    unet_preds_np[:, :, :, i] = np.where(mask_pr[None, :, :], np.nan, unet_preds_np[:, :, :, i])
                elif var == "temp":
                    unet_preds_np[:, :, :, i] = np.where(mask_tas[None, :, :], np.nan, unet_preds_np[:, :, :, i])
            
            
            ds_unet = xr.Dataset(
                {var: (("time", "N", "E"), unet_preds_np[:, :, :, i]) for i, var in enumerate(var_names)},
                coords={
                    "time": times,
                    "lat": (("N", "E"), lat2d) if lat2d is not None else None,
                    "lon": (("N", "E"), lon2d) if lon2d is not None else None,
                }
            )




            unet_out_dir = os.path.join(
            os.path.dirname(config.BIAS_CORRECTED_DIR_SSP370),
            "BC+SR",
            "Bilinear_plus_UNet",
            args.ensemble
    )
            os.makedirs(unet_out_dir, exist_ok=True)



            unet_rel_path = os.path.relpath(bilinear_tas_path, bilinear_dir)
            unet_rel_dir = os.path.dirname(unet_rel_path)

            out_path_unet = os.path.join(
            unet_out_dir,
            unet_rel_dir,
            f"UNet_ssp370_{args.start_year}-{args.end_year}_tas_{model_id}.nc"
                )

            os.makedirs(os.path.dirname(out_path_unet), exist_ok=True)
            ds_unet.to_netcdf(out_path_unet, encoding=encoding)
            ds_unet.close()
            print(f"Saved UNet output: {out_path_unet}")


            for ds_ in input_ds.values():
                ds_.close()
            for ds_ in target_ds.values():
                ds_.close()


            print(f"Processing run: {model_id}")

#----------------------------------------------------------------------#



    elif args.mode == "ddim":

        ddim_config_dict = {
            'variables': {
                'input': {'precip': 'precip', 'temp': 'temp'},
                'target': {'precip': 'precip', 'temp': 'temp'}
            },
            'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
        }

        for tas_path in tas_files:
            batch_size = 32
            tas_id = get_id(tas_path, "tas")
            pr_path = pr_dict.get(tas_id, None)
            if pr_path is None:
                print(f"No matching pr file for {tas_path}")
                continue

            print(f"Processing {tas_path} and {pr_path}")

            unet_ensemble = args.ensemble


            unet_dir = os.path.join(
                os.path.dirname(config.BIAS_CORRECTED_DIR_SSP370),
                "BC+SR",
                "Bilinear_plus_UNet",
                unet_ensemble
            )
            
            unet_file = find_unet_file(unet_dir, pr_path, args.start_year)

            if not unet_file or not os.path.exists(unet_file):
                print(f"UNet file for year {args.start_year} and {pr_path} does not exist, skipping.")
                continue

            out_path_ddim = f"ALP-FINE_4.5/{args.ensemble}/DDIM/DDIM_{num_samples}samples_RCP45_{args.start_year}-{args.end_year}_tas_{get_id(pr_path, 'pr')}"
            
            
            
            if os.path.exists(out_path_ddim):
                print(f"DDIM file already exists for {pr_path}, skipping sampling.")
                continue

            input_ds = xr.open_dataset(unet_file)
            target_ds = xr.open_dataset(unet_file)

            time_start = f"{args.start_year}-01-01"
            time_end = f"{args.end_year}-12-31"

            input_ds = input_ds.sel(time=slice(time_start, time_end))
            target_ds = target_ds.sel(time=slice(time_start, time_end))

            ds = DownscalingDataset(
                {"precip": input_ds, "temp": input_ds},
                {"precip": target_ds, "temp": target_ds},
                ddim_config_dict,
                elevation_path=elevation_array
            )

            input_tensor, _ = ds[0]
            spatial_shape = input_tensor.shape[1:]
            times = input_ds['time'].values
            N = len(ds)
            var_names = ["precip", "temp"]

            lat2d, lon2d = None, None
            if ref_lat.ndim == 2 and ref_lon.ndim == 2:
                lat2d, lon2d = ref_lat, ref_lon
            elif ref_lat.ndim == 1 and ref_lon.ndim == 1:
                lat2d, lon2d = np.meshgrid(ref_lat, ref_lon, indexing="ij")
            encoding = {}


            all_ds_ddim = []

            for batch_start in tqdm(range(0, N, batch_size), desc="DDIM Sampling"):
                batch_end = min(batch_start + batch_size, N)
                batch_inputs = []
                for idx in range(batch_start, batch_end):
                    input_tensor, _ = ds[idx]
                    input_tensor_np = input_tensor.numpy()
                    input_tensor_np[0] = norm_pr(input_tensor_np[0], pr_params)
                    input_tensor_np[1] = norm_temp(input_tensor_np[1], temp_params)
                    batch_inputs.append(torch.from_numpy(input_tensor_np))
                batch_tensor = torch.stack(batch_inputs).to(device)

                with torch.no_grad():
                    unet_pred = unet_regr(batch_tensor)
                    context = [(unet_pred, None)]
                    sample_shape = unet_pred.shape  # [batch, channels, N, E]

                    ddim_pred_denorm_all = np.empty((batch_end - batch_start, num_samples, *sample_shape[1:]), dtype=np.float32)

                    for j in range(num_samples):
                        torch.manual_seed(manual_seed + j)
                        np.random.seed(manual_seed + j)
                        z = torch.randn((batch_end - batch_start, *sample_shape[1:]), device=device)
                        residual, _ = sampler.sample(
                            S=S,
                            batch_size=(batch_end - batch_start),
                            shape=sample_shape[1:],
                            conditioning=context,
                            eta=eta,
                            verbose=False,
                            x_T=z,
                            schedule="cosine"
                        )


                        final_pred = unet_pred + residual
                        final_pred_np = final_pred.cpu().numpy()
                        ddim_pred_denorm = np.empty_like(final_pred_np)


                        for varindex, var in enumerate(var_names):
                            if var == "precip":
                                ddim_pred_denorm[:, varindex] = denorm_pr(final_pred_np[:, varindex], pr_params)
                            elif var == "temp":
                                ddim_pred_denorm[:, varindex] = denorm_temp(final_pred_np[:, varindex], temp_params)
                            else:
                                raise ValueError(f"Unknown variable name: {var}")
                        ddim_pred_denorm_all[:, j] = ddim_pred_denorm


                ddim_preds_np = np.transpose(ddim_pred_denorm_all, (0, 1, 3, 4, 2)) 
                batch_times = times[batch_start:batch_end]

                for i, var in enumerate(var_names):
                    if var == "precip":
                        ddim_preds_np[:, :, :, :, i] = np.where(mask_pr, np.nan, ddim_preds_np[:, :, :, :, i])
                    elif var == "temp":
                        ddim_preds_np[:, :, :, :, i] = np.where(mask_tas, np.nan, ddim_preds_np[:, :, :, :, i])

                ds_ddim_batch = xr.Dataset(
                    {var: (("time", "sample", "N", "E"), ddim_preds_np[:, :, :, :, i])
                     for i, var in enumerate(var_names)},
                    coords={
                        "time": batch_times,
                        "sample": np.arange(num_samples),
                        "lat": (("N", "E"), lat2d) if lat2d is not None else None,
                        "lon": (("N", "E"), lon2d) if lon2d is not None else None,
                    }
                )
                all_ds_ddim.append(ds_ddim_batch)
                ds_ddim_batch.close()

            ds_ddim_full = xr.concat(all_ds_ddim, dim="time")
            ds_ddim_full.to_netcdf(out_path_ddim, encoding=encoding)
            ds_ddim_full.close()

            input_ds.close()
            target_ds.close()