import sys
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import xarray as xr
import properscoring as ps
import config

sys.path.append(config.DDIM_PROJ_PATH)

from models.unet_module import DownscalingUnetLightning
from DownscalingDataModule import DownscalingDataModule
from Downscaling_Dataset_Prep import DownscalingDataset

ref_ds = xr.open_dataset(f"{config.DATASETS_TRAINING_DIR}/TabsD_target_train_scaled.nc")
ref_lat = ref_ds["lat"].values
ref_lon = ref_ds["lon"].values

elev_ds = xr.open_dataset("elevation.tif")
elev = elev_ds["elevation"].values
elev_norm = (elev - np.mean(elev)) / np.std(elev)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open(f"{config.DATASETS_TRAINING_DIR}/TabsD_scaling_params.json", 'r') as f:
    temp_params = json.load(f)

def normalize_temp(x, params):
    return (x - params['mean']) / params['std']

def zlog_pr(x, params):
    return (np.log(x + params['epsilon']) - params['mean']) / params['std']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']

def reverse_zlog_pr(x, params):
    return np.exp(x * params['std'] + params['mean']) - params['epsilon']

# UNet
unet_regr = DownscalingUnetLightning(
    in_ch=3,
    out_ch=2,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp"],
    precip_scaling_json=f"{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params.json",
)



unet_regr_ckpt = torch.load(
    f"{config.LDM_PROJ_PATH}/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_logtransform_lr0.01_precip_loss_weight5.0_1.0_crps[0, 1]_factor0.5_pat3.ckpt.ckpt",
    map_location="cpu"
)["state_dict"]


unet_regr.load_state_dict(unet_regr_ckpt, strict=False)
unet_regr = unet_regr.to(device)
unet_regr.eval()



tas_dir = f"{config.BIAS_CORRECTED_DIR_RCP26}/CDFT"
pr_dir = f"{config.BIAS_CORRECTED_DIR_RCP26}/CDFT"

def get_id(path, var_prefix):
    fname = os.path.basename(path)
    return fname.replace(f"{var_prefix}_day_", "")

tas_files = [os.path.join(tas_dir, f) for f in os.listdir(tas_dir) if f.startswith("tas_day") and f.endswith(".nc")]
pr_files = [os.path.join(pr_dir, f) for f in os.listdir(pr_dir) if f.startswith("pr_day") and f.endswith(".nc")]

pr_dict = {get_id(f, "pr"): f for f in pr_files}



# config 
config_dict = {
    'variables': {
        'input': {'precip': 'pr', 'temp': 'tas'},
        'target': {'precip': 'pr', 'temp': 'tas'}
    },
    'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
}



os.makedirs("EUR-CH(SR)", exist_ok=True)

for tas_path in tas_files:
    tas_id = get_id(tas_path, "tas")
    pr_path = pr_dict.get(tas_id, None)
    if pr_path is None:
        print(f"No matching pr file for {tas_path}")
        continue

    print(f"Processing {tas_path} and {pr_path}")

    input_ds = {
        'precip': xr.open_dataset(pr_path),
        'temp': xr.open_dataset(tas_path)
    }


    target_ds = {
        'precip': xr.open_dataset(pr_path),
        'temp': xr.open_dataset(tas_path)
    }




    for v in input_ds:
        input_ds[v] = input_ds[v].sel(time=slice("2011-01-01", "2023-12-31"))
        input_ds[v] = input_ds[v].interp(lat=ref_lat, lon=ref_lon, method="cubic")
    for v in target_ds:
        target_ds[v] = target_ds[v].sel(time=slice("2011-01-01", "2023-12-31"))
        target_ds[v] = target_ds[v].interp(lat=ref_lat, lon=ref_lon, method="cubic")



    # DownscalingDataset for consistency ,,, else was giving error
    ds = DownscalingDataset(input_ds, target_ds, config_dict, elevation_path=elev_norm)
    times = input_ds['temp']['time'].values

    unet_all = np.empty((len(ds), 2, elev_norm.shape[0], elev_norm.shape[1]), dtype=np.float32)

    for idx in tqdm(range(len(ds)), desc="Downscaling RCP26 frames for 4 runs"):
        input_tensor, _ = ds[idx] 
        input_tensor = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            unet_pred = unet_regr(input_tensor)
        unet_pred_np = unet_pred[0].cpu().numpy()
        pr_down = reverse_zlog_pr(unet_pred_np[0], pr_params)
        tas_down = denorm_temp(unet_pred_np[1], temp_params)
        unet_all[idx, 0] = pr_down
        unet_all[idx, 1] = tas_down

    out_path = f"EUR-CH(SR)/UNet_downscaled_RCP26_CDFT_{os.path.basename(tas_path)}"
    ds_out = xr.Dataset(
        {
            "pr_downscaled": (("time", "y", "x"), unet_all[:, 0]),
            "tas_downscaled": (("time", "y", "x"), unet_all[:, 1]),
        },
        coords={
            "time": times,
            "y": np.arange(unet_all.shape[2]),
            "x": np.arange(unet_all.shape[3]),
            "lat": (("y", "x"), ref_lat),
            "lon": (("y", "x"), ref_lon),
        }
    )
    ds_out.to_netcdf(out_path)
    print(f"Saved downscaled output to {out_path}")