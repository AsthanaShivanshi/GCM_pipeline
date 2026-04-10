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


import subprocess






num_samples = 5  # Deterministic sample
eta = 0.0       # DDIM
S = 50          # Number of DDIM steps

manual_seed=124



ref_ds_temp = xr.open_dataset(f"{config.DATASETS_TRAINING_DIR}/TabsD_target_train_scaled.nc")
ref_lat = ref_ds_temp["lat"].values
ref_lon = ref_ds_temp["lon"].values


ref_ds_precip = xr.open_dataset(f"{config.DATASETS_TRAINING_DIR}/RhiresD_target_train_scaled.nc")

ref_grid_pr = f"{config.DATASETS_TRAINING_DIR}/RhiresD_step1_latlon.nc"
ref_grid_tas = f"{config.DATASETS_TRAINING_DIR}/TabsD_step1_latlon.nc"

bicubic_dir = f"{config.BIAS_CORRECTED_DIR_RCP85}/dOTC"





elevation_path = f"{config.BASE_DIR}/sasthana/Downscaling/GCM_pipeline/elevation.tif"
with rasterio.open(elevation_path) as src:
    elevation_array = src.read(1).astype(np.float32) #elev as numpy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params.json", 'r') as f:
    pr_params = json.load(f)
with open(f"{config.DATASETS_TRAINING_DIR}/TabsD_scaling_params.json", 'r') as f:
    temp_params = json.load(f)

def denorm_pr(x, pr_params):
    return np.exp(x * pr_params['std'] + pr_params['mean']) - pr_params['epsilon']

def denorm_temp(x, params):
    return x * params['std'] + params['mean']






unet_regr = DownscalingUnetLightning(
    in_ch=3,
    out_ch=2,
    features=[64, 128, 256, 512],
    channel_names=["precip", "temp"],
    precip_scaling_json=f"{config.DATASETS_TRAINING_DIR}/RhiresD_scaling_params.json",
)
unet_regr_ckpt = torch.load(
    f"{config.DM_DIR}/LDM_conditional/trained_ckpts_optimised/12km/UNet_ckpts/LDM_conditional.models.unet_module.DownscalingUnetLightning_12km_logtransform_lr0.001_precip_loss_weight1.0_1.0_crps[]_factor0.5_pat3.ckpt.ckpt",
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
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
    use_ema=True,
    ema_decay=0.9999,
    lr=1e-4
)



ddim_ckpt = torch.load(
    f"{config.DDIM_PROJ_PATH}/trained_ckpts/12km/DDIM_checkpoint_L1_cosine_schedule_loss_parameterisation_v.ckpt",
    map_location=device
)



ddim.load_state_dict(ddim_ckpt["state_dict"], strict=False)
ddim = ddim.to(device)
ddim.eval()
sampler = DDIMSampler(ddim, device=device)

tas_dir = f"{config.BIAS_CORRECTED_DIR_RCP85}/dOTC"
pr_dir = f"{config.BIAS_CORRECTED_DIR_RCP85}/dOTC"



def get_id(path, var_prefix):
    fname = os.path.basename(path)
    return fname.replace(f"{var_prefix}_day_", "")




tas_files = [os.path.join(tas_dir, f) for f in os.listdir(tas_dir) if f.startswith("tas_day") and f.endswith(".nc")]
pr_files = [os.path.join(pr_dir, f) for f in os.listdir(pr_dir) if f.startswith("pr_day") and f.endswith(".nc")]

pr_dict = {get_id(f, "pr"): f for f in pr_files}




config_dict = {
    'variables': {
        'input': {'precip': 'pr', 'temp': 'tas'},
        'target': {'precip': 'pr', 'temp': 'tas'}
    },
    'preprocessing': {'nan_to_num': True, 'nan_value': 0.0}
}



os.makedirs("ALP-FINE_8.5/dOTC", exist_ok=True)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, required=True, help="Start year")
    parser.add_argument("--end_year", type=int, required=True, help="End year")
    args = parser.parse_args()




    for tas_path in tas_files:
        tas_id = get_id(tas_path, "tas")
        pr_path = pr_dict.get(tas_id, None)
        if pr_path is None:
            print(f"No matching pr file for {tas_path}")
            continue

        print(f"Processing {tas_path} and {pr_path}")



# bicubic
        bicubic_tas_path = os.path.join(
            bicubic_dir, f"tas_bicubic_{tas_id}"
        )
        bicubic_pr_path = os.path.join(
            bicubic_dir, f"pr_bicubic_{get_id(pr_path, 'pr')}"
        )



        if not os.path.exists(bicubic_tas_path):
            print(f"Running CDO bicubic interpolation for {tas_path}")
            subprocess.run([
                "cdo", "remapbic," + ref_grid_tas, tas_path, bicubic_tas_path
            ], check=True)
        else:
            print(f"Bicubic tas already exists: {bicubic_tas_path}")

        # Interpolate pr
        if not os.path.exists(bicubic_pr_path):
            print(f"Running CDO bicubic interpolation for {pr_path}")
            subprocess.run([
                "cdo", "remapbic," + ref_grid_pr, pr_path, bicubic_pr_path
            ], check=True)
        else:
            print(f"Bicubic pr already exists: {bicubic_pr_path}")
        
        
        
        tas_path = bicubic_tas_path
        pr_path = bicubic_pr_path


        print(f"Processing {tas_path} and {pr_path}")
        
        input_ds = {
            'precip': xr.open_dataset(pr_path),
            'temp': xr.open_dataset(tas_path)
        }
        target_ds = {
            'precip': xr.open_dataset(pr_path),
            'temp': xr.open_dataset(tas_path)
        } #used for dataloader, not used for inf. 




        time_start = f"{args.start_year}-01-01"
        time_end = f"{args.end_year}-12-31"
        for v in input_ds:
            input_ds[v] = input_ds[v].sel(time=slice(time_start, time_end))
        for v in target_ds:
            target_ds[v] = target_ds[v].sel(time=slice(time_start, time_end))




        ds = DownscalingDataset(input_ds, target_ds, config_dict, elevation_path=elevation_array)
        
        
        input_tensor, _ = ds[0]
        spatial_shape = input_tensor.shape[1:]  # (H, W)
        times = input_ds['temp']['time'].values
        N = len(ds)
        unet_all = np.empty((N, 2, *spatial_shape), dtype=np.float32)
        ddim_all = np.empty((N, num_samples, 2, *spatial_shape), dtype=np.float32)
        params_list = [pr_params, temp_params]




        var_names = ["precip", "temp"]
        N_vals = np.arange(spatial_shape[0])
        E_vals = np.arange(spatial_shape[1])
        lat2d, lon2d = None, None
        if ref_lat.ndim == 2 and ref_lon.ndim == 2:
            lat2d, lon2d = ref_lat, ref_lon
        elif ref_lat.ndim == 1 and ref_lon.ndim == 1:
            lat2d, lon2d = np.meshgrid(ref_lat, ref_lon, indexing="ij")
        encoding = {}

        

        #UNet 

        for idx in tqdm(range(N), desc=f"Downscaling {os.path.basename(tas_path)}"):
            input_tensor, _ = ds[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                unet_pred = unet_regr(input_tensor)
                unet_pred_np = unet_pred[0].cpu().numpy()
                unet_pred_denorm = np.empty_like(unet_pred_np)
                for i, params in enumerate(params_list):
                    unet_pred_denorm[i] = denorm_pr(unet_pred_np[i], pr_params) if i == 0 else denorm_temp(unet_pred_np[i], params)
                unet_all[idx] = unet_pred_denorm

        unet_preds_np = np.transpose(unet_all, (0, 2, 3, 1))  # (time, y, x, channel)








        ds_unet = xr.Dataset(
            {
                var: (("time", "N", "E"), unet_preds_np[:, :, :, i])
                for i, var in enumerate(var_names)
            },
            coords={
                "time": times,
                "N": N_vals,
                "E": E_vals,
                "lat": (("N", "E"), lat2d) if lat2d is not None else None,
                "lon": (("N", "E"), lon2d) if lon2d is not None else None,
            }
        )

        out_path_unet = f"ALP-FINE_8.5/dOTC/UNet_RCP85_{args.start_year}-{args.end_year}_tas_{get_id(pr_path, 'pr')}"
        ds_unet.to_netcdf(out_path_unet, encoding=encoding)
        print(f"UNet Op saved to {out_path_unet}")






        #DDIM
        for idx in tqdm (range(N), desc="DDIM Sampling"):
            input_tensor,_ = ds[idx]
            input_tensor= input_tensor.unsqueeze(0).to(device)


            with torch.no_grad():
                unet_pred = unet_regr(input_tensor)
                context = [(unet_pred, None)]  # Pass the full unet_pred and None for t_relative
                sample_shape = unet_pred.shape[1:]  # (C, H, W)

                for j in range(num_samples):
                    torch.manual_seed(manual_seed + j)
                    np.random.seed(manual_seed + j)

                    z = torch.randn((1, *sample_shape), device=device)

#residual is a tensor, niot a tuple : AsthanaSh
                    residual, _ = sampler.sample(
                        S=S,
                        batch_size=1,
                        shape=sample_shape,
                        conditioning=context,
                        eta=eta,
                        verbose=False,
                        x_T=z,
                        schedule="cosine"
                    )


                    final_pred = unet_pred + residual

                    final_pred_np = final_pred[0].cpu().numpy()
                    ddim_pred_denorm = np.empty_like(final_pred_np)
                                
                                
                    for i, params in enumerate(params_list):
                        ddim_pred_denorm[i] = denorm_pr(final_pred_np[i], pr_params) if i == 0 else denorm_temp(final_pred_np[i], params)
                    ddim_all[idx, j] = ddim_pred_denorm

        ddim_preds_np = np.transpose(ddim_all, (0, 1, 3, 4, 2))  # (time, sample, y, x, channel)

        ds_ddim= xr.Dataset(
            {var: (("time", "sample", "N", "E"), ddim_preds_np[:, :, :, :, i]) 
            for i, var in enumerate(var_names)},
            coords={"time":times,
            "sample": np.arange(num_samples),
             "lat": (("N", "E"), lat2d) if lat2d is not None else None,
            "lon": (("N", "E"), lon2d) if lon2d is not None else None,
                    }
                )

        out_path_ddim = f"ALP-FINE_8.5/dOTC/DDIM_5samples_RCP85_{args.start_year}-{args.end_year}_tas_{get_id(pr_path, 'pr')}"
        ds_ddim.to_netcdf(out_path_ddim, encoding=encoding)

        for ds_ in input_ds.values():
            ds_.close()
        for ds_ in target_ds.values():
            ds_.close()