import os
import glob
import subprocess
import config


CH_BOX = [5, 12, 45, 48]  # rough coords

PR_TARGET = f"{config.DATASETS_TRAINING_DIR}/RhiresD_step2_coarse.nc"
TAS_TARGET = f"{config.DATASETS_TRAINING_DIR}/TabsD_step2_coarse.nc"
BILINEAR_TARGET_PR = f"{config.DATASETS_TRAINING_DIR}/RhiresD_step3_interp.nc"
BILINEAR_TARGET_TAS = f"{config.DATASETS_TRAINING_DIR}/TabsD_step3_interp.nc"

SWISS_ROOT = f"{config.GCM_PIPELINE_DIR}/ALP-FINEv1.0/Swiss"
BILINEAR_ROOT = f"{config.GCM_PIPELINE_DIR}/ALP-FINEv1.0/Bilinear"
os.makedirs(SWISS_ROOT, exist_ok=True)
os.makedirs(BILINEAR_ROOT, exist_ok=True)

for model in ["EC-Earth3-Veg", "MPI-ESM1-2-HR", "NorESM2-MM"]:
    for time in ["historical", "ssp370"]:
        for var in ["tas", "pr"]:
            rel_path = f"{model}/{time}/r1i1p1f1/RegCM5-0/v1-r1/day/{var}/v20250415"

            model_path_allfiles = f"{config.CORDEX_CMIP6}/{rel_path}"
            swiss_outpath = f"{SWISS_ROOT}/{rel_path}"
            bilinear_outpath = f"{BILINEAR_ROOT}/{rel_path}"

            os.makedirs(swiss_outpath, exist_ok=True)
            os.makedirs(bilinear_outpath, exist_ok=True)

            print(f"Looking in: {model_path_allfiles}", flush=True)

            files = sorted(glob.glob(f"{model_path_allfiles}/*.nc"))
            print(f"Found {len(files)} files", flush=True)

            if not files:
                continue

            coarse_target_grid = TAS_TARGET if var == "tas" else PR_TARGET
            bilinear_target_grid = BILINEAR_TARGET_TAS if var == "tas" else BILINEAR_TARGET_PR

            merged_input = os.path.join(swiss_outpath, f"{model}_{time}_{var}_merged_input.nc")
            step1 = os.path.join(swiss_outpath, f"{model}_{time}_{var}_step1_crop.nc")
            step2 = os.path.join(swiss_outpath, f"{model}_{time}_{var}_step2_raw.nc")

            swiss_outname = os.path.join(
                swiss_outpath, f"{model}_{time}_{var}_Swiss.nc"
            )
            bilinear_outname = os.path.join(
                bilinear_outpath, f"{model}_{time}_{var}_bilinear.nc"
            )

            print(f"Merging {len(files)} files into: {merged_input}", flush=True)
            subprocess.run(["cdo", "mergetime", *files, merged_input], check=True)

            print(f"Creating Swiss file: {swiss_outname}", flush=True)
            subprocess.run(
                [
                    "cdo",
                    f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
                    merged_input,
                    step1,
                ],
                check=True,
            )

            subprocess.run(
                [
                    "cdo",
                    f"remapnn,{coarse_target_grid}",
                    step1,
                    step2,
                ],
                check=True,
            )

            if var == "tas":
                subprocess.run(
                    [
                        "cdo",
                        "subc,273.15",
                        step2,
                        swiss_outname,
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "cdo",
                        "mulc,86400",
                        step2,
                        swiss_outname,
                    ],
                    check=True,
                )

            print(f"Creating bilinear file: {bilinear_outname}", flush=True)
            subprocess.run(
                [
                    "cdo",
                    f"remapbil,{bilinear_target_grid}",
                    swiss_outname,
                    bilinear_outname,
                ],
                check=True,
            )

            for tmp in [merged_input, step1, step2]:
                if os.path.exists(tmp):
                    os.remove(tmp)