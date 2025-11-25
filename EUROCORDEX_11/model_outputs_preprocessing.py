import os
import xarray as xr
import config

# Mask file paths
PRECIP_MASK_PATH = os.path.join(config.EUROCORDEX_11_DIR, "precip_mask_11km.nc")
TEMP_MASK_PATH = os.path.join(config.EUROCORDEX_11_DIR, "temp_mask_11km.nc")

# IO folders


VAR_FOLDERS = {
    "pr": ("pr", "pr_Swiss", PRECIP_MASK_PATH),
    "tas": ("tas", "tas_Swiss", TEMP_MASK_PATH),
    "tasmax": ("tasmax", "tasmax_Swiss", TEMP_MASK_PATH),
    "tasmin": ("tasmin", "tasmin_Swiss", TEMP_MASK_PATH),

}

def mask_and_save(input_path, mask_path, output_folder, varname):
    fname = os.path.basename(input_path)
    outname = os.path.join(output_folder, fname.replace(".nc", "_Swiss_masked_11km.nc"))



    if os.path.exists(outname):
        print(f"Masked file exists: {outname}, skipping.")
        return
    ds = xr.open_dataset(input_path)
    mask_ds = xr.open_dataset(mask_path)
    mask = mask_ds["mask"]



    # Align mask to data grid using nearest neighbor interpolation
    mask_aligned = mask.reindex_like(ds[varname].isel(time=0), method="nearest") #Had to be done due to alignment issues between grids
    
    
    if "time" in ds[varname].dims and "time" not in mask_aligned.dims:
        mask_broadcast = mask_aligned.expand_dims({"time": ds[varname].coords["time"]}, axis=0)
    else:
        mask_broadcast = mask_aligned


    ds[varname] = ds[varname].where(mask_broadcast)
    os.makedirs(output_folder, exist_ok=True)

    ds.to_netcdf(outname, mode="w")
    ds.close()
    print(f"Masked and saved: {outname}")


for var, (in_folder, out_folder, mask_path) in VAR_FOLDERS.items():
    input_dir = os.path.join(config.EUROCORDEX_11_DIR, in_folder)
    output_dir = os.path.join(config.EUROCORDEX_11_DIR, out_folder)
    
    for fname in os.listdir(input_dir):
        if fname.endswith(".nc"):
            input_path = os.path.join(input_dir, fname)
            mask_and_save(input_path, mask_path, output_dir, var)