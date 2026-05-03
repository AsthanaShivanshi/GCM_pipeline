import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
import pandas as pd
sns.set_style("whitegrid")

import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

import concurrent.futures

np.Inf = np.inf
import os
num_workers = os.cpu_count() 




BINS = 100
RANGE = (0.01, 300)
EPS = 1e-10 #for non zero log error 

#--------------------------------------------------
plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 28
})



colors = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#CC79A7",  # purple
    "#000000"   # black for Observations
]

#--------------------------------------------------


def filewise_stats(files, varname='pr', mask=None):
    medians = []
    p99s = []
    for file in tqdm(files, desc="Stats per file"):
        ds = xr.open_dataset(file, chunks={'time': 1000})
        if varname not in ds.variables:
            if 'precip' in ds.variables:
                varname_used = 'precip'
            else:
                ds.close()
                continue
        else:
            varname_used = varname
        pr = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        if mask is not None:
            mask_aligned = mask
            if 'time' not in mask.dims:
                mask_aligned = mask.broadcast_like(pr.isel(time=0))
            masked = pr.where(mask_aligned).values.flatten()
        else:
            masked = pr.values.flatten()
        masked = masked[~np.isnan(masked)]
        masked = np.clip(masked, 0, None)
        if masked.size > 0:
            medians.append(np.median(masked))
            p99s.append(np.percentile(masked, 99))
        ds.close()
    return np.median(medians), np.median(p99s)



def filewise_stats_ddim(files, mask):
    medians = []
    p99s = []
    for file in tqdm(files, desc="Stats per file (DDIM)"):
        ds = xr.open_dataset(file, chunks={'time': 1000})
        if 'pr' in ds.variables:
            varname_used = 'pr'
        elif 'precip' in ds.variables:
            varname_used = 'precip'
        else:
            ds.close()
            continue
        pr = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        mask_aligned = mask
        if 'time' not in mask.dims:
            mask_aligned = mask.broadcast_like(pr.isel(time=0))
        masked = pr.where(mask_aligned).values.flatten()
        masked = masked[~np.isnan(masked)]
        masked = np.clip(masked, 0, None)
        if masked.size > 0:
            medians.append(np.median(masked))
            p99s.append(np.percentile(masked, 99))
        ds.close()
    return np.median(medians), np.median(p99s)



def pooled_parallel(files, varname='pr', mask=None):
    def process_file(file):
        ds = xr.open_dataset(file, chunks={'time': 1000})
        if varname in ds.variables:
            varname_used = varname
        elif 'pr' in ds.variables:
            varname_used = 'pr'
        elif 'RhiresD' in ds.variables:
            varname_used = 'RhiresD'
        elif 'precip' in ds.variables:
            varname_used = 'precip'
        else:
            print(f"File {file} has no recognized precip variable. Variables: {list(ds.variables)}")
            ds.close()
            return None
        temp = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        if mask is not None:
            mask_aligned = mask
            if 'time' not in mask.dims:
                mask_aligned = mask.broadcast_like(temp.isel(time=0))
            masked = temp.where(mask_aligned).values.flatten()
        else:
            masked = temp.values.flatten()
        masked = masked[~np.isnan(masked)]
        masked = np.clip(masked, EPS, None)
        ds.close()



        if masked.size > 0:
            return masked
        else:
            print(f"File {file} has no valid data after masking/time selection.")
            return None


        pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        
        results = list(executor.map(process_file, files))
        samples = [r for r in results if r is not None and r.size > 0]
        if samples:
            return np.concatenate(samples)
        else:
            return np.array([])
    


#--------------------------------------------------





obs_precip = xr.open_dataset("../Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc")["RhiresD"].sel(time=slice("2011-01-01", "2020-12-30"))
obs_temp = xr.open_dataset("../Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc")["TabsD"].sel(time=slice("2011-01-01", "2011-01-02"))

obs_mask_precip = ~np.isnan(obs_temp.isel(time=0))

coarse_precip_files = glob.glob("EUROCORDEX_11_RCP8.5/pr_Swiss/pr_day_EUR-11_*_1971-2099.nc")
bc_precip_files = glob.glob("EUROCORDEX_11_RCP8.5_BC/EQM/pr_day_EUR-11_*_1971-2099.nc")
bc_bicubic_precip_files = glob.glob("EUROCORDEX_11_RCP8.5_BC/EQM/pr_bicubic_EUR-11_*_1971-2099.nc")

bc_bicubic_unet_precip_files = glob.glob("ALP-FINE_8.5/EQM/UNet/UNet_RCP85_2011-2020_tas_*_rcp85_1971-2099.nc")




bc_bicubic_unet_precip_files = [
    f for f in bc_bicubic_unet_precip_files
    if any(str(year) in f for year in range(2011, 2021))
]

bc_bicubic_ddim_precip_files = glob.glob("ALP-FINE_8.5/EQM/DDIM/DDIM_6samples_RCP85_*.nc")
bc_bicubic_ddim_precip_files = [
    f for f in bc_bicubic_ddim_precip_files
    if any(str(year) in f for year in range(2011, 2021))
]




#--------------------------------------------------


coarse_samples = pooled_parallel(coarse_precip_files, mask=obs_mask_precip)
bc_samples = pooled_parallel(bc_precip_files, mask=obs_mask_precip)
bc_bicubic_samples = pooled_parallel(bc_bicubic_precip_files, mask=obs_mask_precip)
bc_bicubic_unet_samples = pooled_parallel(bc_bicubic_unet_precip_files, mask=obs_mask_precip)
bc_bicubic_ddim_samples = pooled_parallel(bc_bicubic_ddim_precip_files, mask=obs_mask_precip)

obs_precip_clipped = np.clip(obs_precip.values[~np.isnan(obs_precip.values)], 0.01, None)
obs_samples = obs_precip_clipped



data = [
    coarse_samples,
    bc_samples,
    bc_bicubic_samples,
    bc_bicubic_unet_samples,
    bc_bicubic_ddim_samples,
    obs_samples
]

#--------------------------------------------------

labels = [
    "Coarse",
    "EQM",
    "EQM+Bicubic",
    "EQM+Bicubic+UNet",
    "EQM+Bicubic+UNet+DDIM (6 samples)",
    "Observations"
]
#--------------------------------------------------



medians = [
    np.median(coarse_samples),
    np.median(bc_samples),
    np.median(bc_bicubic_samples),
    np.median(bc_bicubic_unet_samples),
    np.median(bc_bicubic_ddim_samples),
    np.median(obs_samples)
]
p99s = [
    np.percentile(coarse_samples, 99),
    np.percentile(bc_samples, 99),
    np.percentile(bc_bicubic_samples, 99),
    np.percentile(bc_bicubic_unet_samples, 99),
    np.percentile(bc_bicubic_ddim_samples, 99),
    np.percentile(obs_samples, 99)
]

#----------------------------------



data_nonempty = []
labels_nonempty = []
colors_nonempty = []

for d, l, c in zip(data, labels, colors):
    d_clean = d[~np.isnan(d)]
    if d_clean.size > 0:
        data_nonempty.append(d_clean)
        labels_nonempty.append(l)
        if l.lower().startswith("obs"):
            colors_nonempty.append("#000000")
        else:
            colors_nonempty.append(c)
    else:
        print(f"Warning: No valid data found for {l}, skipping.")

if not data_nonempty:
    raise RuntimeError("No data available for plotting. Please check your input files.")


medians = [np.median(d) for d in data_nonempty]
p99s = [np.percentile(d, 99) for d in data_nonempty]



#plot
#----------------------------------
# Log-transform for histogram
log_data_nonempty = [np.log10(d) for d in data_nonempty]




plt.figure(figsize=(14, 9))
log_range = (np.log10(0.01), np.log10(300))
x_grid = np.linspace(log_range[0], log_range[1], 1000)

for log_samples, label, color in zip(log_data_nonempty, labels_nonempty, colors_nonempty):
    plt.hist(
        log_samples,
        bins=BINS,
        density=True,
        alpha=0.5,
        label=label,
        color=color,
        linewidth=2
    )

plt.xlabel("log10(Daily Precip [mm/day])")
plt.ylabel("Probability Density")
plt.title("PDF of log10 (Precip) (2011–2020)")
plt.legend() 
plt.tight_layout()
plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/pdf_log_precip_2011_2020_samples.png", dpi=1000)


#--------------------------------------------------

""" #For violin

# Prepare data for violin plot (all points, not subsampled)
all_samples = []
all_labels = []
for samples, label in zip(data_nonempty, labels_nonempty):
    all_samples.extend(samples)
    all_labels.extend([label] * len(samples))

df = pd.DataFrame({'Precipitation': all_samples, 'Dataset': all_labels})

plt.figure(figsize=(14, 9))
sns.violinplot(
    x='Dataset',
    y='Precipitation',
    data=df,
    palette=colors_nonempty,
    cut=0,
    linewidth=1.5
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Daily Precip (mm/day)")
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 250)
plt.tight_layout()
plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/rcp85_EQM_pooled_precip_violinplot_2011_2020_all.png", dpi=1000)


"""