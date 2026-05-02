import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_style("whitegrid")

import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

np.Inf = np.inf

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



def pooled_all_samples(files, varname='pr', mask=None, min_clip=0.01):
    samples = []
    for file in tqdm(files, desc="Pooling all samples"):
        ds = xr.open_dataset(file, chunks={'time': 1000})
        # Robust variable selection
        if varname in ds.variables:
            varname_used = varname
        elif 'pr' in ds.variables:
            varname_used = 'pr'
        elif 'precip' in ds.variables:
            varname_used = 'precip'
        else:
            print(f"File {file} has no pr/precip variable. Variables: {list(ds.variables)}")
            ds.close()
            continue
        pr = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        if mask is not None:
            mask_aligned = mask
            if 'time' not in mask.dims:
                mask_aligned = mask.broadcast_like(pr.isel(time=0))
            masked = pr.where(mask_aligned).values.flatten()
        else:
            masked = pr.values.flatten()
        masked = masked[~np.isnan(masked)]
        masked = np.clip(masked, min_clip, None)  # Clip to min_clip for log
        if masked.size > 0:
            samples.append(masked)
        ds.close()
    if samples:
        return np.concatenate(samples)
    else:
        return np.array([])
    



def pooled_samples_ddim(files, mask, sample_per_file=3000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    samples = []
    for file in tqdm(files, desc="Sampling per file (DDIM)"):
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
        if masked.size > sample_per_file:
            samples.append(rng.choice(masked, size=sample_per_file, replace=False))
        elif masked.size > 0:
            samples.append(masked)
        ds.close()
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



obs_precip_clipped = np.clip(obs_precip.values[~np.isnan(obs_precip.values)], 0, None)
obs_samples = obs_precip_clipped if obs_precip_clipped.size < 10000 else np.random.default_rng(42).choice(obs_precip_clipped, size=10000, replace=False)

coarse_samples = pooled_all_samples(coarse_precip_files, mask=obs_mask_precip)
bc_samples = pooled_all_samples(bc_precip_files, mask=obs_mask_precip)
bc_bicubic_samples = pooled_all_samples(bc_bicubic_precip_files, mask=obs_mask_precip)
bc_bicubic_unet_samples = pooled_all_samples(bc_bicubic_unet_precip_files, mask=obs_mask_precip)
bc_bicubic_ddim_samples = pooled_all_samples(bc_bicubic_ddim_precip_files, mask=obs_mask_precip)

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
# Log-transform for histogram/KDE
log_data_nonempty = [np.log10(d) for d in data_nonempty]

BINS = 100
RANGE = (0.01, 300)
EPS = 1e-10


plt.figure(figsize=(14, 9))
log_range = (np.log10(0.01), np.log10(300))
x_grid = np.linspace(log_range[0], log_range[1], 1000)

for log_samples, label, color in zip(log_data_nonempty, labels_nonempty, colors_nonempty):
    kde = gaussian_kde(log_samples)
    plt.plot(
        x_grid,
        kde(x_grid),
        label=label,
        color=color,
        linewidth=2.5 if color == "#000000" else 2
    )

plt.xlabel("log10(Daily Precip [mm/day])")
plt.ylabel("Probability Density")
plt.title("PDF of log10 (Precip) (2011–2020)")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/pdf_log_precip_2011_2020_samples.png", dpi=1000)


#----------------------------------

#--------------------------------------------------


#For boxplots 
"""fig, ax = plt.subplots(figsize=(14, 9))

box = ax.boxplot(data, labels=labels, patch_artist=True,
           boxprops=dict(color='black'),
           medianprops=dict(color='black', linewidth=2),
           whiskerprops=dict(color='black'),
           capprops=dict(color='black'),
           flierprops=dict(markerfacecolor='gray', marker='o', markersize=4, alpha=0.3))

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


    
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Daily Precip (mm/day)")
#ax.set_title("Pooled precipitation from EUR-12 RCP 8.5 MME (random subsample from 2011–2020)")
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 250)
plt.tight_layout()


plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/rcp85_EQM_pooled_precip_boxplot_2011_2020_samples.png", dpi=1000)"""
