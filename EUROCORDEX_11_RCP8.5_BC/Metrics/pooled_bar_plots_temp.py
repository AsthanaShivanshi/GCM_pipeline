import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_style("whitegrid")

import glob
from tqdm import tqdm

np.Inf = np.inf

from scipy.stats import gaussian_kde

#--------------------------------------------------

 #.. 

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


def filewise_stats(files, varname='tas', mask=None):
    medians = []
    p99s = []
    for file in tqdm(files, desc="Stats per file"):
        ds = xr.open_dataset(file, chunks={'time': 500})
        if varname not in ds.variables:
            if 'tas' in ds.variables:
                varname_used = 'tas'
            else:
                ds.close()
                continue
        else:

            varname_used = varname

        temp = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        if mask is not None:
            mask_aligned = mask
            if 'time' not in mask.dims:
                mask_aligned = mask.broadcast_like(temp.isel(time=0))
            masked = temp.where(mask_aligned).values.flatten()
        else:
            masked = temp.values.flatten()
        masked = masked[~np.isnan(masked)]
        if masked.size > 0:
            medians.append(np.median(masked))
            p99s.append(np.percentile(masked, 99))
        ds.close()
    return np.median(medians), np.median(p99s)



def filewise_stats_ddim(files, mask):
    medians = []
    p99s = []
    for file in tqdm(files, desc="Stats per file (DDIM)"):
        ds = xr.open_dataset(file, chunks={'time': 500})
        if 'tas' in ds.variables:
            varname_used = 'tas'
        elif 'TabsD' in ds.variables:
            varname_used = 'TabsD'
        else:
            ds.close()
            continue
        temp = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        mask_aligned = mask
        if 'time' not in mask.dims:
            mask_aligned = mask.broadcast_like(temp.isel(time=0))
        masked = temp.where(mask_aligned).values.flatten()
        masked = masked[~np.isnan(masked)]
        if masked.size > 0:
            medians.append(np.median(masked))
            p99s.append(np.percentile(masked, 99))
        ds.close()
    return np.median(medians), np.median(p99s)



def pooled_samples(files, varname='tas', mask=None, sample_per_file=2000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    samples = []
    for file in tqdm(files, desc="Sampling per file"):
        ds = xr.open_dataset(file, chunks={'time': 500})
        # Try to find the correct variable
        if varname in ds.variables:
            varname_used = varname
        elif 'tas' in ds.variables:
            varname_used = 'tas'
        elif 'TabsD' in ds.variables:
            varname_used = 'TabsD'
        elif 'temp' in ds.variables:
            varname_used = 'temp'
        else:
            print(f"File {file} has no tas, TabsD, or temp variable. Variables: {list(ds.variables)}")
            ds.close()
            continue
        temp = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        if mask is not None:
            mask_aligned = mask
            if 'time' not in mask.dims:
                mask_aligned = mask.broadcast_like(temp.isel(time=0))
            masked = temp.where(mask_aligned).values.flatten()
        else:
            masked = temp.values.flatten()
        masked = masked[~np.isnan(masked)]
        if masked.size > sample_per_file:
            samples.append(rng.choice(masked, size=sample_per_file, replace=False))
        elif masked.size > 0:
            samples.append(masked)
        else:
            print(f"File {file} has no valid data after masking/time selection.")
        ds.close()
    if samples:
        return np.concatenate(samples)
    else:
        return np.array([])


def pooled_samples_ddim(files, sample_per_file=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    samples = []
    for file in tqdm(files, desc="Sampling per file (DDIM)"):
        ds = xr.open_dataset(file, chunks={'time': 500})
        if 'tas' in ds.variables:
            varname_used = 'tas'
        elif 'TabsD' in ds.variables:
            varname_used = 'TabsD'
        elif 'temp' in ds.variables:
            varname_used = 'temp'
        else:
            print(f"File {file} has no tas, TabsD, or temp variable. Variables: {list(ds.variables)}")
            ds.close()
            continue
        temp = ds[varname_used].sel(time=slice("2011-01-01", "2020-12-30"))
        masked = temp.values.flatten()
        masked = masked[~np.isnan(masked)]
        if masked.size > sample_per_file:
            samples.append(rng.choice(masked, size=sample_per_file, replace=False))
        elif masked.size > 0:
            samples.append(masked)
        else:
            print(f"File {file} has no valid data after time selection.")
        ds.close()
    if samples:
        return np.concatenate(samples)
    else:
        return np.array([])
    
    
#--------------------------------------------------





obs_temp = xr.open_dataset("../Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc")["TabsD"].sel(time=slice("2011-01-01", "2020-12-30"))

coarse_temp_files = glob.glob("EUROCORDEX_11_RCP8.5/tas_Swiss/tas_day_EUR-11_*_1971-2099.nc")

bc_temp_files = glob.glob("EUROCORDEX_11_RCP8.5_BC/EQM/tas_day_EUR-11_*_1971-2099.nc")

bc_bicubic_temp_files = glob.glob("EUROCORDEX_11_RCP8.5_BC/EQM/tas_bicubic_EUR-11_*_1971-2099.nc")

bc_bicubic_unet_temp_files = glob.glob("ALP-FINE_8.5/EQM/UNet/UNet_RCP85_2011-2020_tas_*_rcp85_1971-2099.nc")
bc_bicubic_unet_temp_files = [
    f for f in bc_bicubic_unet_temp_files
    if any(str(year) in f for year in range(2011, 2021))
]

bc_bicubic_ddim_temp_files = glob.glob("ALP-FINE_8.5/EQM/DDIM/DDIM_6samples_RCP85_*.nc")


bc_bicubic_ddim_temp_files = [
    f for f in bc_bicubic_ddim_temp_files
    if any(str(year) in f for year in range(2011, 2021))
]




#--------------------------------------------------


obs_temp_flat = obs_temp.values.flatten()


obs_samples = obs_temp_flat if obs_temp_flat.size < 20000 else np.random.default_rng(42).choice(obs_temp_flat, size=20000, replace=False)


coarse_samples = pooled_samples(coarse_temp_files)

bc_samples = pooled_samples(bc_temp_files)

bc_bicubic_samples = pooled_samples(bc_bicubic_temp_files)

bc_bicubic_unet_samples = pooled_samples(bc_bicubic_unet_temp_files)
bc_bicubic_ddim_samples = pooled_samples_ddim(bc_bicubic_ddim_temp_files)


#--------------------------------------------------

data = [
    coarse_samples,
    bc_samples,
    bc_bicubic_samples,
    bc_bicubic_unet_samples,
    bc_bicubic_ddim_samples,
    obs_samples
]

labels = [
    "Coarse",
    "EQM",
    "EQM+Bicubic",
    "EQM+Bicubic+UNet",
    "EQM+Bicubic+UNet+DDIM (6 samples)",
    "Observations"
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


#----------------------------------

BINS = 100
RANGE = (min([d.min() for d in data_nonempty]), max([d.max() for d in data_nonempty]))

def compute_pdf(samples, bin_edges):
    hist, _ = np.histogram(samples, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    pdf = hist / (np.sum(hist) * bin_widths)
    return pdf

bin_edges = np.linspace(RANGE[0], RANGE[1], BINS + 1)
pdfs = [compute_pdf(d, bin_edges) for d in data_nonempty]






plt.figure(figsize=(14, 9))

x_grid = np.linspace(RANGE[0], RANGE[1], 1000)
for samples, label, color in zip(data_nonempty, labels_nonempty, colors_nonempty):
    kde = gaussian_kde(samples)
    plt.plot(
        x_grid,
        kde(x_grid),
        label=label,
        color=color,
        linewidth=2.5 if color == "#000000" else 2
    )

plt.xlabel("Daily Temp (°C)")
plt.ylabel("Probability Density")
plt.title("PDF of Daily Temperature (2011–2020)")
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/pdf_temp_2011_2020_samples.png", dpi=1000)


#----------------------------------

"""
fig, ax = plt.subplots(figsize=(14, 9))
box = ax.boxplot(data_nonempty, labels=labels_nonempty, patch_artist=True,
           boxprops=dict(color='black'),
           medianprops=dict(color='black', linewidth=2),
           whiskerprops=dict(color='black'),
           capprops=dict(color='black'),
           flierprops=dict(markerfacecolor='gray', marker='o', markersize=4, alpha=0.3))



for patch, color in zip(box['boxes'], colors_nonempty):
    patch.set_facecolor(color)



ax.set_xticklabels(labels_nonempty, rotation=20, ha="right")
ax.set_ylabel("Daily Temp (°C)")
#ax.set_title("Pooled daily temperature from EUR-12 RCP 8.5 MME (random subsample from 2011–2020)")
ax.grid(axis='y', linestyle='--', alpha=0.7)


plt.tight_layout()

plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/rcp85_EQM_pooled_temp_boxplot_2011_2020_samples.png", dpi=1000)
"""