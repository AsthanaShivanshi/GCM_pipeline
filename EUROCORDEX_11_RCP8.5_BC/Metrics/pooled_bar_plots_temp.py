
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

sns.set_style("whitegrid")

import glob
from tqdm import tqdm
import os
np.Inf = np.inf

import concurrent.futures

#--------------------------------------------------

BINS = 100
colors = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#000000"
]


num_workers = os.cpu_count()

#-----------------------------------------------------


plt.rcParams.update({
    "font.size": 28,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 28
})


#--------------------------------------------------


def filewise_stats(files, varname='tas', mask=None):
    medians = []
    p99s = []
    for file in tqdm(files, desc="Stats per file"):
        ds = xr.open_dataset(file, chunks={'time': 1000})
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
        ds = xr.open_dataset(file, chunks={'time': 1000})
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



def pooled_parallel(files, varname='tas', mask=None):

    def process_file(file):
        ds = xr.open_dataset(file, chunks={'time': 1000})
        if varname in ds.variables:
            varname_used = varname
        elif 'tas' in ds.variables:
            varname_used = 'tas'
        elif 'TabsD' in ds.variables:
            varname_used = 'TabsD'
        elif 'temp' in ds.variables:
            varname_used = 'temp'
        else:
            print(f"File {file} has no recognized temperature variable. Variables: {list(ds.variables)}")
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


obs_samples = obs_temp_flat

coarse_samples = pooled_parallel(coarse_temp_files)

bc_samples = pooled_parallel(bc_temp_files)

bc_bicubic_samples = pooled_parallel(bc_bicubic_temp_files)

bc_bicubic_unet_samples = pooled_parallel(bc_bicubic_unet_temp_files)


bc_bicubic_ddim_samples = pooled_parallel(bc_bicubic_ddim_temp_files)


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

#----------------------------------


if not data_nonempty:
    raise RuntimeError("No data available for plotting. Please check your input files.")

medians = [np.median(d) for d in data_nonempty]
p99s = [np.percentile(d, 99) for d in data_nonempty]


#----------------------------------


RANGE = (np.concatenate(data_nonempty).min(), np.concatenate(data_nonempty).max())

#----------------------------------


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
    plt.hist(
        samples,
        bins=50,
        density=True,
        alpha=0.5,
        label=label,
        color=color,
        linewidth=2
    )

plt.xlabel("Daily Temp (°C)")
plt.ylabel("Density")
plt.title("PDF of Daily Temperature (2011–2020)")
plt.legend(fontsize=14)


plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/pdf_temp_2011_2020_samples.png", dpi=1000)


#----------------------------------
""" #For violin

all_samples = []
all_labels = []
for samples, label in zip(data_nonempty, labels_nonempty):
    all_samples.extend(samples)
    all_labels.extend([label] * len(samples))

df = pd.DataFrame({'Temperature': all_samples, 'Dataset': all_labels})

plt.figure(figsize=(14, 9))
sns.violinplot(
    x='Dataset',
    y='Temperature',
    data=df,
    palette=colors_nonempty,
    cut=0,
    linewidth=1.5
)
plt.xticks(rotation=20, ha="right")
plt.ylabel("Daily Temp (°C)")
plt.xlabel("")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("EUROCORDEX_11_RCP8.5_BC/outputs/rcp85_EQM_pooled_temp_violinplot_2011_2020_all.png", dpi=1000)

"""