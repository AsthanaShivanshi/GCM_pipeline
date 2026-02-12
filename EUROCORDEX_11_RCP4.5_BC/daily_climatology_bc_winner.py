import xarray as xr
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import subprocess 
import os

np.Inf = np.inf

cdf_path = "CDFT/ensemble_median_pr_day_rcp45.nc"
dotc_path = "dOTC/ensemble_median_pr_day_rcp45.nc"
eqm_path = "EQM/ensemble_median_pr_day_rcp45.nc"
ref_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc"
bicubic_target_grid = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/RhiresD_step3_interp.nc"

def cdo_bicubic_interpolate(input_file, target_grid_file, output_file):
    if not os.path.exists(output_file):
        cmd = [
            "cdo",
            f"remapbic,{target_grid_file}",
            input_file,
            output_file
        ]
        subprocess.run(cmd, check=True)


cdf_interp_path = "CDFT/ensemble_median_pr_day_rcp45_bicubic.nc"
dotc_interp_path = "dOTC/ensemble_median_pr_day_rcp45_bicubic.nc"
eqm_interp_path = "EQM/ensemble_median_pr_day_rcp45_bicubic.nc"

cdo_bicubic_interpolate(cdf_path, bicubic_target_grid, cdf_interp_path)
cdo_bicubic_interpolate(dotc_path, bicubic_target_grid, dotc_interp_path)
cdo_bicubic_interpolate(eqm_path, bicubic_target_grid, eqm_interp_path)

cdf_ds = xr.open_dataset(cdf_interp_path)
dotc_ds = xr.open_dataset(dotc_interp_path)
eqm_ds = xr.open_dataset(eqm_interp_path)
ref_ds = xr.open_dataset(ref_path)

start = np.datetime64('2011-01-01')
end = np.datetime64('2023-12-31')
cdf_ds = cdf_ds.sel(time=slice(start, end))
dotc_ds = dotc_ds.sel(time=slice(start, end))
eqm_ds = eqm_ds.sel(time=slice(start, end))
ref_ds = ref_ds.sel(time=slice(start, end))

cdf = cdf_ds['pr']  # (time, lat, lon)
dotc = dotc_ds['pr']
eqm = eqm_ds['pr']
ref = ref_ds['RhiresD']

# Mask for Swiss domain
mask = ~np.isnan(ref.isel(time=0).values)

def gridwise_perkins_skill_score_vec(a, b, nbins=50):
    # a, b: shape (time, lat, lon)
    lat = min(a.shape[1], b.shape[1])
    lon = min(a.shape[2], b.shape[2])
    pss = np.full((lat, lon), np.nan)
    for i in range(lat):
        for j in range(lon):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask_ = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask_) > 10:
                a_valid = a1[mask_]
                b_valid = b1[mask_]
                combined_data = np.concatenate([a_valid, b_valid])
                bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
                hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
                hist_a = hist_a / np.sum(hist_a)
                hist_b = hist_b / np.sum(hist_b)
                pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
    return pss

# Vectorized PSS 
pss_cdf = gridwise_perkins_skill_score_vec(cdf.values, ref.values)
pss_dotc = gridwise_perkins_skill_score_vec(dotc.values, ref.values)
pss_eqm = gridwise_perkins_skill_score_vec(eqm.values, ref.values)

pss_stack = np.stack([pss_cdf, pss_dotc, pss_eqm], axis=0) 

winner = np.argmax(pss_stack, axis=0).astype(float)
winner[np.any(np.isnan(pss_stack), axis=0) | ~mask] = np.nan



total_cells = np.sum(mask)
percentages = []
for i in range(3):
    count = np.sum((winner == i) & mask)
    perc = 100 * count / total_cells
    percentages.append(perc)

labels = [
    f"CDFT ({percentages[0]:.1f}%)",
    f"dOTC ({percentages[1]:.1f}%)",
    f"EQM ({percentages[2]:.1f}%)"
]

cbf_colors = ['green', 'orange', 'purple']  # CDFT = green, dOTC = orange, EQM = purple
cmap = ListedColormap(cbf_colors)

fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)
im = ax.imshow(winner, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("", fontsize=15)
ax.set_ylabel("", fontsize=15)
for spine in ax.spines.values():
    spine.set_visible(False)
legend_elements = [
    Patch(facecolor='green', label=labels[0]),
    Patch(facecolor='orange', label=labels[1]),
    Patch(facecolor='purple', label=labels[2])
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=18, frameon=False)
ax.set_title("BC+ Bicubically Interpolated RCP45 Ensemble Median; 2011-2023 for Precipitation", fontsize=22, fontweight='bold')
plt.savefig("../outputs/gridwise_pss_winner_pr_climatology_rcp45_bicubic.png", dpi=1000, bbox_inches='tight')
plt.close()