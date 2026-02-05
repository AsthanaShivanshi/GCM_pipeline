import xarray as xr
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.Inf= np.inf

cdf_path = "CDFT/ensemble_median_tas_day_rcp45.nc"
dotc_path = "dOTC/ensemble_median_tas_day_rcp45.nc"
eqm_path = "EQM/ensemble_median_tas_day_rcp45.nc"
ref_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km/TabsD_step2_coarse.nc"

cdf_ds = xr.open_dataset(cdf_path)
dotc_ds = xr.open_dataset(dotc_path)
eqm_ds = xr.open_dataset(eqm_path)
ref_ds = xr.open_dataset(ref_path)

start = np.datetime64('2011-01-01')
end = np.datetime64('2023-12-31')
cdf_ds = cdf_ds.sel(time=slice(start, end))
dotc_ds = dotc_ds.sel(time=slice(start, end))
eqm_ds = eqm_ds.sel(time=slice(start, end))
ref_ds = ref_ds.sel(time=slice(start, end))

cdf = cdf_ds['tas']  # (time, lat, lon)
dotc = dotc_ds['tas']
eqm = eqm_ds['tas']
ref = ref_ds['TabsD']

# Mask for Swiss domain (assume ref is the mask)
mask = ~np.isnan(ref.isel(time=0).values)

def gridwise_perkins_skill_score(a, b, nbins=50):
    pss = np.full(a.shape[1:], np.nan)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            a1 = a[:, i, j]
            b1 = b[:, i, j]
            mask_ = ~np.isnan(a1) & ~np.isnan(b1)
            if np.sum(mask_) > 10:
                try:
                    a_valid = a1[mask_]
                    b_valid = b1[mask_]
                    combined_data = np.concatenate([a_valid, b_valid])
                    bins = np.linspace(np.min(combined_data), np.max(combined_data), nbins + 1)
                    hist_a, _ = np.histogram(a_valid, bins=bins, density=True)
                    hist_b, _ = np.histogram(b_valid, bins=bins, density=True)
                    hist_a = hist_a / np.sum(hist_a)
                    hist_b = hist_b / np.sum(hist_b)
                    pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
                except Exception:
                    pss[i, j] = np.nan
    return pss



pss_cdf = gridwise_perkins_skill_score(cdf.values, ref.values)
pss_dotc = gridwise_perkins_skill_score(dotc.values, ref.values)
pss_eqm = gridwise_perkins_skill_score(eqm.values, ref.values)

# (highest PSS) at each (COARSE) grid cell

winner = np.full(pss_cdf.shape, np.nan)
for i in range(pss_cdf.shape[0]):
    for j in range(pss_cdf.shape[1]):
        if not mask[i, j]:
            continue
        vals = [pss_cdf[i, j], pss_dotc[i, j], pss_eqm[i, j]]
        if np.any(np.isnan(vals)):
            continue
        winner[i, j] = np.argmax(vals)
winner[~mask] = np.nan


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
ax.set_title("Best PSS for Median Daily Temperature (RCP45 Ensemble Median;2011-2023)", fontsize=22, fontweight='bold')
plt.savefig("../outputs/gridwise_pss_winner_tas_climatology_rcp45.png", dpi=1000, bbox_inches='tight')
plt.close()