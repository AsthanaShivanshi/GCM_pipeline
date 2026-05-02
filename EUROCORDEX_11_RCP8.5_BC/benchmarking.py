import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

from tqdm import tqdm



methods = ["CDFT", "dOTC", "EQM"]
results = []



start = np.datetime64('2011-01-01')

end = np.datetime64('2023-12-31')


def bivariate_PSS_gridwise(pr_a, tas_a, pr_b, tas_b, nbins=20):
    lat = min(pr_a.shape[1], pr_b.shape[1])
    lon = min(pr_a.shape[2], pr_b.shape[2])
    pss = np.full((lat, lon), np.nan)
    for i in range(lat):
        for j in range(lon):
            a1 = np.stack([pr_a[:, i, j], tas_a[:, i, j]], axis=1)
            b1 = np.stack([pr_b[:, i, j], tas_b[:, i, j]], axis=1)
            mask_ = ~np.isnan(a1).any(axis=1) & ~np.isnan(b1).any(axis=1)
            if np.sum(mask_) > 10:
                a_valid = a1[mask_]
                b_valid = b1[mask_]
                pr_min = min(a_valid[:,0].min(), b_valid[:,0].min())
                pr_max = max(a_valid[:,0].max(), b_valid[:,0].max())
                tas_min = min(a_valid[:,1].min(), b_valid[:,1].min())
                tas_max = max(a_valid[:,1].max(), b_valid[:,1].max())
                bins_pr = np.linspace(pr_min, pr_max, nbins+1)
                bins_tas = np.linspace(tas_min, tas_max, nbins+1)
                hist_a, _, _ = np.histogram2d(a_valid[:,0], a_valid[:,1], bins=[bins_pr, bins_tas], density=True)
                hist_b, _, _ = np.histogram2d(b_valid[:,0], b_valid[:,1], bins=[bins_pr, bins_tas], density=True)
                hist_a = hist_a / np.sum(hist_a)
                hist_b = hist_b / np.sum(hist_b)
                pss[i, j] = np.sum(np.minimum(hist_a, hist_b))
    return pss





ref_pr_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/RhiresD_1971_2023.nc"


ref_tas_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full/TabsD_1971_2023.nc"





ref_pr = xr.open_dataset(ref_pr_path).sel(time=slice(start, end))['RhiresD']
ref_tas = xr.open_dataset(ref_tas_path).sel(time=slice(start, end))['TabsD']


mask = ~np.isnan(ref_pr.isel(time=0).values)




pr_files = sorted(glob.glob("CDFT/pr_bicubic_EUR-11_*_rcp85_1971-2099.nc"))


members = [os.path.basename(f).replace("pr_bicubic_EUR-11_", "").replace("_rcp85_1971-2099.nc", "") for f in pr_files]


for member in tqdm(members):

    
    row = {"member": member}
    for method in methods:
        pr_path = f"{method}/pr_bicubic_EUR-11_{member}_rcp85_1971-2099.nc"
        tas_path = f"{method}/tas_bicubic_EUR-11_{member}_rcp85_1971-2099.nc"
        if not (os.path.exists(pr_path) and os.path.exists(tas_path)):
            row[method] = np.nan
            continue

        pr = xr.open_dataset(pr_path).sel(time=slice(start, end))['pr']
        tas = xr.open_dataset(tas_path).sel(time=slice(start, end))['tas']

        #intersection only ... ..,,


        pr_aligned, ref_pr_aligned = xr.align(pr, ref_pr, join='inner')
        tas_aligned, ref_tas_aligned = xr.align(tas, ref_tas, join='inner')

        pss = bivariate_PSS_gridwise(
            pr_aligned.values, tas_aligned.values,
            ref_pr_aligned.values, ref_tas_aligned.values
        )
        pss_masked = np.where(mask, pss, np.nan)
        mean_pss = np.nanmean(pss_masked)
        row[method] = mean_pss

    results.append(row)


df = pd.DataFrame(results)


df.to_csv("outputs/bivariate_pss_table_by_member.csv", index=False)


print(df)
