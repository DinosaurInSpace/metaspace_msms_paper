import json
from pathlib import Path

import numpy as np
import pandas as pd
from metaspace import SMInstance
from pyimzml.ImzMLParser import ImzMLParser

from msms_scoring.datasets import dataset_ids
#%%
sm = SMInstance()
#%%
for ds_id in dataset_ids:
    print(f'Downloading {ds_id}')
    sm.dataset(id=ds_id).download_to_dir('raw_datasets', ds_id)



#%%

def recentroid_peaks(all_peaks, sample_size, bg_peaks=None, bg_size=0, min_frac=0.1, max_ppm_width=3):
    def finish_group(group):
        if len(group) >= min_cnt:
            mzs, ints = np.array(group).T
            ppm_diff = (mzs.max() / mzs.min() - 1) * 1e6
            if ppm_diff > max_ppm_width:
                # If group is too wide, split it at the biggest inter-peak gap, hopefully preserving "clumps" of related peaks
                biggest_gap = np.argmax(mzs[1:] - mzs[:-1]) + 1
                finish_group(group[:biggest_gap])
                finish_group(group[biggest_gap:])
            else:
                centroids.append([
                    np.average(mzs, weights=ints),
                    np.sum(ints) / sample_size,
                    mzs.min(),
                    mzs.max(),
                    mzs,
                    ints,
                ])

    min_cnt = max(int(min_frac * len(spectra)), 1)
    mz_tol = 1 + max_ppm_width * 1e-6
    centroids = []
    group = []
    limit_mz = 0
    for mz, intensity in all_peaks:
        if mz > limit_mz:
            finish_group(group)
            group.clear()

        limit_mz = mz * mz_tol
        group.append((mz, intensity))

    finish_group(group)



    return np.array(centroids)

def merge_spectra(spectra):
    all_peaks = np.stack([
        np.concatenate([spectrum[0] for spectrum in spectra]),
        np.concatenate([spectrum[1] for spectrum in spectra]),
    ]).T
    all_peaks = all_peaks[all_peaks[:, 1] > 0, :]
    all_peaks = all_peaks[np.argsort(all_peaks[:, 0]), :]
    return all_peaks

# Find optimal ppm
# for ppm in np.arange(1.5, 5, 0.25):
#     groups = merge_spectra(spectra, 0.1, ppm)
#     print(ppm, [np.max(groups[groups[:, 0] > mz, 3]) for mz in [100, 300, 500, 700]])

%time groups = recentroid_peaks(merge_spectra(spectra), len(spectra))
# print(len(groups))
# fat_groups = groups[groups[:, 3] > 3, :]

#%%
# grid_mapping = pd.read_csv('spotting/msms_grid_mapping.csv')

for ds_id in dataset_ids:
    parser = ImzMLParser(f'raw_datasets/{ds_id}.imzML')
    grid_mask = np.load(f'spotting/grids/{ds_id}.npy')
    mask_names = json.load(open(f'spotting/grids/{ds_id}_mask_names.json'))
    Path(f'sum_spectra/{ds_id}/').mkdir(parents=True, exist_ok=True)

    # Make a mapping of coordinate -> spectrum index
    coords = np.array(parser.coordinates)[:, :2]
    base_coord = np.min(coords, axis=0)
    coord_to_idx = np.ones(np.max(coords, axis=0) - base_coord + 1, dtype='i') * -1
    for i, (x, y) in enumerate(coords):
        coord_to_idx[x - base_coord[0], y - base_coord[1]] = i

    # Collect spectra for each mask item
    for i, mask_name in enumerate(mask_names[1:], 1):
        spectra_ys, spectra_xs = np.nonzero(grid_mask == i)
        spectra = [parser.getspectrum(idx) for idx in coord_to_idx[spectra_xs, spectra_ys]]
        sum_spectrum = recentroid_peaks(merge_spectra(spectra), len(spectra))
        np.savetxt(f'sum_spectra/{ds_id}/{mask_name}.txt', sum_spectrum, fmt='%.5f')


#%%


#%%