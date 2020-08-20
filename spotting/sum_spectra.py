import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from pyimzml.ImzMLParser import ImzMLParser

from msms_scoring.datasets import dataset_aliases2


def merge_spectra(spectra):
    all_peaks = np.stack([
        np.concatenate([spectrum[0] for spectrum in spectra]),
        np.concatenate([spectrum[1] for spectrum in spectra]),
    ]).T
    all_peaks = all_peaks[all_peaks[:, 1] > 0, :]
    mzs, ints = all_peaks.T
    order = np.argsort(mzs)
    return mzs[order], ints[order]


def get_ds_spots(ds_id):
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
    spots = {}
    for i, mask_name in enumerate(mask_names):
        if mask_name != 'background':
            spectra_ys, spectra_xs = np.nonzero(grid_mask == i)
            spectra = [parser.getspectrum(idx) for idx in coord_to_idx[spectra_xs, spectra_ys]]
            mzs, ints = merge_spectra(spectra)
            spots[mask_name] = mzs, ints, len(spectra)
    return spots


def get_background_spot_mapping():
    """
    For each spot, find the set of other spots that should have no on-sample chemical overlap,
    so that they can be used to build a baseline of background peaks.
    """
    grid = pd.read_csv('spotting/msms_grid_mapping.csv', index_col=0)
    grid_halves = [grid[grid.row < 10], grid[grid.row >= 10]]
    bg_map = {}
    for mol_set in grid_halves:
        content_map = {}
        for cell, row in mol_set.iterrows():
            content_map[cell] = set([row.hmdb_id1, row.hmdb_id2, row.hmdb_id3]).difference([np.nan])
        for cell, contents in content_map.items():
            bg_map[cell] = [other for other, other_contents in content_map.items() if contents.isdisjoint(other_contents)]

    return bg_map


@njit()
def group_peaks(all_mzs, all_ints, min_peaks=3, max_ppm_width=3.0):
    mz_tol = 1 + max_ppm_width * 1e-6

    finished_groups = []
    unfinished_groups = [(all_mzs, all_ints)]
    while len(unfinished_groups) > 0:
        todo = unfinished_groups
        unfinished_groups = []
        for mzs, ints in todo:
            if len(mzs) < min_peaks:
                continue
            if mzs.max() > mzs.min() * mz_tol:
                biggest_gap = np.argmax(np.diff(mzs)) + 1
                unfinished_groups.append((mzs[:biggest_gap], ints[:biggest_gap]))
                unfinished_groups.append((mzs[biggest_gap:], ints[biggest_gap:]))
            else:
                finished_groups.append((mzs, ints))

    group_mzs = np.array([mzs[0] for mzs, ints in finished_groups])
    order = np.argsort(group_mzs)
    return [finished_groups[idx] for idx in order]


def merge_groups(groups, sample_size):
    out_mzs = np.array([np.average(mzs, weights=ints) for mzs, ints in groups])
    out_ints = np.array([np.sum(ints) / sample_size for mzs, ints in groups])
    return out_mzs, out_ints


def get_spot_bg(bg_map, spot_peaks, spot):
    bg_spectra = [spot_peaks[bg_spot] for bg_spot in bg_map[spot]]
    bg_mzs, bg_ints = merge_spectra(bg_spectra)
    return bg_mzs, bg_ints, len(bg_map[spot])
    # bg_cnt = len(bg_map[spot])
    # bg_mzs, bg_ints = merge_spectra(bg_spectra)
    # return merge_groups(group_peaks(bg_mzs, bg_ints, bg_cnt // 4, 0.5), bg_cnt)


def subtract_bg(mzs, ints, bg_mzs, bg_ints, min_hits, min_int_increase=1.5, max_ppm_dist=0.5):
    peak_l = np.searchsorted(bg_mzs, mzs * (1 - max_ppm_dist * 1e-6), 'l')
    peak_r = np.searchsorted(bg_mzs, mzs * (1 + max_ppm_dist * 1e-6), 'r')
    results = []

    for mz, intensity, l, r in zip(mzs, ints, peak_l, peak_r):
        if r - l > min_hits:
            bg_intensity = np.mean(bg_ints[l:r])
            if intensity > bg_intensity * min_int_increase:
                results.append((mz, intensity))
        else:
            results.append((mz, intensity))

    fg_mzs, fg_ints = np.array(results).T
    return fg_mzs, fg_ints


def batch_run(ds_id):
    ds_name = dataset_aliases2.get(ds_id, ds_id)
    spots = get_ds_spots(ds_id)
    bg_map = get_background_spot_mapping()
    spot_groups = {
        spot: group_peaks(mzs, ints, spot_size // 10) for spot, (mzs, ints, spot_size) in spots.items()
    }
    spot_peaks = {
        spot: merge_groups(spot_groups[spot], spot_size) for spot, (mzs, ints, spot_size) in spots.items()
    }
    for spot in spots.keys():
        peak_mzs, peak_ints = spot_peaks[spot]
        sum_spectrum = np.stack([peak_mzs, peak_ints]).T
        out_path = Path(f'sum_spectra/{ds_name}/raw/{spot}.txt')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, sum_spectrum, fmt='%.5f')

        if spot in bg_map:
            bg_mzs, bg_ints, bg_size = get_spot_bg(bg_map, spot_peaks, spot)
            fg_mzs, fg_ints = subtract_bg(peak_mzs, peak_ints, bg_mzs, bg_ints, bg_size // 4)
            sum_fg_spectrum = np.stack([fg_mzs, fg_ints]).T
            out_path = Path(f'sum_spectra/{ds_name}/cleaned/{spot}.txt')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, sum_fg_spectrum, fmt='%.5f')
