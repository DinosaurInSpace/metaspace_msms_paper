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
            norm_spectra = [(mzs, ints * 1e6 / np.sum(ints)) for mzs, ints in spectra]
            mzs, ints = merge_spectra(norm_spectra)
            spots[mask_name] = mzs, ints, len(norm_spectra)
    return spots


def get_background_spot_mapping(grid):
    """
    For each spot, find the set of other spots that should have no on-sample chemical overlap,
    so that they can be used to build a baseline of background peaks.
    """

    bg_map = {}
    content_map = {}
    for spot, row in grid.iterrows():
        content_map[spot] = set([row.hmdb_id1, row.hmdb_id2, row.hmdb_id3]).difference([np.nan])
    for spot, contents in content_map.items():
        bg_map[spot] = [other for other, other_contents in content_map.items() if contents.isdisjoint(other_contents)]

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


def batch_run(ds_id, remove_background_signal=False):
    """remove_background_signal needs further development if we decide to use it"""
    ds_name = dataset_aliases2.get(ds_id, ds_id)
    spots = get_ds_spots(ds_id)
    grid = pd.read_csv('spotting/msms_grid_mapping.csv', index_col=0)
    if remove_background_signal:
        bg_map = {
            # Split into upper and lower grids, as the replicates seem to sometimes have intensity differences
            **get_background_spot_mapping(grid[grid.row < 10]),
            **get_background_spot_mapping(grid[grid.row >= 10]),
        }
    else:
        bg_map = {}
    spot_groups = {
        spot: group_peaks(mzs, ints, spot_size // 10) for spot, (mzs, ints, spot_size) in spots.items()
    }
    spot_peaks = {
        spot: merge_groups(spot_groups[spot], spot_size) for spot, (mzs, ints, spot_size) in spots.items()
    }
    spot_names = {
        spot: '_'.join(m for m in [r.name1, r.name2, r.name3] if pd.notna(m))
        for spot, r in grid.iterrows()
    }
    for spot in spots.keys():
        spot_name = spot_names.get(spot, 'Empty')
        peak_mzs, peak_ints = spot_peaks[spot]
        sum_spectrum = np.stack([peak_mzs, peak_ints]).T
        out_path = Path(f'sum_spectra/{ds_id}/raw/{spot}.txt')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, sum_spectrum, fmt='%.5f')

        if remove_background_signal and spot in bg_map:
            bg_mzs, bg_ints, bg_size = get_spot_bg(bg_map, spot_peaks, spot)
            fg_mzs, fg_ints = subtract_bg(peak_mzs, peak_ints, bg_mzs, bg_ints, bg_size // 4)
            sum_fg_spectrum = np.stack([fg_mzs, fg_ints]).T
            out_path = Path(f'sum_spectra/{ds_id}/cleaned/{spot}.txt')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, sum_fg_spectrum, fmt='%.5f')
