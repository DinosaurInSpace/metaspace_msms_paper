import pickle
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

from msms_scoring.fetch_data import get_msms_results_for_ds, DSResults
from msms_scoring.metrics import get_fdr, average_precision


def add_fdr2(res: DSResults, include_off_sample=False):
    def coloc_int_score(parent, frags):
        ints = res.anns_df.intensity
        if parent in ints and (include_off_sample or parent not in off_sample_parents):
            max_int = ints[parent]
            return sum(res.get_coloc(parent, f) for f in frags if f in ints.index and max_int > ints[f])
        else:
            return 0

    def get_decoy_coloc_int_scores(n_decoys, max_n_frags):
        result = np.zeros((max_n_frags+1, n_decoys), 'f')
        parents = np.random.choice(valid_parent_formulas, n_decoys)
        ints = res.anns_df.intensity
        for i, parent in enumerate(parents):
            max_int = res.anns_df.intensity.get(parent, 0)
            if max_int > 0 and max_n_frags > 1 and (include_off_sample or parent not in off_sample_parents):
                frags = np.random.choice(valid_frag_formulas, max_n_frags - 1, replace=False)
                result[2:, i] = np.cumsum([res.get_coloc(parent, f) if f in ints.index and max_int > ints[f] else 0 for f in frags])

        return result

    valid_frag_formulas = np.unique([f for fs in res.msms_df.all_frag_formulas for f in fs])
    valid_parent_formulas = res.anns_df[res.anns_df.index.isin(valid_frag_formulas)].index.unique()
    off_sample_parents = set(res.anns_df[res.anns_df.index.isin(valid_frag_formulas) & res.anns_df.off_sample].index)
    all_decoy_scores = get_decoy_coloc_int_scores(n_decoys=1000, max_n_frags=res.msms_df.parent_n_frags.max())
    alg_scores = []
    alg_results = []
    for n_frags, grp in res.msms_df.groupby('parent_n_frags'):
        decoy_scores = all_decoy_scores[n_frags, :]
        target_scores = grp.apply(lambda row: coloc_int_score(row.parent_formula, row.frag_formulas), axis=1)

        alg_scores.append(target_scores)
        alg_results.append(get_fdr(decoy_scores, target_scores))
    all_alg_results = pd.concat(alg_results)

    fdrs_df = pd.DataFrame({
        'coloc_int': pd.concat(alg_scores),
        'coloc_int_fdr': all_alg_results.fdr,
        'coloc_int_fc': all_alg_results.fold_change,
    })
    res.msms_df.drop(columns=fdrs_df.columns, errors='ignore', inplace=True)
    res.msms_df = res.msms_df.join(fdrs_df, how='left')


def add_metric_scores2(res: DSResults, include_off_sample=False):
    df = res.msms_df.sort_values('is_expected')
    if not include_off_sample:
        df = df[~df.off_sample]

    res.metric_scores = pd.DataFrame([
        {
            'metric': 'random',
            'avg_prec': np.mean([average_precision(df.is_expected.sample(frac=1).values) for i in range(10000)]),
        },
        {
            'metric': 'coloc_int_fdr',
            'avg_prec': average_precision(df.sort_values('coloc_int_fdr').is_expected.values),
        },
    ])


def get_ds_results2(ds_id, mz_range=None, db_ver=None, include_lone_isotopic_peaks=False, include_off_sample=False, use_cache=True):
    cache_name_parts = [
        ds_id,
        mz_range and f'{mz_range[0]}-{mz_range[1]}',
        db_ver,
        include_lone_isotopic_peaks and '1iso',
        include_off_sample and 'offs',
    ]
    cache_name = '_'.join(part for part in cache_name_parts if part)
    cache_path = Path(f'./scoring_results/cache/ds_result_metrics2/{cache_name}.pickle')
    if not use_cache or not cache_path.exists():
        print(f'get_ds_results2({ds_id})')
        res = get_msms_results_for_ds(
            ds_id,
            mz_range=mz_range,
            db_ver=db_ver,
            include_lone_isotopic_peaks=include_lone_isotopic_peaks,
            use_cache=use_cache
        )
        add_fdr2(res, include_off_sample=include_off_sample)
        add_metric_scores2(res, include_off_sample=include_off_sample)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, cache_path.open('wb'))

    res = pickle.load(cache_path.open('rb'))
    return res


def _get_ds_results2_kwargs(ds_id, kwargs):
    return get_ds_results2(ds_id, **kwargs)


def get_many_ds_results2(ds_ids, **kwargs):
    with ProcessPoolExecutor() as p:
        return list(p.map(_get_ds_results2_kwargs, ds_ids, repeat(kwargs)))
