from functools import lru_cache
import numpy as np
import pandas as pd

from msms_scoring.datasets import dataset_mol_lists
from msms_scoring.fetch_data import DSResults, get_msms_results_for_ds


class DSResults2:
    ds_id: str
    name: str
    ds_coloc: pd.DataFrame
    msms_df: pd.DataFrame
    mols_df: pd.DataFrame
    anns_df: pd.DataFrame

    def get_coloc(self, f1, f2):
        if f1 == f2:
            return 1
        if f1 not in self.ds_coloc.index or f2 not in self.ds_coloc.index:
            return 0
        return self.ds_coloc.loc[f1, f2]


def combine_mols(df):
    df = df.sort_values('hmdb_id')
    row = df.iloc[0]
    return pd.Series({
        'hmdb_id': ','.join(df.hmdb_id),
        'mz': row.mz,
        'is_detected': df.is_detected.any(),
        'is_expected': df.is_expected.any(),
        'off_sample': row.off_sample,
        'mol_name': ','.join(df.mol_name),
        'intensity': row.intensity,
        'parent_formula': row.formula,
        'frag_formulas': sorted(f for f in row.all_frag_formulas.split(',') if f != row.formula),
        'all_frag_formulas': sorted(f for f in row.all_frag_formulas.split(',')),
        'parent_n_frags': row.parent_n_frags,
        'mols_in_group': len(df),
        'expected_mols_in_group': df.is_expected.sum(),
        'unexpected_mols_in_group': (~df.is_expected).sum(),
        'mol_href': row.mol_href,
    })


def get_msms_results_for_ds2(ds_id, include_off_sample=False):
    print(f'get_msms_results_for_ds2({ds_id})')
    ds = get_msms_results_for_ds(ds_id)
    ds2 = DSResults2()
    ds2.ds_id = ds.ds_id
    ds2.name = ds.name
    ds2.ds_coloc = ds.ds_coloc
    expected_mol_ids = dataset_mol_lists.get(ds.ds_id, set())
    ds2.mols_df = ds.mols_df
    ds2.msms_df = (
        ds.mols_df
        .assign(is_expected=ds.mols_df.index.isin(expected_mol_ids))
        .reset_index()
        .groupby('all_frag_formulas')
        .apply(combine_mols)
        .set_index('hmdb_id', drop=False)
        [lambda df: ~df.off_sample | include_off_sample]
    )
    ds2.anns_df = ds.anns_df
    return ds2


def get_fdr(decoy_scores, target_scores, rule_of_succession=True):
    decoys_df = pd.DataFrame({'id': None, 'score': decoy_scores, 'decoy_cnt': 1, 'target_cnt': 0})
    targets_df = pd.DataFrame({'id': target_scores.index, 'score': target_scores.values, 'decoy_cnt': 0, 'target_cnt': 1})
    fdr_df = pd.concat([decoys_df, targets_df], ignore_index=True).sort_values('score', ascending=False)
    fdr_df['decoy_cnt'] = np.cumsum(fdr_df.decoy_cnt)
    fdr_df['target_cnt'] = np.cumsum(fdr_df.target_cnt)
    # METASPACE-style FDR

    # "Rule-of-succession" FDR (conservative - won't claim 0% FDR when decoys are sparse)
    if rule_of_succession:
        bias = (len(decoys_df) + 1) / (len(targets_df) + 1)
        fdr_df['fdr_raw'] = (fdr_df.decoy_cnt + 1) / (fdr_df.target_cnt + 1) / bias
    else:
        bias = len(decoys_df) / len(targets_df)
        fdr_df['fdr_raw'] = fdr_df.decoy_cnt / np.clip(fdr_df.target_cnt, 1, None) / bias
    fdr_df.sort_values('score', ascending=False, inplace=True)
    fdr_df['fdr'] = np.minimum.accumulate(fdr_df.fdr_raw.iloc[::-1])[::-1]
    fdr_df.loc[fdr_df.score == 0, 'fdr'] = 1

    target_fdrs = fdr_df[~fdr_df.id.isna()].set_index('id')[['fdr']]

    # Add "fold change", calculated as "target score / average decoy score"
    avg_decoy_score = np.mean(decoy_scores)
    if avg_decoy_score != 0:
        target_fdrs['fold_change'] = target_scores / avg_decoy_score
    else:
        target_fdrs['fold_change'] = 1

    return target_fdrs


def add_fdr(res: DSResults2):
    def coloc_int_score(parent, frags):
        ints = res.anns_df.intensity
        max_int = ints[parent] if parent in ints else 0
        return sum(res.get_coloc(parent, f) for f in frags if f in ints.index and max_int > ints[f])

    def get_decoy_coloc_int_scores(n_decoys, max_n_frags):
        result = np.zeros((max_n_frags+1, n_decoys), 'f')
        parents = np.random.choice(valid_parent_formulas, n_decoys)
        ints = res.anns_df.intensity
        for i, parent in enumerate(parents):
            max_int = res.anns_df.intensity.get(parent, 0)
            if max_int > 0 and max_n_frags > 1:
                frags = np.random.choice(valid_frag_formulas, max_n_frags - 1, replace=False)
                result[2:, i] = np.cumsum([res.get_coloc(parent, f) if f in ints.index and max_int > ints[f] else 0 for f in frags])

        return result

    valid_frag_formulas = np.unique([f for fs in res.msms_df.all_frag_formulas for f in fs])
    valid_parent_formulas = res.anns_df[res.anns_df.index.isin(valid_frag_formulas) & ~res.anns_df.off_sample].index.unique()
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


def average_precision(s):
    return np.sum((np.cumsum(s) / (np.arange(len(s)) + 1)) * s) / np.count_nonzero(s)


def add_metric_scores2(res: DSResults2):
    scores = []
    df = res.msms_df.sort_values('is_expected')

    for m in ['random', 'coloc_int_fdr']:
        avg_prec = None
        if m in res.msms_df.columns:
            ranking = df.sort_values(m).is_expected.values
            avg_prec = average_precision(ranking)
        elif m == 'random':
            avg_prec = np.mean([average_precision(df.is_expected.sample(frac=1).values) for i in range(10000)])

        if avg_prec is not None:
            scores.append({
                'metric': m,
                'avg_prec': avg_prec,
            })
    res.metric_scores = pd.DataFrame(scores)
    res.metric_counts = {
        'n_expected': df.is_expected.sum(),
        'n_unexpected': (~df.is_expected).sum(),
        # 'n_expected_mols': df[df.is_expected].mols_in_group.sum(),
        # 'n_unexpected_mols': df[~df.is_expected].mols_in_group.sum(),
    }


# @lru_cache(maxsize=None)
def get_ds_results2(ds_id):
    res = get_msms_results_for_ds2(ds_id)
    add_fdr(res)
    add_metric_scores2(res)
    return res

