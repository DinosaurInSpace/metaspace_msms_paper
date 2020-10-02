import pandas as pd
from msms_scoring.datasets import dataset_ids, msms_mol_ids
from msms_scoring.metrics import get_ds_results
from msms_scoring.exports import export
from msms_scoring.datasets_full import from_mans
from scipy.spatial.distance import cosine

#%%

def compare_dss(ds_id_a, ds_id_b, hmdb_ids, use_latest_db=None):
    res_a = get_ds_results(ds_id_a, use_latest_db=use_latest_db)
    res_b = get_ds_results(ds_id_b, use_latest_db=use_latest_db)
    results = []
    for hmdb_id in hmdb_ids:
        a = res_a.ann_mols_df[res_a.ann_mols_df.hmdb_id == hmdb_id]
        b = res_b.ann_mols_df[res_b.ann_mols_df.hmdb_id == hmdb_id]
        if a.empty or b.empty: continue
        a_mol = res_a.mols_df.loc[hmdb_id]
        b_mol = res_b.mols_df.loc[hmdb_id]
        all_formulas = list(a.formula)
        a_detected = a.formula[a.is_detected & (a.coloc_to_parent > 0.1)]
        b_detected = b.formula[b.is_detected & (b.coloc_to_parent > 0.1)]
        both = set(a_detected).intersection(b_detected)
        either = set(a_detected).union(b_detected)
        only_a = set(a_detected).difference(b_detected)
        only_b = set(b_detected).difference(a_detected)
        neither = set(all_formulas).difference(a_detected).difference(b_detected)
        a_score = a.coloc_to_parent[a.formula.isin(both) & ~a.is_parent].sum()
        b_score = b.coloc_to_parent[b.formula.isin(both) & ~b.is_parent].sum()

        paired = a.merge(b, on='formula', suffixes=('_a', '_b'))[lambda df: df.formula.isin(both)]
        corr = 1-cosine(paired.intensity_a, paired.intensity_b)

        results.append({
            'a_name': res_a.name,
            'b_name': res_b.name,
            'mol_name': a_mol.mol_name,
            'a_fdr': a_mol.coloc_int_fdr,
            'b_fdr': b_mol.coloc_int_fdr,
            'a_common_sum_coloc': a_score,
            'b_common_sum_coloc': b_score,
            'intensity_corr': corr,
            'goodness': a_score * b_score * corr * (len(both) / len(either)),
            'in both': len(both),
            'in only A': len(only_a),
            'in only B': len(only_b),
            'in neither': len(neither),
            'a_both_href': f'https://beta.metaspace2020.eu/annotations?ds={res_a.ds_id}&db_id={res_a.db_id}&sort=-mz&fdr=1&q=' + '|'.join(both),
            'b_both_href': f'https://beta.metaspace2020.eu/annotations?ds={res_b.ds_id}&db_id={res_b.db_id}&sort=-mz&fdr=1&q=' + '|'.join(both),
            'a_either_href': f'https://beta.metaspace2020.eu/annotations?ds={res_a.ds_id}&db_id={res_a.db_id}&sort=-mz&fdr=1&q=' + '|'.join(either),
            'b_either_href': f'https://beta.metaspace2020.eu/annotations?ds={res_b.ds_id}&db_id={res_b.db_id}&sort=-mz&fdr=1&q=' + '|'.join(either),
            'ab_both_href': f'https://beta.metaspace2020.eu/annotations?ds={res_a.ds_id}|{res_b.ds_id}&db_id={res_a.db_id}&sort=-mz&fdr=1&q=' + '|'.join(either),
            'ab_either_href': f'https://beta.metaspace2020.eu/annotations?ds={res_a.ds_id}|{res_b.ds_id}&db_id={res_a.db_id}&sort=-mz&fdr=1&q=' + '|'.join(either),
        })

    return pd.DataFrame(results)


common_df = pd.concat([
    compare_dss(ds_a, ds_b, msms_mol_ids, ds_b in from_mans)
    for ds_a in dataset_ids[:2]
    for ds_b in [*from_mans, '2020-05-26_17h58m22s', '2020-05-26_17h58m19s', '2020-05-26_17h57m57s', '2020-05-26_17h57m50s']
]).sort_values('goodness', ascending=False)[lambda df: df.goodness > 0]
export({
    'Common': (common_df, {'index': False}),
}, 'scoring_results/common_mols_with_lone_isotopes v2.xlsx', grouped_sheets=False)
