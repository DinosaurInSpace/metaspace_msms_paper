# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import TfidfTransformer
from enrichmentanalysis.enrich_run import EnrichmentRun

from msms_scoring.datasets import dataset_mol_lists
from msms_scoring.fetch_data import DSResults, get_msms_results_for_ds


#%%
# Wider maximum width of pandas columns (needed to see the full lists of molecules)

pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
#%%
def add_expected_mols(res: DSResults):
    expected_mol_ids = dataset_mol_lists.get(res.ds_id, set())
    res.mols_df['is_expected'] = res.mols_df.index.isin(expected_mol_ids)
    res.ann_mols_df['is_expected'] = res.ann_mols_df.hmdb_id.isin(expected_mol_ids)

    expected_formulas = set(res.mols_df[res.mols_df.is_expected].formula)
    res.mols_df['is_expected_isomer'] = res.mols_df.formula.isin(expected_formulas) & ~res.mols_df.is_expected
    res.ann_mols_df['is_expected_isomer'] = res.ann_mols_df.parent_formula.isin(expected_formulas) & ~res.ann_mols_df.is_expected

    is_redundant = (
        res.mols_df
        .sort_values(['is_expected'], ascending=False)
        .duplicated('all_frag_formulas')
        .rename('is_redundant')
    )
    res.mols_df['is_redundant'] = is_redundant
    res.ann_mols_df = res.ann_mols_df.merge(is_redundant, how='left', left_on='hmdb_id', right_index=True)


if __name__ == '__main__':
    test_results = get_msms_results_for_ds('2020-06-19_16h39m10s')
    add_expected_mols(test_results)
    df = test_results.mols_df
#%%

def add_tfidf_score(res: DSResults):
    """
    TF-IDF where:
    * each parent molecule is a "document"
    * each annotated formula is a "term"
    * a term is only in a document when it is a parent ion or a predicted fragment ion
    * a term's "frequency" in a document is (colocalization to parent) / (number of predicted ions for this parent ion)

    Caveats:
    * The output value is just summed per document to produce a score. May not be optimal.
    """
    terms_df = (
        res.ann_mols_df
        # [lambda df: ~df.is_parent]
        # To use constant value:
        # .assign(value=1)
        # To use value based on 1/num_features:
        # Note that this equally scales the parent annotation
        .assign(value=res.ann_mols_df.coloc_to_parent / res.ann_mols_df.parent_n_frags)
        .pivot_table(index='hmdb_id', columns='formula', values='value', fill_value=0, aggfunc='sum')
    )
    # terms_df /= np.array([features_per_parent_s.reindex_like(terms_df).values]).T
    terms_matrix = csr_matrix(terms_df.values)

    tfidf_raw = TfidfTransformer().fit_transform(terms_matrix)
    tfidf_s = pd.Series(
        tfidf_raw.toarray().sum(axis=1),
        index=terms_df.index,
        name='tfidf',
    )
    res.mols_df['inv_tfidf'] = 1 / tfidf_s
    res.ann_mols_df = res.ann_mols_df.merge(tfidf_s, left_on='hmdb_id', right_index=True)


if __name__ == '__main__':
    add_tfidf_score(test_results)
#%%

def add_enrichment_analysis(res: DSResults):
    """
    Over-representation analysis where:
    * The "population" is all predicted ion formulas
    * The "study set" is all observed ion formulas
    * "Associations" are groups linking each ion formula to potential parent molecules

    Caveats:
    * Colocalization is not considered, even though it can be a compelling way to disprove fragments
      association with their parents
    * A bad P-value is not evidence that the candidate molecule is wrong. It only indicates the
      possibility that the observed fragment distribution was due to random chance.
    """

    def enrich(df, prefix=''):
        population = set(df.formula.unique())
        study_ids = set(df[df.is_detected].formula.unique())
        associations = df.groupby('formula').apply(lambda grp: set(grp.hmdb_id)).to_dict()

        enrichment_results = (
            EnrichmentRun(population, associations, alpha=0.05, methods=('sm_bonferroni',))
            .run_study(study_ids, study_name='results')
            .results
        )

        # HACK: r.get_nt_prt() converts lots of fields to strings, so manually grab all the interesting fields
        raw_data = pd.DataFrame([{
            'hmdb_id': r.termid,
            **r.ntpval._asdict(),
            **r.multitests._asdict(),
            'stu_items': r.stu_items
        } for r in enrichment_results if r.ntpval.study_cnt]).set_index('hmdb_id')

        return pd.DataFrame({
            # prefix + 'enrich_ratio': raw_data.study_ratio / raw_data.pop_ratio,
            prefix + 'enrich_p': raw_data.sm_bonferroni,
            prefix + 'enrich_uncorr': raw_data.pval_uncorr,
        }, index=raw_data.index, dtype='f')

    enrich_data = enrich(res.ann_mols_df, 'global_')
    res.mols_df = res.mols_df.join(enrich_data, how='left')
    res.ann_mols_df = res.ann_mols_df.merge(enrich_data, how='left', left_on='hmdb_id', right_index=True)

    mini_df = res.ann_mols_df[['hmdb_id', 'formula', 'is_detected']]
    detected_formulas = set(mini_df[mini_df.is_detected].formula)
    group_enrich_datas = []
    for formula, formula_df in mini_df[mini_df.formula.isin(detected_formulas)].groupby('formula'):
        related_df = pd.concat([formula_df, mini_df[mini_df.hmdb_id.isin(formula_df.hmdb_id)]])
        related_df['coloc'] = [res.get_coloc(formula, f) for f in related_df.formula]
        related_df['is_detected'] = related_df.is_detected & (related_df.coloc > 0.55)
        group_enrich_data = enrich(related_df, 'group_').assign(formula=formula).set_index('formula', append=True)
        group_enrich_datas.append(group_enrich_data)

    res.ann_mols_df = res.ann_mols_df.merge(
        pd.concat(group_enrich_datas),
        how='left',
        left_on=['hmdb_id', 'formula'],
        right_index=True
    )


if __name__ == '__main__':
    add_enrichment_analysis(test_results)
#%%
def calc_pvalue(study_count, study_n, pop_count, pop_n):
    """Calculate uncorrected p-values."""
    # http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.stats.fisher_exact.html
    #
    #         Atlantic  Indian                              YES       NO
    # whales     8        2    | 10 whales    study_genes    8 scnt   2    | 10 = study_n
    # sharks     1        5    |  6 sharks    not s_genes    1        5    |  6
    #         --------  ------                            --------   -----
    #            9        7      16 = pop_n     pop_genes    9 pcnt   7      16 = pop_n
    #
    # We use the preceeding table to find the p-value for whales/sharks:
    #
    # >>> import scipy.stats as stats
    # >>> oddsratio, pvalue = stats.fisher_exact([[8, 2], [1, 5]])
    #                                              a  b    c  d
    avar = study_count
    bvar = study_n - study_count
    cvar = pop_count - study_count
    dvar = pop_n - pop_count - bvar
    assert avar >= 0 and bvar >= 0 and cvar >= 0 and dvar >= 0, (avar, bvar, cvar, dvar)
    _, p_uncorrected = fisher_exact([[avar, bvar], [cvar, dvar]], alternative='greater')
    return p_uncorrected

def add_p_values(res: DSResults):
    df = res.ann_mols_df
    pop_n = df.formula.nunique()
    p_values = {}
    for cos_threshold in [20,50,80]:
        is_in_grp = df.is_detected & (df.coloc_to_parent >= (cos_threshold / 100))
        pop_count = df[is_in_grp].formula.nunique()
        grps = is_in_grp.groupby(df.hmdb_id)
        p_values[f'p_value_{cos_threshold}'] = grps.apply(lambda grp: calc_pvalue(np.count_nonzero(grp), len(grp), pop_count, pop_n))

    pvals_df = pd.DataFrame(p_values)
    res.ann_mols_df.drop(columns=pvals_df.columns, errors='ignore', inplace=True)
    res.mols_df.drop(columns=pvals_df.columns, errors='ignore', inplace=True)
    res.ann_mols_df = res.ann_mols_df.merge(pvals_df, how='left', left_on='hmdb_id', right_index=True)
    res.mols_df = res.mols_df.merge(pvals_df, how='left', left_on='hmdb_id', right_index=True)

if __name__ == '__main__':
    add_p_values(test_results)
    df = test_results.mols_df.sort_values(['is_expected', 'p_value_80'])
#%%

def get_fdr(decoy_scores, target_scores, rule_of_succession=True):
    decoys_df = pd.DataFrame({'id': None, 'score': decoy_scores, 'decoy_cnt': 1, 'target_cnt': 0})
    targets_df = pd.DataFrame({'id': target_scores.index, 'score': target_scores.values, 'decoy_cnt': 0, 'target_cnt': 1})
    fdr_df = pd.concat([decoys_df, targets_df], ignore_index=True).sort_values('score', ascending=False)
    fdr_df['decoy_cnt'] = np.cumsum(fdr_df.decoy_cnt)
    fdr_df['target_cnt'] = np.cumsum(fdr_df.target_cnt)
    # METASPACE-style FDR

    # fdr_df.sort_values('fdr', inplace=True)
    # fdr_df['fdr_mono'] = np.minimum.accumulate(fdr_df.fdr.iloc[::-1])[::-1]
    # "Rule-of-succession" FDR (conservative - won't claim 0% FDR when decoys are sparse)
    if rule_of_succession:
        bias = (len(decoys_df) + 1) / (len(targets_df) + 1)
        fdr_df['fdr_raw'] = (fdr_df.decoy_cnt + 1) / (fdr_df.target_cnt + 1) / bias
    else:
        bias = len(decoys_df) / len(targets_df)
        fdr_df['fdr_raw'] = fdr_df.decoy_cnt / np.clip(fdr_df.target_cnt, 1, None) / bias
    fdr_df.sort_values('score', ascending=False, inplace=True)
    fdr_df['fdr'] = np.minimum.accumulate(fdr_df.fdr_raw.iloc[::-1])[::-1]
    return fdr_df[~fdr_df.id.isna()].set_index('id').fdr

def add_fdr(res: DSResults):
    global results
    def coloc_int_score(parent, frags):
        ints = res.anns_df.intensity
        max_int = ints[parent] if parent in ints else 0
        return sum(res.get_coloc(parent, f) for f in frags if f in ints.index and max_int > ints[f])
    def coloc_score(parent, frags):
        return sum(res.get_coloc(parent, f) for f in frags)
    def msm_coloc_score(parent, frags):
        return coloc_score(parent, frags) + res.anns_df.msm[frags].sum()
    def msm_x_coloc_score(parent, frags):
        return sum(res.get_coloc(parent, f) * res.anns_df.msm[f] for f in frags if f in res.anns_df.index)

    def get_decoy(alg_func, n_frags):
        parent = np.random.choice(res.anns_df.index)
        frags = np.random.choice(all_formulas, n_frags-1, replace=False) if n_frags > 1 else []
        return alg_func(parent, frags)

    def get_old_decoy(alg_func, n_frags):
        parent, *frags = np.random.choice(all_formulas, n_frags, replace=False)
        return alg_func(parent, frags)

    all_formulas = np.array(res.ann_mols_df.formula.unique())
    n_decoys = 1000
    results = {}
    # ('msm_coloc', msm_coloc_score), ('msm_x_coloc', msm_x_coloc_score)
    for alg, alg_func in [('coloc', coloc_score), ('coloc_int', coloc_int_score)]:
        alg_scores = []
        alg_results = []
        old_results = []
        for n_frags, grp in res.ann_mols_df.groupby('parent_n_frags'):
            decoy_scores = [get_decoy(alg_func, n_frags) for n in range(n_decoys)]
            old_decoy_scores = [get_old_decoy(alg_func, n_frags) for n in range(n_decoys)]
            target_scores = grp.sort_values('is_parent', ascending=False).groupby('hmdb_id').formula.apply(lambda fs: alg_func(fs.values[0], fs.values[1:]))

            alg_scores.append(target_scores)
            alg_results.append(get_fdr(decoy_scores, target_scores))
            old_results.append(get_fdr(old_decoy_scores, target_scores))
            # print(f'{alg} {n_frags}: {len(grp)}, {np.mean(target_scores)}, {np.mean(decoy_scores)}')
        results[alg] = pd.concat(alg_scores)
        results[f'{alg}_fdr'] = pd.concat(alg_results)
        results[f'old_{alg}_fdr'] = pd.concat(old_results)

    fdrs_df = pd.DataFrame(results)
    res.ann_mols_df.drop(columns=fdrs_df.columns, errors='ignore', inplace=True)
    res.ann_mols_df = res.ann_mols_df.merge(fdrs_df, how='left', left_on='hmdb_id', right_index=True)
    res.mols_df.drop(columns=fdrs_df.columns, errors='ignore', inplace=True)
    res.mols_df = res.mols_df.join(fdrs_df, how='left')

if __name__ == '__main__':
    add_fdr(test_results)
    df = test_results.mols_df.sort_values(['is_expected', 'coloc_fdr']).drop(columns=['parent_n_frags_unfiltered','ann_href','mol_href'], errors='ignore')
# %%

def average_precision(s):
    return np.sum((np.cumsum(s) / (np.arange(len(s)) + 1)) * s) / np.count_nonzero(s)


def add_metric_scores(res: DSResults, params: str = 'unfiltered', min_mz=None):
    scores = []
    df = res.mols_df.sort_values('is_expected')

    assert params in ("unfiltered", "no_off_sample", "no_zero_coloc", "no_structural_analogues", "all_filters")
    if min_mz is not None:
        df = df[df.mz > min_mz]
    if params == 'all_filters' or params == 'no_off_sample':
        df = df[~df.off_sample.astype(np.bool)]
    if params == 'all_filters' or params == 'no_zero_coloc':
        if min_mz is not None:
            zero_coloc = (
                res.ann_mols_df[lambda df: df.is_detected].groupby(['hmdb_id', 'parent_formula'])
                .apply(lambda grp: any(res.get_coloc(row.formula, grp.name[1]) > 0 for id, row in grp.iterrows() if row.formula != grp.name[1] and row.mz >= min_mz))
                .reset_index(level='parent_formula', drop=True)
            )
            df = df[zero_coloc[df.index] > 0]
        else:
            df = df[df.coloc > 0]

    if params == 'all_filters' or params == 'no_structural_analogues':
        df = df[~df.is_redundant]

    for m in ['random', 'inv_tfidf', 'global_enrich_uncorr', 'p_value_20', 'p_value_50', 'p_value_80', 'coloc_fdr', 'coloc_int_fdr']:
        avg_prec = None
        if m in res.mols_df.columns:
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
    res.metric_counts = {'n_expected': df.is_expected.sum(), 'n_unexpected': (~df.is_expected).sum()}

if __name__ == '__main__':
    add_metric_scores(test_results)

#%%
def add_filter_reason(res):
    filter_reason = res.mols_df.apply(lambda s: 'off-sample parent' if s.off_sample else 'no coloc' if s.coloc == 0 else 'structural analogue' if s.is_redundant else '', axis=1)
    res.mols_df['filter_reason'] = filter_reason
    res.ann_mols_df = res.ann_mols_df.merge(res.mols_df.filter_reason, how='left', left_on='hmdb_id', right_index=True)

if __name__ == '__main__':
    add_filter_reason(test_results)
# %%
def clip_mz_range(res: DSResults, lo_mz, hi_mz):
    pass
if __name__ == '__main__':
    test_results = get_msms_results_for_ds('2020-06-19_16h39m10s')
    clip_mz_range(test_results, 80, 1000)
    print(test_results.anns_df.mz.min(), test_results.anns_df.mz.max())
# %%

@lru_cache(maxsize=None)
def get_ds_results(ds_id, mz_range=None):
    res = get_msms_results_for_ds(ds_id, mz_range)
    __import__('__main__').res = res
    add_expected_mols(res)
    # add_tfidf_score(res)
    # add_enrichment_analysis(res)
    # add_p_values(res)
    add_fdr(res)
    # add_metric_scores(res)
    add_filter_reason(res)
    return res


#%%