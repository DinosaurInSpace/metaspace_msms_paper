# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, Union, Tuple, List

import numpy as np
import pandas as pd

from metaspace_msms_mirror_spectra import mirror_main
from msms_scoring.fetch_data import DSResults, get_msms_df
from msms_scoring.metrics import get_ds_results, add_metric_scores


# %%
from msms_scoring.metrics2 import get_ds_results2


def find_interesting_groups(res: DSResults):
    # Build lookup of parent scores, indexed by fragment formulas
    df = res.ann_mols_df[res.ann_mols_df.is_detected].set_index('hmdb_id').drop(columns=['is_detected'])
    parents_df = df[df.is_parent]
    frags_df = df[~df.is_parent]

    # Summarize stats per group
    def get_isobar_summary(df):
        return pd.Series({
            'n_total': len(df),
            'n_confident': (df.global_enrich_uncorr <= 0.1).sum(),
            'n_unsure': ((df.global_enrich_uncorr > 0.1) & (df.global_enrich_uncorr <= 0.5)).sum(),
            'n_unlikely': (df.global_enrich_uncorr > 0.5).sum(),
        })

    parents_summary_df = parents_df.groupby('formula').apply(get_isobar_summary)
    frags_summary_df = frags_df.groupby('formula').apply(get_isobar_summary)

    summary_df = (
        parents_summary_df.add(frags_summary_df, fill_value=0)
            .merge(parents_summary_df, how='left', left_index=True, right_index=True, suffixes=('', '_p'))
            .merge(frags_summary_df, how='left', left_index=True, right_index=True, suffixes=('', '_f'))
            .fillna(0)
    )

    # Pick interesting groups
    can_pick_one = summary_df[(summary_df.n_confident_p == 1) & (summary_df.n_confident_f == 0) & (summary_df.n_unsure == 0) & (summary_df.n_total > 1)]
    can_refine = summary_df[(summary_df.n_confident > 0) & (summary_df.n_unlikely > 0)]
    doubtful_annotation = summary_df[(summary_df.n_confident_p == 0) & (summary_df.n_unsure_p == 0) & (summary_df.n_unlikely_p > 0)]

    def candidates_matching(values):
        # col_order = [
        #     'parent_formula', 'is_parent',
        #     'enrich_ratio', 'enrich_p', 'enrich_p_uncorr', 'tfidf_score',
        #     'mol_name', 'feature_n', 'parent_num_features', 'ann_href'
        # ]
        col_order = [
            'is_parent', 'is_lipid', 'is_expected', 'mz',
            'tfidf_score',
            'global_enrich_uncorr',
            'group_enrich_uncorr',
            'mol_name', 'coloc_to_parent', 'parent_n_detected', 'parent_n_frags',
            'parent_n_frags_unfiltered', 'ann_href', 'mol_href']
        return (
            pd.concat([
                parents_df[parents_df.formula.isin(values)],
                frags_df[frags_df.formula.isin(values)],
            ])
                .sort_values(['formula', 'group_enrich_p', 'is_parent'])
                .rename_axis(index='hmdb_id')
                .reset_index()
                .set_index(['formula', 'hmdb_id'])
            [col_order]
        )

    can_pick_one_ids = candidates_matching(can_pick_one.index)
    can_refine_ids = candidates_matching(can_refine.index)
    doubtful_annotation_ids = candidates_matching(doubtful_annotation.index)
    summary_df_ids = candidates_matching(summary_df.index)
    summary_by_id = summary_df_ids.reset_index().set_index(['hmdb_id', 'formula']).sort_index()

    return {
        'One good assignment anns': can_pick_one,
        'One good assignment mols': can_pick_one_ids,
        'Good split anns': can_refine,
        'Good split mols': can_refine_ids,
        'Doubtful annotation anns': doubtful_annotation,
        'Doubtful annotation mols': doubtful_annotation_ids,
        'All anns': summary_df,
        'All mols': summary_df_ids,
        'All mols by id': summary_by_id,
    }


# ds = get_ds_results('2020-06-19_16h39m10s')
# find_interesting_groups(ds)
# %%

def get_raw_export_data(res: DSResults):
    mols = res.mols_df.sort_values(['coloc_int_fdr'])
    return {
        'Mols': mols[[
            'mz', 'mol_name', 'formula', 'is_expected', 'parent_n_detected', 'parent_n_frags',
            'coloc_int', 'coloc_int_fdr', 'is_lipid', 'filter_reason', 'mol_href'
        ]],
        'Mols (full)': mols,
        'Mols by annotation': res.ann_mols_df.set_index(['formula', 'hmdb_id']).sort_values(['formula', 'coloc_int_fdr']),
        'Annotations by mol': res.ann_mols_df.set_index(['hmdb_id', 'formula']).sort_values(['hmdb_id', 'mz']),
        # 'Annotations': res.anns_df.sort_values(['mz']),
    }


# %% Export
SCALE_ONE_TO_ZERO = {
    'type': '3_color_scale',
    'min_type': 'num', 'min_value': 0.0, 'min_color': '#63BE7B',
    'mid_type': 'num', 'mid_value': 0.5, 'mid_color': '#FFEB84',
    'max_type': 'num', 'max_value': 1.0, 'max_color': '#F8696B',
}
SCALE_DEFAULT = {
    'type': '3_color_scale',
    'min_color': '#FFFFFF',
    'mid_color': '#FFEB84',
    'max_color': '#63BE7B',
}
COLUMN_SCALES = {
    'global_enrich_p': SCALE_ONE_TO_ZERO,
    'global_enrich_uncorr': SCALE_ONE_TO_ZERO,
    'group_enrich_p': SCALE_ONE_TO_ZERO,
    'group_enrich_uncorr': SCALE_ONE_TO_ZERO,
    'feature_n': None,
    'parent_num_features': None,
}


def export(export_data: Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]], out_file: str, grouped_sheets=False):
    drop_cols = [
        'inv_tfidf', 'tfidf', 'global_enrich_p', 'global_enrich_uncorr', 'group_enrich_p', 'group_enrich_uncorr',
        'p_value_20','p_value_50','p_value_80',
        # 'msm_coloc_fdr','msm_x_coloc_fdr'
        'all_frag_formulas',
    ]

    def to_excel_colorize(writer, df, sheet_options, autofilter=True, **kwargs):
        column_scales = {**COLUMN_SCALES, **sheet_options.get('column_scales', {})}
        column_scale_default = sheet_options.get('column_scale_default', SCALE_DEFAULT)
        index = sheet_options.get('index', True)
        header = sheet_options.get('header', True)

        if df is not None and not df.empty:
            df = df.drop(columns=drop_cols, errors='ignore')

            index_cols = df.index.nlevels if index else 0
            header_rows = df.columns.nlevels if header else 0
            df.to_excel(writer, index=index, header=header, freeze_panes=(header_rows, index_cols), merge_cells=not autofilter, **kwargs)

            worksheet = writer.book.worksheets()[-1]
            if index:
                indexes = [(name or i, df.index.get_level_values(i).dtype, df.index.get_level_values(i)) for i, name in enumerate(df.index.names)]
            else:
                indexes = []
            columns = [(name, df.dtypes[name], df[name]) for name in df.columns]
            for col_i, (name, dtype, values) in enumerate([*indexes, *columns]):
                if np.issubdtype(dtype.type, np.number):
                    options = column_scales.get(name, column_scale_default)
                    # options = {'type': '3_color_scale'}
                    if options:
                        worksheet.conditional_format(header_rows, col_i, worksheet.dim_rowmax, col_i, options)

                width = max(10, len(str(name)))
                if not np.issubdtype(dtype.type, np.number):
                    for v in values:
                        width = min(max(width, len(str(v)) * 3 // 2), 50)
                worksheet.set_column(col_i, col_i, width=width)

            if autofilter:
                worksheet.autofilter(0, 0, header_rows + len(df.index) - 1, index_cols + len(df.columns) - 1)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', ".*Ignoring URL .* since it exceeds Excel's limit of .*")
        with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
            for sheet_name, data in export_data.items():
                if isinstance(data, tuple):
                    data, sheet_options = data
                else:
                    sheet_options = {}
                if data is not None:
                    if grouped_sheets is None:
                        to_excel_colorize(writer, data, sheet_options, autofilter=True, sheet_name=sheet_name)
                        if data.index.nlevels > 1:
                            sn = sheet_name if grouped_sheets is True else sheet_name + ' (grouped)'
                            to_excel_colorize(writer, data, sheet_options, autofilter=False, sheet_name=sn)
                    else:
                        to_excel_colorize(writer, data, sheet_options, autofilter=not grouped_sheets, sheet_name=sheet_name)

    print(f'Saved {out_file}')


def export_fragments(ds_id: Union[DSResults, str], prefix='raw data'):
    res = get_ds_results(ds_id) if isinstance(ds_id, str) else ds_id
    export_data = get_raw_export_data(res)
    export(export_data, f'./scoring_results/{prefix}_{res.ds_id}_{res.name}.xlsx', grouped_sheets=False)


def get_raw_export_data2(res: DSResults):
    full_msms = res.msms_df.sort_values(['coloc_int_fdr']).drop(columns='hmdb_id')
    return {
        'MS MS annotations': full_msms[[
            'mz', 'mol_name', 'parent_formula', 'is_expected', 'off_sample',
            'parent_n_detected', 'parent_n_frags', 'coloc_int', 'coloc_int_fdr', 'mol_href'
        ]],
        'MS MS annotations (full)': full_msms,
        'Annotations': res.anns_df.sort_values(['mz']),
    }


def export_fragments2(ds_id: Union[DSResults, str], prefix='raw data'):
    res = get_ds_results2(ds_id) if isinstance(ds_id, str) else ds_id
    export_data = get_raw_export_data2(res)
    export(export_data, f'./scoring_results/{prefix}_r2_{res.ds_id}_{res.name}.xlsx', grouped_sheets=False)


# %%
def plot_pseudo_ms_spectra(ds_ids):
    spectra_df = pd.read_pickle('./input/cm3_reference_spectra_df.pickle')
    # 'binary', 'fdr',
    for y_axis in ['msm', 'cos', 'intensity']:
        for ds_id in ds_ids:
            print(f'{ds_id} {y_axis}')
            ds = get_ds_results(ds_id)

            mirror_main(
                ds.ds_id,
                ds.db_id,
                ds.ds_id,
                f'./scoring_results/pseudo-msms plots/data/',
                f'./scoring_results/pseudo-msms plots/{ds.name}_{y_axis}/',
                ds.sm_ds.polarity.lower(),
                y_axis,
                spectra_df
            )

# %% Report mean-average-precision for each metric
def export_mean_average_precision(ds_ids: List[Union[DSResults, str]], name, skip_mAP=False):
    PARAMS = ['selected']
    DS_NAMES = []
    METRICS = ['random', 'coloc_int_fdr']
    metrics_df = []
    for ds_id in ds_ids:
        ds = get_ds_results(ds_id) if isinstance(ds_id, str) else ds_id
        DS_NAMES.append(ds.name)
        for params in PARAMS:
            add_metric_scores(ds, params)
            metrics_df.append(ds.metric_scores.assign(ds=ds.name, params=params, **ds.metric_counts))

    metrics_df = pd.concat(metrics_df, ignore_index=True)
    metrics_df = metrics_df.set_index(['params', 'ds', 'metric']).reindex(product(PARAMS, DS_NAMES, METRICS)).reset_index()
    metric_counts = metrics_df[['ds', 'params', 'n_expected', 'n_unexpected']].drop_duplicates()
    metric_counts = metric_counts.set_index(['ds', 'params']).reindex(product(DS_NAMES, PARAMS))
    mAP_stats = metrics_df.groupby(['params', 'metric']).avg_prec.describe().drop(columns=['count']).reindex(product(PARAMS, METRICS))
    mAP_vs_random = mAP_stats.reset_index().groupby('params').apply(lambda df: df.assign(mean=df['mean'] / df['mean'][df.metric == 'random'].mean())).pivot(index='metric', columns='params', values='mean').reindex(index=METRICS, columns=PARAMS)
    mAP_by_params = mAP_stats.reset_index().pivot(index='metric', columns='params', values='mean').reindex(index=METRICS, columns=PARAMS)
    wmAP = metrics_df.groupby(['params', 'metric']).apply(lambda df: np.average(df.avg_prec, weights=df.n_expected)).reindex(product(PARAMS, METRICS))
    wmAP = pd.DataFrame({'wmAP': wmAP}).reset_index()
    wmAP_by_params = wmAP.pivot(index='metric', columns='params', values='wmAP').reindex(index=METRICS, columns=PARAMS)

    avg_prec = metrics_df.pivot_table(index=['params', 'metric'], columns='ds', values='avg_prec').reindex(index=product(PARAMS, METRICS), columns=DS_NAMES)

    export({
        'Weighted mAP': wmAP_by_params,
        'mAP vs random': mAP_vs_random if not skip_mAP else None,
        'mAP': mAP_by_params if not skip_mAP else None,
        'Filter stats': metric_counts,
        'Avg precision': avg_prec,
        'mAP stats': mAP_stats,
        'Raw data': metrics_df.set_index(['params','ds','metric']),
    }, f'./scoring_results/{name}_metric_results.xlsx')
# %% Report mean-average-precision with & without an m/z range filter
def export_mean_average_precision_with_range_filter(ds_ids):
    PARAMS = ['unfiltered', 'no_off_sample', 'no_zero_coloc', 'no_structural_analogues', 'all_filters']
    DS_NAMES = []
    METRICS = ['random', 'coloc_fdr', 'coloc_int_fdr']
    MZ_RANGES = ['full', 'semi', 'clipped']
    metric_scores = []
    metric_counts = []
    # ds_ids = [
    #     '2020-06-19_16h39m10s',
    #     '2020-06-19_16h39m12s',
    # ]
    for ds_id in ds_ids:
        # Unclipped results
        ds = get_ds_results(ds_id)
        ds_name = ds.name  # Use unclipped name for both
        print(ds.name)
        DS_NAMES.append(ds_name)
        for params in PARAMS:
            add_metric_scores(ds, params)
            extra = {'ds': ds_name, 'params': params, 'mz_range': 'full'}
            metric_scores.append(ds.metric_scores.assign(**extra))
            metric_counts.append({**ds.metric_counts, **extra})
        # Semi-clipped results (Unclipped results for mols that pass the filter while clipped)
        for params in PARAMS:
            add_metric_scores(ds, params, min_mz=100)
            extra = {'ds': ds_name, 'params': params, 'mz_range': 'semi'}
            metric_scores.append(ds.metric_scores.assign(**extra))
            metric_counts.append({**ds.metric_counts, **extra})
        # Clipped results
        ds = get_ds_results(ds_id, mz_range=(100, 1000))
        print(ds.name)
        for params in PARAMS:
            add_metric_scores(ds, params)
            extra = {'ds': ds_name, 'params': params, 'mz_range': 'clipped'}
            metric_scores.append(ds.metric_scores.assign(**extra))
            metric_counts.append({**ds.metric_counts, **extra})


    metric_scores = pd.concat(metric_scores, ignore_index=True)
    metric_scores = metric_scores.set_index(['params','ds','mz_range','metric']).reindex(product(PARAMS, DS_NAMES, MZ_RANGES, METRICS)).reset_index()
    metric_counts = pd.DataFrame(metric_counts)
    metric_counts = (metric_counts
                     .pivot_table(index=['ds','params'], columns=['mz_range'], values=['n_expected','n_unexpected'])
                     .reindex(index=product(DS_NAMES, PARAMS), columns=product(['n_expected','n_unexpected'], MZ_RANGES)))
    mAP_stats = metric_scores.groupby(['mz_range', 'params', 'metric']).avg_prec.describe().drop(columns=['count']).reindex(product(MZ_RANGES, PARAMS, METRICS))
    mAP_vs_random = (mAP_stats.reset_index()
                     .groupby(['params'], as_index=False)
                     .apply(lambda df: df.assign(mean=df['mean'] / df['mean'][(df.metric == 'random') & (df.mz_range == 'full')].mean()))
                     .pivot_table(index=['params'], columns=['metric', 'mz_range'], values='mean')
                     .reindex(index=PARAMS, columns=product(METRICS, MZ_RANGES)))
    mAP_by_params = (mAP_stats.reset_index()
                     .pivot_table(index=['params'], columns=['metric', 'mz_range'], values='mean')
                     .reindex(index=PARAMS, columns=product(METRICS, MZ_RANGES)))
    avg_prec = (metric_scores
                .pivot_table(index=['ds', 'params'], columns=['metric', 'mz_range'], values='avg_prec')
                .reindex(index=product(DS_NAMES, PARAMS), columns=product(METRICS, MZ_RANGES)))

    export({
        'Mean avg prec vs random': mAP_vs_random,
        'Mean avg prec': mAP_by_params,
        'Filter stats': metric_counts,
        'Avg precision': avg_prec,
        'Mean avg prec stats': mAP_stats,
        # 'Raw data': metric_scores.set_index(['params','ds','metric']),
    }, 'scoring_results/metric_scores_clipped.xlsx', grouped_sheets=True)

# %% Report mean-average-precision for each metric
def export_mean_average_precision2(ds_ids: List[Union[DSResults, str]], name: str):
    DS_NAMES = []
    METRICS = ['random', 'coloc_int_fdr']
    metrics_df = []
    metric_counts = []
    for ds_id in ds_ids:
        ds = get_ds_results2(ds_id) if isinstance(ds_id, str) else ds_id
        DS_NAMES.append(ds.name)
        ds_df = ds.msms_df
        n_expected = np.count_nonzero(ds_df[lambda df: df.is_expected & ~df.off_sample])
        metrics_df.append(ds.metric_scores.assign(ds=ds.name, n_expected=n_expected))
        metric_counts.append({
            'ds': ds.name,
            'off_sample_expected': np.count_nonzero(ds_df.is_expected & ds_df.off_sample),
            'off_sample_unexpected': np.count_nonzero(~ds_df.is_expected & ds_df.off_sample),
            'on_sample_expected': np.count_nonzero(ds_df.is_expected & ~ds_df.off_sample),
            'on_sample_unexpected': np.count_nonzero(~ds_df.is_expected & ~ds_df.off_sample),
        })

    metrics_df = pd.concat(metrics_df, ignore_index=True)
    metrics_df = metrics_df.set_index(['ds', 'metric']).reindex(product(DS_NAMES, METRICS)).reset_index()
    metric_counts = pd.DataFrame(metric_counts).set_index(['ds']).reindex(DS_NAMES)
    mAP_stats = metrics_df.groupby(['metric']).avg_prec.describe().drop(columns=['count']).reindex(METRICS)
    wmAP = metrics_df.groupby(['metric']).apply(lambda df: np.average(df.avg_prec, weights=df.n_expected)).reindex(METRICS)
    wmAP = pd.DataFrame({
        'weighted mAP': wmAP,
        'raw mAP': mAP_stats['mean'],
    })
    wmAP.loc['coloc_int_fdr / random'] = wmAP.loc['coloc_int_fdr'] / wmAP.loc['random']
    avg_prec = metrics_df.pivot_table(index='ds', columns='metric', values='avg_prec').reindex(index=DS_NAMES, columns=METRICS)

    export({
        'mAP': wmAP,
        'Annotation counts': metric_counts,
        'Avg precision': avg_prec,
        'Raw data': metrics_df.set_index(['ds','metric']),
    }, f'./scoring_results/{name}_r2_metric_results.xlsx')
# %%

def export_molecule_well_behavedness(ds_ids, hmdb_ids):
    msms_df = get_msms_df()
    dss = [get_ds_results(ds_id) for ds_id in ds_ids]
    ds_names = [ds.name for ds in dss]
    mol_names = [msms_df[msms_df.hmdb_id == hmdb_id].mol_name.iloc[0] for hmdb_id in hmdb_ids]
    coloc_int_fdr = pd.DataFrame(index=mol_names, columns=ds_names, dtype='f')
    sum_coloc = pd.DataFrame(index=mol_names, columns=ds_names, dtype='f')
    num_frags = pd.DataFrame(index=mol_names, columns=ds_names, dtype='f')
    intensity = pd.DataFrame(index=mol_names, columns=ds_names, dtype='f')
    filter_reason = pd.DataFrame(index=mol_names, columns=ds_names, dtype='O')
    good = pd.DataFrame(index=mol_names, columns=ds_names, dtype='f')

    for ds, ds_name in zip(dss, ds_names):
        for hmdb_id, mol_name in zip(hmdb_ids, mol_names):
            if hmdb_id in ds.mols_df.index:
                row = ds.mols_df.loc[hmdb_id]
                coloc_int_fdr.loc[mol_name, ds_name] = row.coloc_int_fdr
                sum_coloc.loc[mol_name, ds_name] = row.coloc
                num_frags.loc[mol_name, ds_name] = row.parent_n_detected
                intensity.loc[mol_name, ds_name] = row.intensity
                filter_reason.loc[mol_name, ds_name] = row.filter_reason
                if row.coloc_int_fdr < 0.1 and not row.filter_reason:
                    good.loc[mol_name, ds_name] = 1

    good['at_least_2'] = good.apply(lambda r: np.nansum(r) > 1, axis=1)

    export({
        'coloc_int_fdr': (coloc_int_fdr, {'column_scale_default': SCALE_ONE_TO_ZERO}),
        'filter_reason': filter_reason,
        'sum_coloc': sum_coloc,
        'num_frags': num_frags,
        'intensity': intensity,
        'good': good,
    }, 'scoring_results/well_behavedness.xlsx', grouped_sheets=False)

# %%

def export_top_molecules(ds_ids, output_name):
    results_dfs = []
    for ds_id in ds_ids:
        res = get_ds_results2(ds_id) if isinstance(ds_id, str) else ds_id
        results_dfs.append(res.msms_df.assign(ds_name=res.name))
    results_df = pd.concat(results_dfs)
    grp_fdr = results_df.groupby('ds_name').coloc_int_fdr
    ds_stats = pd.DataFrame({
        **{
            f'FDR <= {i*100}': grp_fdr.apply(lambda s: np.count_nonzero(s <= i))
            for i in [0.05,0.1,0.2,0.5]
        },
        'FDR < 100%': grp_fdr.apply(lambda s: np.count_nonzero(s < 1)),
        'FDR incalculable - no fragments': grp_fdr.apply(lambda s: np.count_nonzero(s >= 1)),
    })

    fdr_10 = (
        results_df[(results_df.coloc_int_fdr <= 0.1)]
        .sort_values('coloc_int_fdr')
        .reset_index(drop=True)
        .set_index(['ds_name', 'hmdb_id'])
    )
    summary = (
        fdr_10
        .reset_index()
        .groupby('hmdb_id')
        .agg({'ds_name': 'count', 'mol_name': 'max', 'coloc_int_fdr': 'min'})
        .rename(columns={'ds_name': 'detections', 'coloc_int_fdr': 'min_coloc_int_fdr'})
        .sort_values('min_coloc_int_fdr')
        .sort_values('detections', ascending=False, kind='mergesort')
    )

    export({
        'summary': (summary, {'column_scales': {'mean_coloc_int_fdr': SCALE_ONE_TO_ZERO}}),
        'ds_stats': ds_stats,
        'fdr_10': (fdr_10, {'column_scales': {'coloc_fdr': SCALE_ONE_TO_ZERO, 'coloc_int_fdr': SCALE_ONE_TO_ZERO}}),
    }, f'scoring_results/top_mols_{output_name}.xlsx', grouped_sheets=False)

# %%

def export_mols_for_chemrich(ds_id):
    res = get_ds_results(ds_id)
    df = res.mols_df[['mol_name','inchikey','pubchem_cid','smiles','coloc_int_fdr','coloc_int_fc']]
    df = df[df.coloc_int_fc > 0]
    df = df[df.coloc_int_fdr < 1]
    df['coloc_int_fdr'] = df.coloc_int_fdr / 2
    df['coloc_int_fc'] = np.where(df.coloc_int_fc < 1, df.coloc_int_fc / 2, df.coloc_int_fc * 2)
    df = df.drop_duplicates('pubchem_cid')
    df = df.rename(columns={
        'mol_name': 'Compound Name',
        'inchikey': 'InChiKeys',
        'pubchem_cid': 'Pubchem ID',
        'smiles': 'SMILES',
        'coloc_int_fdr': 'pvalue',
        'coloc_int_fc': 'foldchange',
    })
    export({
        'Sheet1': (df, {'index': False}),
    }, f'scoring_results/chemrich/{res.name} half-p-value double-foldchange/input data.xlsx', grouped_sheets=False)


def export_mols_for_chemrich_summarization(ds_ids, output_name):
    def get_median_fdr_fc(grp):
        grp = grp.sort_values('coloc_int_fdr')
        mid_idx = len(grp) // 2
        if len(grp) % 2 == 0:
            mid = grp.iloc[mid_idx:mid_idx + 2]
        else:
            mid = grp.iloc[mid_idx:mid_idx + 1]
        return pd.Series({
            'median_fdr': mid.coloc_int_fdr.mean(),
            'median_fc': mid.coloc_int_fc.mean(),
            'count': len(grp),
        })
    results_dfs = []
    for ds_id in ds_ids:
        res = get_ds_results(ds_id)
        results_dfs.append(res.mols_df.assign(ds_name=res.name))
    results_df = pd.concat(results_dfs)
    results_df = results_df[results_df.coloc_int_fc > 0]
    results_df = results_df[results_df.coloc_int_fdr < 1]
    mol_median_scores = results_df.groupby('pubchem_cid')[['coloc_int_fdr', 'coloc_int_fc']].apply(get_median_fdr_fc)
    df = (
        results_df[['mol_name','inchikey','pubchem_cid','smiles']]
        # Some HMDB IDs translated to the same CID, e.g. D-Fructose/L-Sorbose both get CID 92092.
        # Just pick one of the options arbitrarily
        .drop_duplicates('pubchem_cid')
        .merge(mol_median_scores, left_on='pubchem_cid', right_index=True)
        # ChemRICH doesn't have a configurable minimum p-value threshold, so scale the FDRs
        # to align our desired threshold to their imposed threshold
        .assign(median_fdr=lambda df: df.median_fdr / 2)
        .sort_values('median_fdr')
        .sort_values('count', ascending=False, kind='mergesort')
        .rename(columns={
            'mol_name': 'Compound Name',
            'inchikey': 'InChiKeys',
            'pubchem_cid': 'Pubchem ID',
            'smiles': 'SMILES',
            'median_fdr': 'pvalue',
            'median_fc': 'foldchange',
        })
    )

    export({
        'Sheet1': (df, {'index': False}),
    }, f'scoring_results/chemrich/{output_name}/input data.xlsx', grouped_sheets=False)

# %%