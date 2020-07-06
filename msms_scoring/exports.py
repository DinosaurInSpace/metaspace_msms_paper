# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from metaspace_msms_mirror_spectra import mirror_main
from msms_scoring.fetch_data import DSResults, get_msms_df
from msms_scoring.datasets import spotting_ds_ids, whole_body_ds_ids
from msms_scoring.metrics import get_ds_results


#%%
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
#%%

def get_raw_export_data(res: DSResults):
    return {
        'Mols': res.mols_df.sort_values(['is_expected', 'mz']),
        'Mols by annotation': res.ann_mols_df.set_index(['formula', 'hmdb_id']).sort_values(['formula', 'hmdb_id']),
        'Annotations by mol': res.ann_mols_df.set_index(['hmdb_id', 'formula']).sort_values(['hmdb_id', 'formula']),
        'Annotations': res.anns_df.sort_values(['mz']),
    }

#%%
def export(export_data: Dict[str, pd.DataFrame], out_file: str):

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
    column_formats = {
        'global_enrich_p': SCALE_ONE_TO_ZERO,
        'global_enrich_uncorr': SCALE_ONE_TO_ZERO,
        'group_enrich_p': SCALE_ONE_TO_ZERO,
        'group_enrich_uncorr': SCALE_ONE_TO_ZERO,
        'feature_n': None,
        'parent_num_features': None,
    }

    drop_cols = [
        # 'tfidf', 'global_enrich_p', 'global_enrich_uncorr', 'group_enrich_p', 'group_enrich_uncorr',
        # 'p_value_20','p_value_50','p_value_80',
        # 'msm_coloc_fdr','msm_x_coloc_fdr'
        'old_coloc_fdr', 'old_coloc_int_fdr',
    ]

    def to_excel_colorize(writer, df, index=True, header=True, autofilter=True, **kwargs):
        if not df.empty:
            df = df.drop(columns=drop_cols, errors='ignore')
            if 'coloc' in df.columns:
                df = df[df.coloc > 0]

            index_cols = df.index.nlevels if index else 0
            header_rows = df.columns.nlevels if header else 0
            df.to_excel(writer, index=index, header=header, freeze_panes=(header_rows, index_cols), merge_cells=not autofilter, **kwargs)

            worksheet = writer.book.worksheets()[-1]
            indexes = [(name or i, df.index.get_level_values(i).dtype, df.index.get_level_values(i)) for i, name in enumerate(df.index.names)]
            columns = [(name, df.dtypes[name], df[name]) for name in df.columns]
            for col_i, (name, dtype, values) in enumerate([*indexes, *columns]):
                if np.issubdtype(dtype.type, np.number):
                    options = column_formats.get(name, SCALE_DEFAULT)
                    # options = {'type': '3_color_scale'}
                    if options:
                        worksheet.conditional_format(header_rows, col_i, worksheet.dim_rowmax, col_i, options)

                width = max(10, len(name))
                if not np.issubdtype(dtype.type, np.number):
                    for v in values:
                        width = min(max(width, len(str(v)) * 3 // 2), 50)
                worksheet.set_column(col_i, col_i, width=width)

            if autofilter:
                worksheet.autofilter(0, 0, header_rows + len(df.index) - 1, index_cols + len(df.columns) - 1)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        for sheet_name, data in export_data.items():
            to_excel_colorize(writer, data, sheet_name=sheet_name)
            if data.index.nlevels > 1:
                to_excel_colorize(writer, data, autofilter=False, sheet_name=sheet_name + ' (grouped)')

    print(f'Saved {out_file}')


#%%
# Export reports for all datasets
def run(_ds_id, ds_set):
    res = get_ds_results(_ds_id)
    export_data = get_raw_export_data(res)
    export(export_data, f'./mol_scoring/{ds_set}_{res.ds_id}_{res.name}.xlsx')

# for ds_id in ['2020-06-19_16h39m10s']: # OurCon DHB+ (60-360)
#     run(ds_id, 'raw data/spotting')
for ds_id in spotting_ds_ids:
    run(ds_id, 'raw data/spotting')
# for ds_id in high_quality_ds_ids:
#     run(ds_id, 'raw data/high_quality')
for ds_id in whole_body_ds_ids:
    run(ds_id, 'raw data/whole_body')


#%%
spectra_df = pd.read_pickle('./input/cm3_reference_spectra_df.pickle')
# 'binary', 'fdr',
for y_axis in ['msm', 'cos', 'intensity']:
    for ds_id in spotting_ds_ids:
        print(f'{ds_id} {y_axis}')
        ds = get_ds_results(ds_id)

        mirror_main(
            ds.ds_id,
            ds.db_id,
            ds.ds_id,
            f'./mol_scoring/pseudo-msms plots/data/',
            f'./mol_scoring/pseudo-msms plots/{ds.name}_{y_axis}/',
            ds.sm_ds.polarity.lower(),
            y_axis,
            spectra_df
        )

#%%