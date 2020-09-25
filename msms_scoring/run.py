# This file is used for iterative development. If running in an IPython environment, autoreload all
# imported modules if they change
try:
  from IPython import get_ipython
  get_ipython().run_line_magic('load_ext', 'autoreload')
  get_ipython().run_line_magic('autoreload', '2')
except ImportError:
  pass

#%%
import numpy as np
import pandas as pd
from msms_scoring.datasets import dataset_ids, msms_mol_ids, datasets_df
from msms_scoring.datasets_full import whole_body_ds_ids, high_quality_ds_ids, high_quality_full_ds_ids
from msms_scoring.fetch_data import get_msms_df, get_msms_results_for_ds, fetch_ds_results
from msms_scoring.metrics import get_ds_results
from msms_scoring.plots import plot_fdr_vs_precision
from msms_scoring.exports import export_mean_average_precision, export_fragments, export_molecule_well_behavedness, export_top_molecules, export_mols_for_chemrich, export_mols_for_chemrich_summarization, export_fragments2, export_mean_average_precision2
from msms_scoring.chemrich import get_summary_df_for_chemrich2, run_chemrich

#%%
pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 103)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
#%%
ds = fetch_ds_results(dataset_ids[0])
#%%
ds = get_msms_results_for_ds(dataset_ids[0], use_cache=False)
#%%
ds = get_ds_results(dataset_ids[0])
#%%
msms_df = get_msms_df()
#%%

for ds_id in dataset_ids:
    export_fragments(ds_id, 'raw data/spotting_v3')
#%%
export_mean_average_precision(dataset_ids, 'scoring_results/metric scores.xlsx')
#%%

for ds_id in dataset_ids:
    export_fragments2(ds_id, 'raw data/spotting_v3')
#%%
export_mean_average_precision2(dataset_ids, 'scoring_results/metric scores.xlsx')
#%%
export_molecule_well_behavedness(dataset_ids, msms_mol_ids)
#%%

plot_fdr_vs_precision([dataset_ids[0], dataset_ids[4], dataset_ids[6]])
# plot_fdr_vs_precision(dataset_ids)
#%%

export_top_molecules(dataset_ids, 'spotting')
# hq_in_mz_range = [ds_id for ds_id in high_quality_full_ds_ids if get_msms_results_for_ds(ds_id).anns.mz.min() < 200]

# for ds_id in hq_in_mz_range:
#     export_fragments(ds_id, 'raw data/hq_mz_range_includes_150')
# export_top_molecules(hq_in_mz_range, 'high_quality_mz_range_includes_150')
#%%
# for ds_id in dataset_ids[4:6]:
#     export_mols_for_chemrich(ds_id)
export_mols_for_chemrich_summarization([*whole_body_ds_ids, *high_quality_full_ds_ids], 'summary')
#%%
summary_df = get_summary_df_for_chemrich2([*whole_body_ds_ids, *high_quality_full_ds_ids])

#%%
sn_level = summary_df[summary_df.lipid_sn_name.isna() | ~summary_df.lipid_sn_name.duplicated()]
run_chemrich(sn_level, 'summary sn-level lipids')

s_level = summary_df[summary_df.lipid_sn_name.isna() | ~summary_df.lipid_s_name.duplicated()]
run_chemrich(s_level, 'summary species-level lipids')

run_chemrich(summary_df[summary_df.is_lipid], 'summary only lipids')
run_chemrich(summary_df[~summary_df.is_lipid], 'summary no lipids')

#%%
pos_ds_ids = list(datasets_df[datasets_df['set'].isin(['whole_body','high_quality']) & (datasets_df.polarity == 'positive')].ds_id)
pos_summary_df = get_summary_df_for_chemrich2(pos_ds_ids)
run_chemrich(pos_summary_df, 'summary only positive')

neg_ds_ids = list(datasets_df[datasets_df['set'].isin(['whole_body','high_quality']) & (datasets_df.polarity == 'negative')].ds_id)
neg_summary_df = get_summary_df_for_chemrich2(neg_ds_ids)
run_chemrich(neg_summary_df, 'summary only negative')
#%%