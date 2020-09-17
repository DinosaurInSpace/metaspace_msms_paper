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
from msms_scoring.datasets import dataset_ids, msms_mol_ids
# from msms_scoring.datasets_full import whole_body_ds_ids, high_quality_ds_ids, high_quality_full_ds_ids
from msms_scoring.fetch_data import get_msms_df, get_msms_results_for_ds, fetch_ds_results
from msms_scoring.metrics import get_ds_results
from msms_scoring.plots import plot_fdr_vs_precision
from msms_scoring.exports import export_mean_average_precision, export_fragments, export_molecule_well_behavedness, export_top_molecules, export_mols_for_chemrich

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
    export_fragments(ds_id, 'raw data/spotting')

#%%

export_mean_average_precision(dataset_ids, 'scoring_results/metric scores.xlsx')
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
for ds_id in dataset_ids[4:6]:
    export_mols_for_chemrich(ds_id)
#%%