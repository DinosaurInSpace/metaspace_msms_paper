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
from msms_scoring.datasets import dataset_ids, whole_body_ds_ids, old_spotting_ds_ids, ito_spotting_ds_ids, high_quality_ds_ids
from msms_scoring.exports import export_mean_average_precision, export_fragments
from msms_scoring.fetch_data import get_msms_df
#%%
pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 103)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
#%%
msms_df = get_msms_df()
#%%
# for ds_id in ['2020-06-19_16h39m10s']: # OurCon DHB+ (60-360)
#     run(ds_id, 'raw data/spotting')
# for ds_id in spotting_ds_ids:
#     export_fragments(ds_id, 'raw data/spotting')
for ds_id in ito_spotting_ds_ids:
    export_fragments(ds_id, 'raw data/spotting')


# export_mean_average_precision(spotting_ds_ids, 'mol_scoring/metric scores (new datasets only).xlsx')
export_mean_average_precision(old_spotting_ds_ids + dataset_ids + ito_spotting_ds_ids, 'mol_scoring/metric scores (all datasets).xlsx', skip_mAP=True)
# export_mean_average_precision(old_spotting_ds_ids, 'mol_scoring/metric scores (old datasets only).xlsx')

#%%

for ds_id in high_quality_ds_ids:
    export_fragments(ds_id, 'raw data/high_quality')
for ds_id in whole_body_ds_ids:
    export_fragments(ds_id, 'raw data/whole_body')
#%%