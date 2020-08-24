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
from msms_scoring.fetch_data import get_msms_df
from msms_scoring.metrics import get_ds_results
from msms_scoring.exports import export_mean_average_precision, export_fragments, export_molecule_well_behavedness

#%%
pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 103)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
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