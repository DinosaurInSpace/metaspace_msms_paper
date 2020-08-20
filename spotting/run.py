import json
from pathlib import Path

import numpy as np
import pandas as pd
from metaspace import SMInstance

from msms_scoring.datasets import dataset_ids
from spotting.sum_spectra import get_ds_spots, group_peaks, get_background_spot_mapping, merge_groups, get_spot_bg, subtract_bg, batch_run

#%%

sm = SMInstance()
for ds_id in dataset_ids:
    print(f'Downloading {ds_id}')
    sm.dataset(id=ds_id).download_to_dir('raw_datasets', ds_id)


#%%
for ds_id in dataset_ids:
    print(ds_id)
    batch_run(ds_id)

#%%