# This file is used for iterative development. If running in an IPython environment, autoreload all
# imported modules if they change

try:
  from IPython import get_ipython
  get_ipython().run_line_magic('load_ext', 'autoreload')
  get_ipython().run_line_magic('autoreload', '2')
except ImportError:
  pass

#%%
import pandas as pd
from msms_scoring.datasets import dataset_ids, whole_body_ds_ids, high_quality_ds_ids, msms_mol_ids, datasets_df
from msms_scoring.fetch_data import get_msms_df, get_msms_results_for_ds, fetch_ds_results
from msms_scoring.metrics import get_ds_results, get_many_ds_results
from msms_scoring.metrics2 import get_many_ds_results2
from msms_scoring.plots import plot_fdr_vs_precision
from msms_scoring.exports import export_mean_average_precision, export_fragments, export_molecule_well_behavedness, export_top_molecules, export_mols_for_chemrich, export_mols_for_chemrich_summarization, export_fragments2, export_mean_average_precision2, export
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
for db_ver in ['v4','v3','v2','v1']:
    msms_df = get_msms_df(db_ver)
del msms_df # PyCharrrmmmm!!!!
#%%
ds2s = get_many_ds_results2(dataset_ids, db_ver=db_ver, include_lone_isotopic_peaks=False)
for ds in ds2s:
    export_fragments2(ds, f'raw data/spotting')
export_mean_average_precision2(ds2s, 'spotting')
#%%

for db_ver in ['v4']:
    for targeted in [True, False]:
        name = 'spotting_' + db_ver + ('_anymsm' if targeted else '_needsmsm')
        print(name, ' r2')
        ds2s = get_many_ds_results2(dataset_ids, db_ver=db_ver, include_lone_isotopic_peaks=targeted)
        print(name, 'get_many_ds_results2')
        if db_ver == 'v4':
            for ds in [ds2s[0], ds2s[4]]:
                export_fragments2(ds, f'raw data/{name}')
        export_mean_average_precision2(ds2s, name)
#%%

for db_ver in ['v4','v3','v2','v1']:
    for targeted in [True, False]:
        name = 'spotting_' + db_ver + ('_anymsm' if targeted else '_needsmsm')
        print(name, 'r1')
        ds1s = get_many_ds_results(dataset_ids, db_ver=db_ver, include_lone_isotopic_peaks=targeted)
        print(name, 'get_many_ds_results')
        for ds in [ds1s[0], ds1s[4]]:
            export_fragments(ds, f'raw data/{name}')
        export_mean_average_precision(ds1s, name)

#%%
export_molecule_well_behavedness(dataset_ids, msms_mol_ids)
#%%

tissue_ds_ids = list(datasets_df[datasets_df['set'].isin(['whole_body','high_quality'])].ds_id)
tdss = get_many_ds_results2(tissue_ds_ids, db_ver='v4', include_lone_isotopic_peaks=False)
for ds in tdss:
    export_fragments2(ds, f'tissue datasets/tissue')
export_top_molecules(tdss, 'tissue datasets')

# plot_fdr_vs_precision(dataset_ids)
#%%
ds2s = get_many_ds_results2(dataset_ids, db_ver='v4', include_lone_isotopic_peaks=False)
export_top_molecules(dataset_ids, 'spotting')
# hq_in_mz_range = [ds_id for ds_id in high_quality_full_ds_ids if get_msms_results_for_ds(ds_id).anns.mz.min() < 200]

# for ds_id in hq_in_mz_range:
#     export_fragments(ds_id, 'raw data/hq_mz_range_includes_150')
# export_top_molecules(hq_in_mz_range, 'high_quality_mz_range_includes_150')
#%%
# for ds_id in dataset_ids[4:6]:
#     export_mols_for_chemrich(ds_id)
export_mols_for_chemrich_summarization([*whole_body_ds_ids, *high_quality_ds_ids], 'summary')
#%%
summary_df = get_summary_df_for_chemrich2([*whole_body_ds_ids, *high_quality_ds_ids])

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

#%%