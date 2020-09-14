import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from msms_scoring.datasets import dataset_aliases, dataset_polarity
from msms_scoring.fetch_data import get_msms_df

#%%
matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = 15, 10

#%%

grid_mapping = pd.read_csv('spotting/msms_grid_mapping.csv', index_col=0)


def get_color_mapping(spots):
    mapping = {'Observed': None}  # Dicts preserve order, which is useful for consistent colors
    for spot in spots:
        spot_info = grid_mapping.loc[spot]
        spot_hmdb_ids = [m for m in spot_info[['hmdb_id1', 'hmdb_id2', 'hmdb_id3']] if pd.notna(m)]
        for hmdb_id in spot_hmdb_ids:
            mapping.setdefault(hmdb_id, None)

    return dict(zip(mapping.keys(), sns.color_palette(n_colors=len(mapping))))


def plot_spot(ax: plt.Axes, spot, ds_id, color_mapping):
    spot_info = grid_mapping.loc[spot]
    spot_mzs, spot_ints = np.loadtxt(f'sum_spectra/{ds_id}/raw/{spot}.txt').T
    spot_mols = [m for m in spot_info[['name1', 'name2', 'name3']] if pd.notna(m)]
    spot_hmdb_ids = [m for m in spot_info[['hmdb_id1', 'hmdb_id2', 'hmdb_id3']] if pd.notna(m)]
    max_intensity = np.max(spot_ints)

    markerline, stemlines, baseline = ax.stem(
        spot_mzs, spot_ints, label='Observed', markerfmt=' ', use_line_collection=True
    )
    stemlines.set_color(color_mapping['Observed'])
    plt.setp(baseline, visible=False)

    # Draw reference spectra underneath observed spectrum
    msms_df = get_msms_df()  # Don't load this globally as PyCharm console tends to freeze with such a big dataframe :(
    ds_polarity = dataset_polarity[ds_id]
    for hmdb_id, mol_name in zip(spot_hmdb_ids, spot_mols):
        ref_mzs = msms_df[(msms_df.polarity == ds_polarity) & (msms_df.hmdb_id == hmdb_id)].mz.values
        ref_intensity = np.ones_like(ref_mzs) * (-max_intensity / 10)

        markerline, stemlines, baseline = ax.stem(
            ref_mzs,
            ref_intensity,
            label=f'{mol_name} reference',
            markerfmt=' ',
            use_line_collection=True,
        )
        stemlines.set_color(color_mapping[hmdb_id])
        plt.setp(baseline, visible=False)


    ax.set_xlim(0, 800)
    ax.set_ylim(-max_intensity/10, max_intensity)
    ax.set_title(' + '.join(spot_mols))
    ax.legend()



ds_id = '2020-08-03_13h23m33s'
ds_name = dataset_aliases[ds_id]

spots = [
    '3_1',  # Spermine
    '4_2',  # Spermine,Spermidine
    '5_3',  # Spermidine
]
color_mapping = get_color_mapping(spots)

plt.close('all')
fig, axs = plt.subplots(len(spots), 1, figsize=(15,15), sharex='all', sharey='all')

fig.suptitle(ds_name)

for ax, spot in zip(np.array(axs).flat, spots):
    plot_spot(ax, spot, ds_id, color_mapping)

fig.show()