# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from msms_scoring.datasets import spotting_ds_ids, whole_body_ds_ids
from msms_scoring.metrics import get_ds_results

#%%
# Bigger plots
matplotlib.use('Qt5Agg')
plt.rcParams['figure.figsize'] = 15, 10

#%%
# Plot distributions of p-value split up by whether the molecule was expected or unexpected
def plot_grid(items, func, title=None, save_as=None):
    plt.close('all')
    w = int(len(items) ** 0.5 * 4 / 3)
    h = int(np.ceil(len(items) / w))
    fig, axs = plt.subplots(h, w)
    if title:
        fig.tight_layout(rect=(0,0,1,0.95))
        fig.suptitle(title)

    for i, item in enumerate(items):
        y, x = divmod(i, w)
        func(axs[y, x], item)

    if save_as:
        out = Path(save_as)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        print(f'Saved {save_as}')

    return fig


def plot_metric_values(ax, ds, field, filter, title):
    if field == 'group_enrich_uncorr':
        # only include groups with more than 1 mol
        formulas = ds.ann_mols_df.groupby('formula').hmdb_id.count()
        formulas = set(formulas.index[formulas > 1])
        df = ds.ann_mols_df[lambda df: df.is_detected & df.formula.isin(formulas)]
        # Test: Grab the best p-value from all candidates
        # df = ds.mols_df.merge(
        #     ds.ann_mols_df.groupby('hmdb_id').group_enrich_uncorr.min(),
        #     left_on='hmdb_id',
        #     right_index=True
        # )
        # df.loc[~df.formula.isin(formulas), 'group_enrich_uncorr'] = df.global_enrich_uncorr[~df.formula.isin(formulas)]
        # Test: Grab the parent p-value only
        # df = ds.ann_mols_df[lambda df: df.is_detected & df.is_parent & df.formula.isin(formulas)]
    else:
        df = ds.mols_df[ds.mols_df.is_detected]
    df = df[df.coloc > 0]

    C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    UA, EA, E, U, EI = C[:5]

    if filter == 'isomer':
        data = [
            df[df.is_expected],
            df[~df.is_expected & df.is_expected_isomer],
            df[~df.is_expected & ~df.is_expected_isomer],
        ]
        label = [
            'Expected mols',
            'Isomers of expected mols',
            'Unexpected mols'
        ]
        color = [E, EI, U]
    elif filter == 'analogue':
        data = [
            df[df.is_expected],
            df[~df.is_expected & df.is_expected_isomer & df.is_redundant],
            df[~df.is_expected & ~df.is_expected_isomer & ~df.is_redundant],
            df[~df.is_expected & ~df.is_expected_isomer & df.is_redundant],
        ]
        label = [
            'Expected mols',
            'Analogues of expected mols',
            'Unexpected mols',
            'Analogues of unexpected mols',
        ]
        color = [E, EA, U, UA]
    else:
        data = [
            df[df.is_expected],
            df[~df.is_expected & df.is_expected_isomer & df.is_redundant],
            df[~df.is_expected & df.is_expected_isomer & ~df.is_redundant],
            df[~df.is_expected & ~df.is_expected_isomer & ~df.is_redundant],
            df[~df.is_expected & ~df.is_expected_isomer & df.is_redundant],
        ]
        label = [
            'Expected mols',
            'Analogues of expected',
            'Non-analogous isomers of expected',
            'Unexpected mols',
            'Analogues of unexpected mols',
        ]
        color = [E, EA, EI, U, UA]
    data = [df[field] for df in data]
    range = {
        'tfidf': (1, 3),
        'coloc_fdr': (0, 1),
        'coloc_int_fdr': (0, 1),
        'old_coloc_fdr': (0, 1),
        'old_coloc_int_fdr': (0, 1),
        'msm_coloc_fdr': (0, 1),
        'msm_x_coloc_fdr': (0, 1),
    }.get(field, None)
    bins = 40 if sum(map(len, data)) > 200 else 20
    label = [f'{l} ({len(df)})' for l, df in zip(label, data)]
    ax.set_title(title, fontsize=8)
    if range:
        data = [np.clip(d, *range) for d in data]
    ax.hist(data, bins=bins, range=range, stacked=True, label=label, color=color)
    ax.legend(loc='upper right')

ds_sets = [
    ('spotting', spotting_ds_ids),
    # ('whole_body', whole_body_ds_ids),
]
# One plot per ds
# for ds_set, ds_ids in ds_sets:
#     dss = [get_ds_results(ds_id) for ds_id in ds_ids]
#     # 'global_enrich_uncorr', 'tfidf', 'p_value_20', 'p_value_50', 'p_value_80',
#     fields = ['coloc', 'coloc_int', 'coloc_fdr', 'coloc_int_fdr', 'old_coloc_fdr', 'old_coloc_int_fdr']
#     for ds in dss:
#         fig = plot_grid(
#             fields,
#             lambda ax, field: plot_metric_values(ax, ds, field, 'both', field),
#             title=ds.name,
#             save_as=f'./mol_scoring/metric histograms/{ds_set}_all_metrics_{ds.name}.png'
#         )

# One plot per field
for ds_set, ds_ids in ds_sets:
    for filter, title in [('isomer', 'isomers'), ('analogue', 'analogues'), ('both', 'isomers & analogues')]:
        dss = [get_ds_results(ds_id) for ds_id in ds_ids]
        # for field in ['global_enrich_uncorr', 'tfidf', 'p_value_20', 'p_value_50', 'p_value_80', 'coloc_fdr', 'msm_coloc_fdr', 'coloc_fdr', 'old_coloc_fdr', 'old_coloc_int_fdr']:
        for field in ['coloc_int_fdr']:
            fig = plot_grid(
                dss,
                lambda ax, ds: plot_metric_values(ax, ds, field, filter, ds.name),
                title=field + ' ' + title,
                save_as=f'./mol_scoring/metric histograms/{ds_set}_all_dss_{field}_{filter}.png'
            )


#%%

# Histogram of # of frags detected vs expected

def plot_n_frags_hist2d(ds, save_as=None):
    global df,a,b
    plt.close('all')
    n_coloc = ds.ann_mols_df.groupby('hmdb_id').coloc_to_parent.apply(lambda grp: np.count_nonzero(grp > 0.2))
    df = ds.mols_df[ds.mols_df.is_detected].assign(n_coloc=n_coloc)
    good = df[df.is_expected]
    bad = df[~df.is_expected & ~df.is_lipid]
    ugly = df[~df.is_expected & df.is_lipid]
    dfs = [good, bad, ugly]
    fig, axs = plt.subplots(3, 3)
    for x in range(3):
        for y in range(3):
            a = dfs[y].coloc_fdr
            b = [dfs[y].parent_n_frags, dfs[y].parent_n_detected, dfs[y].n_coloc][x]
            range_a, range_b = (0, 1), (1, 20)
            a = np.clip(a, *range_a)
            b = np.clip(b, *range_b)
            y_name = ['#frags', '#detected', '#detected & colocalized'][x]
            x_name = ['spotted', 'unspotted non-lipid', 'unspotted lipid'][y]
            # b_bins = [1,2,3,4,6,8,12,16,24,32]
            b_bins = np.arange(range_b[0] - 0.5, range_b[1] + 0.5)
            b_ticks = np.arange(range_b[0], range_b[1] + 1)
            axs[x, y].set_title(f'{y_name} for {x_name} mols')
            axs[x, y].yaxis.set_ticks(b_ticks)
            axs[x, y].hist2d(a, b, bins=[20, b_bins], range=[range_a, range_b])

    if save_as:
        out = Path(save_as)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        print(f'Saved {save_as}')
    return fig

for ds_set, ds_ids in [('spotting', spotting_ds_ids), ('whole_body', whole_body_ds_ids)]:
    dss = [get_ds_results(ds_id) for ds_id in ds_ids]
    for ds in dss:
        plot_n_frags_hist2d(ds, f'./mol_scoring/metric histograms/{ds_set}_n_frags_{ds.name}.png')

# plot_n_frags_hist2d(get_ds_results('2020-06-19_16h39m10s')).show()
#%%

