# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
import pickle
from functools import lru_cache
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from metaspace.sm_annotation_utils import SMInstance, SMDataset
from scipy.ndimage import median_filter
from sklearn.metrics import pairwise_kernels
import cpyMSpec

#%%
from msms_scoring.datasets import dataset_aliases

sm = SMInstance()

def make_id_mapping_df():
    """
    Saved code for generating `id_mapping.csv` needed for get_msms_df. Has some manual steps, so don't run this directly
    """
    # Creating ID mapping using PubChem Exchange service
    get_msms_df().hmdb_id.drop_duplicates().to_csv('hmdb_ids.txt', header=None, index=None)

    # Use https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi to convert the hmdb_ids to
    # CIDs, SMILES and InChIKeys, renaming the output files to
    # 'hmdb_id_to_cid.txt', 'hmdb_id_to_smiles.txt', 'hmdb_id_to_inchikey.txt' respectively

    cids = pd.read_csv('hmdb_id_to_cid.txt', sep='\t', header=None, names=['hmdb_id', 'pubchem_cid'], index_col=0, skipinitialspace=True)
    smiles = pd.read_csv('hmdb_id_to_smiles.txt', sep='\t', header=None, names=['hmdb_id', 'smiles'], index_col=0, skipinitialspace=True)
    inchikeys = pd.read_csv('hmdb_id_to_inchikey.txt', sep='\t', header=None, names=['hmdb_id', 'inchikey'], index_col=0, skipinitialspace=True)
    id_mapping_df = (
        smiles
            .join(cids, how='left')
            .join(inchikeys, how='left')
    )
    id_mapping_df.loc['HMDB0000956', 'pubchem_cid'] = 444305  # L(+)-Tartaric acid mysteriously fails to translate
    id_mapping_df.loc['HMDB0000956', 'smiles'] = '[C@@H]([C@H](C(=O)O)O)(C(=O)O)O'
    id_mapping_df.loc['HMDB0000956', 'inchikey'] = 'FEWJPZIEWOKRBE-JCYAYHJZSA-N'
    id_mapping_df['pubchem_cid'] = id_mapping_df.pubchem_cid.apply(lambda cid: str(int(cid)))
    id_mapping_df.to_csv('to_metaspace/id_mapping.csv')

#%%
PARSE_MOL_ID = re.compile(r'([^_]+)_(\d+)([pf])')


@lru_cache()  # Only load when needed, as it eats a bunch of memory
def get_msms_df(use_v2=False):
    # cache_path = Path(f'./scoring_results/cache/msms_df{"_v2" if use_v2 else ""}.csv')
    # if cache_path.exists():
    #     return pd.read_csv(cache_path)

    if not use_v2:
        msms_df = pd.read_pickle('to_metaspace/cm3_msms_all_both.pickle')
        msms_df.rename(columns={'ion_mass': 'mz'}, inplace=True)
        # msms_df = msms_df[['polarity', 'id', 'name', 'formula', 'mz']]
    else:
        msms_df = pd.concat([
            pd.read_csv('to_metaspace/cm3_msms_all_pos_v2.csv', sep='\t').assign(polarity='positive'),
            pd.read_csv('to_metaspace/cm3_msms_all_neg_v2.csv', sep='\t').assign(polarity='negative'),
        ], ignore_index=True)
        # msms_df = msms_df[['polarity', 'id', 'name', 'formula']]
    msms_df['hmdb_id'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[1])
    msms_df['frag_idx'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[2]).astype(np.int32)
    msms_df['is_parent'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[3]) == 'p'
    msms_df['mol_name'] = msms_df.name.str.replace("^[^_]+_[^_]+_", "")
    msms_df['hmdb_href'] = 'https://hmdb.ca/metabolites/' + msms_df.hmdb_id

    # Calculate m/zs for each molecule
    if use_v2:
        mzs_lookup = []
        for polarity in ['positive', 'negative']:
            for formula in msms_df.formula[msms_df.polarity == polarity].unique():
                iso_pattern = cpyMSpec.isotopePattern(formula)
                iso_pattern.addCharge(1 if polarity == 'positive' else -1)
                mzs_lookup.append((polarity, formula, iso_pattern.masses[0]))
        mzs_lookup_df = pd.DataFrame(mzs_lookup, columns=['polarity', 'formula', 'mz'])
        msms_df = msms_df.merge(mzs_lookup_df, on=['polarity', 'formula'])

    # Clean up results by converting everything to HMDB IDs and removing items that can't be converted
    msms_df.replace({'hmdb_id': {
        'msmls87': 'HMDB0006557' # ADP-GLUCOSE -> ADP-glucose
    }}, inplace=True)
    ids_to_drop = [
        'msmls65',  # 5-HYDROXYTRYPTOPHAN (Different stereochemistry to HMDB0000472 5-Hydroxy-L-tryptophan, which is also included)
        'msmls183',  # DEOXYGUANOSINE-MONOPHOSPHATE (Identical to HMDB0001044 2'-Deoxyguanosine 5'-monophosphate)
        'msmls189',  # DGDP (Identical to HMDB0000960 dGDP)
        'C00968',  # 3',5'-Cyclic dAMP (Not in HMDB at all)
        'msmls142',  # CORTISOL 21-ACETATE (Not in HMDB at all)
        'msmls192',  # DIDECANOYL-GLYCEROPHOSPHOCHOLINE (Not in HMDB at all)
    ]
    msms_df = msms_df[~msms_df.hmdb_id.isin(ids_to_drop)]
    # Add is_lipid column
    # hmdb_lipid_ids.txt is derived from the HMDB 4.0 export, and includes the IDs of all molecules
    # with the super-class "Lipids and lipid-like molecules", or starting with "PC(", which are
    # also present in core_metabolome_v3
    lipid_ids = set(hmdb_id for hmdb_id in open('hmdb_lipid_ids.txt').read().split('\n') if hmdb_id)
    msms_df['is_lipid'] = msms_df.hmdb_id.isin(lipid_ids)
    # Add PubChem CIDs, SMILESs, InChIKeys, if the mapping has been generated
    if Path('to_metaspace/id_mapping.csv').exists():
        id_mapping = pd.read_csv('to_metaspace/id_mapping.csv', index_col=0)
        msms_df = msms_df.merge(id_mapping, left_on='hmdb_id', right_index=True, how='left')
    # Add sorted list of fragments for later deduping
    all_frags = msms_df.groupby(['hmdb_id', 'polarity']).formula.apply(lambda fs: ','.join(sorted(fs))).rename('all_frag_formulas')
    msms_df = msms_df.merge(
        all_frags,
        how='left',
        left_on=['hmdb_id', 'polarity'],
        right_index=True,
    )
    msms_df = msms_df.sort_values(['polarity','hmdb_id','frag_idx']).reset_index(drop=True)

    # cache_path.parent.mkdir(parents=True, exist_ok=True)
    # msms_df[['polarity','hmdb_id', 'mol_name','all_frag_formulas']].drop_duplicates().to_csv(cache_path, index=False)

    return msms_df

# msms_df = get_msms_df()

#%%
class DSResults:
    ds_id: str
    sm_ds: SMDataset
    db_id: str
    name: str
    anns: pd.DataFrame
    ds_images: Dict[str, np.ndarray]
    ds_coloc: pd.DataFrame
    mols_df: pd.DataFrame
    ann_mols_df: pd.DataFrame
    anns_df: pd.DataFrame
    metric_scores: pd.DataFrame

    def get_coloc(self, f1, f2):
        if f1 == f2:
            return 1
        if f1 not in self.ds_coloc.index or f2 not in self.ds_coloc.index:
            return 0
        return self.ds_coloc.loc[f1, f2]


def fetch_ds_results(ds_id):
    res = DSResults()
    res.ds_id = ds_id
    res.sm_ds = sm.dataset(id=ds_id)
    res.db_id = [db['id'] for db in res.sm_ds.database_details if re.match(r'^\d|^ls_cm3_msms_all_', db['name'])][0]
    if ds_id in dataset_aliases:
        res.name = dataset_aliases[res.ds_id]
    else:
        res.name = re.sub('[\W ]+', '_', res.sm_ds.name)
        res.name = re.sub('_cloned_from.*', '', res.name)
        res.name = re.sub('_full_msms.*', '', res.name)

    res.anns = res.sm_ds.results(database=res.db_id)
    ann_images = res.sm_ds.all_annotation_images(
        fdr=1,
        database=res.db_id,
        only_first_isotope=True,
        scale_intensity=False,
    )
    res.ds_images = dict((imageset.formula, imageset[0]) for imageset in ann_images)
    return res

if __name__ == '__main__':
    test_results = fetch_ds_results('2020-05-26_17h58m22s')
#%%

def add_coloc_matrix(res: DSResults):
    keys = list(res.ds_images.keys())
    images = list(res.ds_images.values())
    cnt = len(keys)
    if cnt == 0:
        res.ds_coloc = pd.DataFrame(dtype='f')
    elif cnt == 1:
        res.ds_coloc = pd.DataFrame([[1]], index=keys, columns=keys)
    else:
        h, w = images[0].shape
        flat_images = np.vstack(images)
        flat_images[flat_images < np.quantile(flat_images, 0.5, axis=1, keepdims=True)] = 0
        filtered_images = median_filter(flat_images.reshape((cnt, h, w)), (1, 3, 3)).reshape((cnt, h * w))
        distance_matrix = pairwise_kernels(filtered_images, metric='cosine')
        ds_coloc = pd.DataFrame(distance_matrix, index=keys, columns=keys, dtype='f')
        ds_coloc.rename_axis(index='source', columns='target', inplace=True)
        res.ds_coloc = ds_coloc

if __name__ == '__main__':
    add_coloc_matrix(test_results)
#%%

def add_result_dfs(res: DSResults, lo_mz=None, hi_mz=None):

    # Get detected IDs from dataset
    # This is unreliable - METASPACE only reports the top 50 candidate mols per annotation. Using formula matching instead
    # detected_frag_ids = set()
    # detected_mol_ids = set()
    # for mol_ids in res.anns.moleculeIds:
    #     detected_frag_ids.update(mol_ids)
    #     detected_mol_ids.update(PARSE_MOL_ID.match(mol_id).groups()[0] for mol_id in mol_ids)

    # Exclude fragments of the wrong polarity
    df = get_msms_df()
    df = df[df.polarity == res.sm_ds.polarity.lower()].copy()
    min_mz = max(res.anns.mz.min() - 0.1, lo_mz or 0)
    max_mz = min(res.anns.mz.max() + 0.1, hi_mz or 2000)
    df['in_range'] = (df.mz >= min_mz) & (df.mz <= max_mz)
    df['is_detected'] = df.formula.isin(res.anns.ionFormula) & df.in_range
    df['parent_is_detected'] = df.hmdb_id.isin(df[df.is_parent & df.is_detected].hmdb_id)
    href_base = f'https://beta.metaspace2020.eu/annotations?ds={res.ds_id}&db_id={res.db_id}&sort=mz&fdr=0.5&q='
    df['ann_href'] = href_base + df.formula
    v = pd.DataFrame({
        'parent_formula': df[df.is_parent].set_index('hmdb_id').formula,
        'parent_n_detected': df.groupby('hmdb_id').is_detected.sum().astype(np.int32),
        'parent_n_frags': df.groupby('hmdb_id').in_range.sum().astype(np.int32),
        'parent_n_frags_unfiltered': df.groupby('hmdb_id').frag_idx.max().astype(np.int32),
        'mol_href': df.groupby('hmdb_id').formula.apply(lambda f: href_base + '|'.join(f)),
    })
    df = df.merge(v, how='left', left_on='hmdb_id', right_index=True)

    df['coloc_to_parent'] = [
        # Explicitly check is_detected here so that the mz_range filter is applied
        res.get_coloc(f1, f2) if d1 and d2 else 0
        for f1, f2, d1, d2 in df[[
            'formula', 'parent_formula', 'is_detected', 'parent_is_detected'
        ]].itertuples(False, None)
    ]

    df = df.merge(
        res.anns[['ionFormula','msm','offSample','intensity']]
            .rename(columns={'offSample': 'off_sample'}),
        how='left', left_on='formula', right_on='ionFormula'
    )

    df = df.merge(
        res.anns[['ionFormula','intensity']]
            .rename(columns={'intensity': 'parent_intensity'}),
        how='left', left_on='parent_formula', right_on='ionFormula'
    )

    # Exclude molecules with no matches at all
    # df = df[df.hmdb_id.isin(df[df.is_detected].hmdb_id)]
    # Exclude molecules where the parent isn't matched
    df = df[df.parent_is_detected]
    # Exclude fragments that are outside of detected mz range
    # 0.1 buffer added because centroiding can move the peaks slightly
    df = df[df.in_range]

    res.mols_df = df[df.is_parent][[
        'hmdb_id', 'mz', 'is_detected', 'mol_name', 'formula', 'is_lipid',
        'msm', 'off_sample', 'intensity',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'mol_href', 'hmdb_href',
        'all_frag_formulas', 'pubchem_cid', 'smiles', 'inchikey',
    ]].set_index('hmdb_id')

    res.ann_mols_df = df[[
        'id', 'hmdb_id', 'mz', 'is_parent', 'is_detected', 'mol_name', 'formula', 'is_lipid',
        'msm', 'off_sample', 'intensity', 'parent_intensity',
        'coloc_to_parent', 'parent_formula', 'frag_idx', 'ann_href', 'mol_href', 'hmdb_href',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered',
    ]].set_index('id')

    res.anns_df = df[df.is_detected][[
        'formula', 'mz', 'msm', 'off_sample', 'intensity', 'ann_href',
    ]].drop_duplicates().set_index('formula')

if __name__ == '__main__':
    add_result_dfs(test_results)
# %%

def get_msms_results_for_ds(ds_id, mz_range=None, use_cache=True):
    mz_suffix = f'_{mz_range[0]}-{mz_range[1]}' if mz_range is not None else ''
    cache_path = Path(f'./scoring_results/cache/ds_results/{ds_id}{mz_suffix}.pickle')
    if not use_cache or not cache_path.exists():
        res = fetch_ds_results(ds_id)
        if mz_range is not None:
            res.name = f'{res.name} ({mz_range[0]}-{mz_range[1]})'
        add_coloc_matrix(res)
        add_result_dfs(res, *(mz_range or []))
        if mz_range:
            res.ds_coloc = res.ds_coloc.reindex(index=res.anns_df.index, columns=res.anns_df.index)
        res.ds_images = None  # Save memory/space as downstream analysis doesn't need this
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, cache_path.open('wb'))
    else:
        res = pickle.load(cache_path.open('rb'))
    return res


# test_results = get_msms_results_for_ds('2020-05-26_17h58m22s')

#%%