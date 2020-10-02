# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from metaspace.sm_annotation_utils import SMInstance, SMDataset
from requests import Session
from scipy.ndimage import median_filter
from sklearn.metrics import pairwise_kernels
import cpyMSpec

#%%
from msms_scoring.datasets import dataset_aliases, dataset_mol_lists

def make_id_mapping_df():
    """
    Saved code for generating `id_mapping.csv` needed for get_msms_df. Has some manual steps, so don't run this directly
    """
    # Creating ID mapping using PubChem Exchange service
    get_msms_df('v4').hmdb_id.drop_duplicates().to_csv('hmdb_ids.txt', header=None, index=None)

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
def get_msms_df(db_ver):
    cache_path = Path(f'./scoring_results/cache/msms_df/{db_ver}.pickle')
    if cache_path.exists():
        return pickle.load(cache_path.open('rb'))
    if db_ver == 'v1':
        msms_df = pd.read_pickle('to_metaspace/cm3_msms_all_both.pickle')
        msms_df.rename(columns={'ion_mass': 'mz'}, inplace=True)
    else:
        msms_df = pd.concat([
            pd.read_csv(f'to_metaspace/cm3_msms_all_pos_{db_ver}.csv', sep='\t').assign(polarity='positive'),
            pd.read_csv(f'to_metaspace/cm3_msms_all_neg_{db_ver}.csv', sep='\t').assign(polarity='negative'),
        ], ignore_index=True)
    msms_df = msms_df[['polarity', 'id', 'name', 'formula']]
    msms_df['hmdb_id'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[1])
    msms_df['frag_idx'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[2]).astype(np.int32)
    msms_df['is_parent'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[3]) == 'p'
    msms_df['mol_name'] = msms_df.name.str.replace("^[^_]+_[^_]+_", "")
    msms_df['hmdb_href'] = 'https://hmdb.ca/metabolites/' + msms_df.hmdb_id

    # Calculate m/zs for each molecule
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
    lipid_ids = set(hmdb_id for hmdb_id in open('input/hmdb_lipid_ids.txt').read().split('\n') if hmdb_id)
    msms_df['is_lipid'] = msms_df.hmdb_id.isin(lipid_ids)
    # Add PubChem CIDs, SMILESs, InChIKeys, if the mapping has been generated
    if Path('input/id_mapping.csv').exists():
        id_mapping = pd.read_csv('input/id_mapping.csv', index_col=0)
        msms_df = msms_df.merge(id_mapping, left_on='hmdb_id', right_index=True, how='left')

    # Integrate names for lower structural resolution of lipids
    # Data from https://www.lipidmaps.org/tools/structuredrawing/lipid_levels.php
    if Path('input/lipid_structural_level_names.tsv').exists():
        lipid_mapping = (
            pd.read_csv('input/lipid_structural_level_names.tsv', delimiter='\t')
            .rename(columns={
                'Input name': 'mol_name',
                'sn-position level': 'lipid_sn_name',
                'Species level': 'lipid_s_name'
            })
            [['mol_name', 'lipid_sn_name', 'lipid_s_name']]
            [lambda df: df.lipid_s_name != '']
        )
        msms_df = msms_df.merge(lipid_mapping, on='mol_name', how='left')

    # Add sorted list of fragments for later deduping
    all_frags = msms_df.groupby(['hmdb_id', 'polarity']).formula.apply(lambda fs: ','.join(sorted(fs))).rename('all_frag_formulas')
    msms_df = msms_df.merge(
        all_frags,
        how='left',
        left_on=['hmdb_id', 'polarity'],
        right_index=True,
    )
    msms_df = msms_df.sort_values(['polarity','hmdb_id','frag_idx']).reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(msms_df, cache_path.open('wb'))
    return pickle.load(cache_path.open('rb'))


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
    msms_df: pd.DataFrame  # mols_df, grouped by structural analogue
    ann_mols_df: pd.DataFrame
    anns_df: pd.DataFrame
    metric_scores: pd.DataFrame
    metric_counts: dict

    def get_coloc(self, f1, f2):
        if f1 == f2:
            return 1
        if f1 not in self.ds_coloc.index or f2 not in self.ds_coloc.index:
            return 0
        return self.ds_coloc.loc[f1, f2]


def fetch_ds_results(ds_id):
    print(f'fetch_ds_results({ds_id})')
    res = DSResults()
    res.ds_id = ds_id
    sm = SMInstance()
    res.sm_ds = sm.dataset(id=ds_id)
    res.db_id = [db['id'] for db in res.sm_ds.database_details if re.match(r'^\d|^ls_cm3_msms_all_', db['name'])][0]
    if ds_id in dataset_aliases:
        res.name = dataset_aliases[res.ds_id]
    else:
        res.name = re.sub('[\W ]+', '_', res.sm_ds.name)
        res.name = re.sub('_cloned_from.*', '', res.name)
        res.name = re.sub('_full_msms.*', '', res.name)

    res.anns = res.sm_ds.results(database=res.db_id)
    return res

#%%

def get_images(res: DSResults, formulas: List[str]):
    import matplotlib.image as mpimg
    session = Session()
    formula_urls = {}
    for f, imgs in res.anns[res.anns.ionFormula.isin(formulas)][['ionFormula', 'isotopeImages']].itertuples(False, None):
        if imgs[0].get('url'):
            formula_urls[f] = res.sm_ds._baseurl + imgs[0].get('url')
    print(f'Getting {len(formula_urls)} images out of {len(formulas)}')
    def get_annotation_images(url):
        try:
            im = mpimg.imread(BytesIO(session.get(url).content))
        except:
            im = mpimg.imread(BytesIO(session.get(url).content))
        mask = im[:, :, 3]
        data = im[:, :, 0]
        data[mask == 0] = 0
        assert data.max() <= 1
        return data

    with ThreadPoolExecutor() as pool:
        return dict(zip(
            formula_urls.keys(),
            pool.map(get_annotation_images, formula_urls.values())
        ))

def add_coloc_matrix(res: DSResults):
    res.ds_images = get_images(res, res.ann_mols_df.formula[res.ann_mols_df.is_detected])

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

    res.ann_mols_df['coloc_to_parent'] = [
        # Explicitly check is_detected here so that the mz_range filter is applied
        res.get_coloc(f1, f2) if d1 else 0
        for f1, f2, d1 in res.ann_mols_df[[
            'formula', 'parent_formula', 'is_detected'
        ]].itertuples(False, None)
    ]

#%%

def combine_mols(df):
    df = df.sort_values('hmdb_id')
    row = df.iloc[0]
    return pd.Series({
        'hmdb_id': ','.join(df.hmdb_id),
        'mz': row.mz,
        'is_detected': df.is_detected.any(),
        'is_expected': df.is_expected.any(),
        'off_sample': row.off_sample,
        'mol_name': ','.join(df.mol_name),
        'intensity': row.intensity,
        'parent_formula': row.formula,
        'frag_formulas': sorted(f for f in row.all_frag_formulas.split(',') if f != row.formula),
        'all_frag_formulas': sorted(f for f in row.all_frag_formulas.split(',')),
        'parent_n_detected': row.parent_n_detected,
        'parent_n_frags': row.parent_n_frags,
        'mols_in_group': len(df),
        'expected_mols_in_group': df.is_expected.sum(),
        'unexpected_mols_in_group': (~df.is_expected).sum(),
        'mol_href': row.mol_href,
    })


def ion_image_filter(imgs):
    return imgs[0].get('url') is not None


def add_result_dfs(res: DSResults, lo_mz=None, hi_mz=None, db_ver=None, include_lone_isotopic_peaks=False):

    # Get detected IDs from dataset
    # This is unreliable - METASPACE only reports the top 50 candidate mols per annotation. Using formula matching instead
    # detected_frag_ids = set()
    # detected_mol_ids = set()
    # for mol_ids in res.anns.moleculeIds:
    #     detected_frag_ids.update(mol_ids)
    #     detected_mol_ids.update(PARSE_MOL_ID.match(mol_id).groups()[0] for mol_id in mol_ids)

    # Use latest DB for spotted datasets, original DB for non-spotted datasets, because
    # non-spotted datasets have had their DBs custom-generated from the original DB
    if db_ver is None:
        db_ver = 'v4' if res.ds_id in dataset_mol_lists else 'v1'
    df = get_msms_df(db_ver)
    # Exclude fragments of the wrong polarity
    df = df[df.polarity == res.sm_ds.polarity.lower()].copy()
    min_mz = max(res.anns.mz.min() - 0.1, lo_mz or 0)
    max_mz = min(res.anns.mz.max() + 0.1, hi_mz or 2000)
    expected_mols = dataset_mol_lists.get(res.ds_id, set())
    if include_lone_isotopic_peaks:
        detected_mols = set(res.anns.ionFormula[res.anns.isotopeImages.apply(ion_image_filter)])
    else:
        detected_mols = set(res.anns.ionFormula[res.anns.isotopeImages.apply(ion_image_filter) & (res.anns.msm > 0)])
    df['in_range'] = (df.mz >= min_mz) & (df.mz <= max_mz)
    df['is_detected'] = df.formula.isin(detected_mols) & df.in_range
    df['parent_is_detected'] = df.hmdb_id.isin(df[df.is_parent & df.is_detected].hmdb_id)
    df['is_expected'] = df.hmdb_id.isin(expected_mols)
    href_base = f'https://beta.metaspace2020.eu/annotations?ds={res.ds_id}&db_id={res.db_id}&sort=mz&fdr=0.5&q='
    df['ann_href'] = href_base + df.formula
    v = pd.DataFrame({
        'parent_formula': df[df.is_parent].set_index('hmdb_id').formula,
        'parent_n_detected': df.groupby('hmdb_id').is_detected.sum().astype(np.int32),
        'parent_n_frags': df.groupby('hmdb_id').in_range.sum().astype(np.int32),
        'parent_n_frags_unfiltered': df.groupby('hmdb_id').frag_idx.max().astype(np.int32),
        'mol_href': df.groupby('hmdb_id').formula.apply(lambda f: href_base + '|'.join(f)),
        'all_frag_formulas': pd.Series({
            hmdb_id: ','.join(fs) for hmdb_id, fs in df[df.in_range].sort_values('mz').groupby('hmdb_id').formula
        }),
    })
    df = df.drop(columns='all_frag_formulas')
    df = df.merge(v, how='left', left_on='hmdb_id', right_index=True)

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

    res.mols_df = df[df.is_parent & (df.msm > 0)][[
        'hmdb_id', 'mz', 'is_detected', 'is_expected', 'mol_name', 'formula', 'is_lipid',
        'msm', 'off_sample', 'intensity',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'mol_href', 'hmdb_href',
        'all_frag_formulas', 'pubchem_cid', 'smiles', 'inchikey',
    ]].set_index('hmdb_id')

    res.ann_mols_df = df[[
        'id', 'hmdb_id', 'mz', 'is_parent', 'is_detected', 'is_expected', 'mol_name', 'formula', 'is_lipid',
        'msm', 'off_sample', 'intensity', 'parent_intensity',
        'parent_formula', 'frag_idx', 'ann_href', 'mol_href', 'hmdb_href',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered',
    ]].set_index('id')

    res.anns_df = df[df.is_detected][[
        'formula', 'mz', 'msm', 'off_sample', 'intensity', 'ann_href',
    ]].drop_duplicates().set_index('formula')

    res.msms_df = (
        res.mols_df
        .reset_index()
        .groupby('all_frag_formulas')
        .apply(combine_mols)
        .set_index('hmdb_id', drop=False)
    )
# %%

def get_msms_results_for_ds(ds_id, mz_range=None, db_ver=None, include_lone_isotopic_peaks=False, use_cache=True):
    cache_name_parts = [
        ds_id,
        mz_range and f'{mz_range[0]}-{mz_range[1]}',
        db_ver,
        include_lone_isotopic_peaks and '1iso',
    ]
    cache_name = '_'.join(part for part in cache_name_parts if part)
    cache_path = Path(f'./scoring_results/cache/ds_results/{cache_name}.pickle')
    if not use_cache or not cache_path.exists():
        print(f'get_msms_results_for_ds({ds_id})')
        res = fetch_ds_results(ds_id)
        if mz_range is not None:
            res.name = f'{res.name} ({mz_range[0]}-{mz_range[1]})'
        add_result_dfs(res, *(mz_range or []), db_ver=db_ver, include_lone_isotopic_peaks=include_lone_isotopic_peaks)
        add_coloc_matrix(res)
        if mz_range:
            res.ds_coloc = res.ds_coloc.reindex(index=res.anns_df.index, columns=res.anns_df.index)
        res.ds_images = None  # Save memory/space as downstream analysis doesn't need this
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, cache_path.open('wb'))
    res = pickle.load(cache_path.open('rb'))
    return res


#%%