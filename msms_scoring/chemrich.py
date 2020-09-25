import os
import subprocess
import contextlib
import pandas as pd
from pathlib import Path

from msms_scoring.metrics import get_ds_results, get_many_ds_results

CHEM_RICH_DIR = '/home/lachlan/dev/chemrich'

@contextlib.contextmanager
def run_in_dir(path):
    cwd = os.getcwd()
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def get_df_for_chemrich2(ds_id):
    res = get_ds_results(ds_id)
    df = res.msms_df[['mol_name','pubchem_cid','smiles','coloc_int_fdr','coloc_int_fc']]
    df = df[df.coloc_int_fc > 0]
    df = df[df.coloc_int_fdr < 1]
    df['coloc_int_fdr'] = df.coloc_int_fdr / 2
    df = df.drop_duplicates('pubchem_cid')
    df = df.rename(columns={
        'mol_name': 'compound_name',
        'pubchem_cid': 'pubchem_id',
        'smiles': 'smiles',
        'coloc_int_fdr': 'pvalue',
        'coloc_int_fc': 'effect_size',
    })
    return df


def get_summary_df_for_chemrich2(ds_ids):
    def get_median_fdr_fc(grp):
        grp = grp.sort_values('coloc_int_fdr')
        mid_idx = len(grp) // 2
        if len(grp) % 2 == 0:
            mid = grp.iloc[mid_idx:mid_idx + 2]
        else:
            mid = grp.iloc[mid_idx:mid_idx + 1]
        return pd.Series({
            'median_fdr': mid.coloc_int_fdr.mean(),
            'median_fc': mid.coloc_int_fc.mean(),
            'count': len(grp),
        })
    results_dfs = []
    for res in get_many_ds_results(ds_ids):
        results_dfs.append(res.mols_df.assign(ds_name=res.name, polarity=res.sm_ds.polarity.lower()))

    results_df = pd.concat(results_dfs)
    results_df = results_df[results_df.coloc_int_fc > 0]
    results_df = results_df[results_df.coloc_int_fdr < 1]
    mol_median_scores = results_df.groupby('pubchem_cid')[['coloc_int_fdr', 'coloc_int_fc']].apply(get_median_fdr_fc)
    df = (
        results_df
        # [['mol_name','inchikey','pubchem_cid','smiles','all_frag_formulas']]
        .merge(mol_median_scores, left_on='pubchem_cid', right_index=True)
        # ChemRICH doesn't have a configurable minimum p-value threshold, so scale the FDRs
        # to align our desired threshold to their imposed threshold
        .assign(median_fdr=lambda df: df.median_fdr / 2)
        .sort_values('median_fdr')
        .sort_values('count', ascending=False, kind='mergesort')
        # Some HMDB IDs translated to the same CID, e.g. D-Fructose/L-Sorbose both get CID 92092.
        # Keep the best option based on number of times annotated
        .drop_duplicates('pubchem_cid')
        .rename(columns={
            'mol_name': 'compound_name',
            'pubchem_cid': 'pubchem_id',
            'smiles': 'smiles',
            'median_fdr': 'pvalue',
            'median_fc': 'effect_size',
        })
    )
    return df


def run_chemrich(df: pd.DataFrame, output_name: str):
    outdir = Path(f'scoring_results/chemrich/{output_name}').resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tempdir: Path
    with run_in_dir('./chemrich_temp') as tempdir:
        # MeSH prediction to get chemical classes
        df[['compound_name', 'pubchem_id', 'smiles']].to_excel('to_mesh.xlsx', index=False)
        subprocess.run([
            'R',
            '-e',
            f'''
source("{CHEM_RICH_DIR}/predict_mesh_chemical_class.R")
load.ChemRICH.Packages()
# Code is broken. Load treenames.df as a global variable to fix it.
if(!file.exists("treenames.df.RData")) {{
    load("{CHEM_RICH_DIR}/treenames.df.RData")
    save(treenames.df, file = "treenames.df.RData")
}}
load("treenames.df.RData")
predict_mesh_classes(inputfile = "to_mesh.xlsx")
            '''
        ])
        mesh_df = pd.read_excel('MeSh_Prediction_Results.xlsx')
        mesh_df = mesh_df[['pubchem_id', 'ClusterNumber', 'MeSH_Class']].rename(columns={
            'ClusterNumber': 'order',
            'MeSH_Class': 'set',
        })
        df = df.merge(mesh_df, on='pubchem_id', how='left')
        df = df[df['set'] != '']

        # ChemRICH for Chemical Classes
        df[['compound_name', 'smiles', 'pvalue', 'effect_size', 'order', 'set']].to_excel('to_chem.xlsx', index=False)
        df[['compound_name', 'smiles', 'pvalue', 'effect_size', 'order', 'set']].to_excel(outdir / 'input data.xlsx', index=False)
        subprocess.run([
            'R',
            '-e',
            f'''
                source("{CHEM_RICH_DIR}/chemrich_chemical_classes.R")
                load.ChemRICH.Packages()
                run_chemrich_chemical_classes(inputfile = "to_chem.xlsx")
            '''
        ])
        impact_plot = (tempdir / 'chemrich_class_impact_plot.png')
        if impact_plot.exists():
            impact_plot.replace(outdir / 'class_impact_plot.png')
        results_file = (tempdir / 'chemRICH_class_results.xlsx')
        if results_file.exists():
            results_file.replace(outdir / 'class_results.xlsx')

        # ChemRICH for any set definition
        subprocess.run([
            'R',
            '-e',
            f'''
                source("{CHEM_RICH_DIR}/chemrich_minimum_analysis.R")
                load.ChemRICH.Packages()
                run_chemrich_basic(inputfile = "to_chem.xlsx")
            '''
        ])
        impact_plot = (tempdir / 'chemrich_class_impact_plot.png')
        if impact_plot.exists():
            impact_plot.replace(outdir / 'any_impact_plot.png')
        results_file = (tempdir / 'chemRICH_class_results.xlsx')
        if results_file.exists():
            results_file.replace(outdir / 'any_results.xlsx')





