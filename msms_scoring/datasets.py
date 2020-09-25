import pandas as pd

datasets_df = pd.read_csv('input/datasets.csv')

# Lists of METASPACE IDs of datasets that have been processed with a fragmentation database
dataset_ids = datasets_df.ds_id[datasets_df.set == 'spotting'].to_list()
whole_body_ds_ids = datasets_df.ds_id[datasets_df.set == 'whole_body'].to_list()
high_quality_ds_ids = datasets_df.ds_id[datasets_df.set == 'high_quality'].to_list()


# Optional aliases for datasets that have long names
dataset_aliases = datasets_df.set_index('ds_id').name.to_dict()
dataset_polarity = datasets_df.set_index('ds_id').polarity.to_dict()
dataset_matrix = datasets_df.set_index('ds_id').matrix.to_dict()
dataset_ms_mode = {
    '2020-08-03_13h23m33s': 'MS1',
    '2020-08-03_19h56m47s': 'AIF',
    '2020-08-04_14h06m33s': 'MS1',
    '2020-08-04_14h06m47s': 'AIF',
    '2020-08-11_15h28m01s': 'MS1',
    '2020-08-11_15h28m27s': 'AIF',
    '2020-08-11_15h28m38s': 'MS1',
    '2020-08-11_16h49m43s': 'AIF',
}

msms_mol_ids = set(pd.read_csv('./spotting/msms_spotted_mols.csv').hmdb_id)

# Optional lists mapping dataset IDs to sets of HMDB IDs for molecules that are expected
dataset_mol_lists = {
    '2020-08-03_13h23m33s': msms_mol_ids,
    '2020-08-03_19h56m47s': msms_mol_ids,
    '2020-08-04_14h06m33s': msms_mol_ids,
    '2020-08-04_14h06m47s': msms_mol_ids,
    '2020-08-11_15h28m01s': msms_mol_ids,
    '2020-08-11_15h28m27s': msms_mol_ids,
    '2020-08-11_15h28m38s': msms_mol_ids,
    '2020-08-11_16h49m43s': msms_mol_ids,
}