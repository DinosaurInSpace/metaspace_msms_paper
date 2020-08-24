import pandas as pd

# List of METASPACE IDs of datasets that have been processed with a fragmentation database
dataset_ids = [
    '2020-08-03_13h23m33s',
    '2020-08-03_19h56m47s',
    '2020-08-04_14h06m33s',
    '2020-08-04_14h06m47s',
    '2020-08-11_15h28m01s',
    '2020-08-11_15h28m27s',
    '2020-08-11_15h28m38s',
    '2020-08-11_16h49m43s',
]

# Optional aliases for datasets that have long names
dataset_aliases = {
    '2020-08-03_13h23m33s': 'MSMS DHB+ MS1 (100-800)',
    '2020-08-03_19h56m47s': 'MSMS DHB+ AIF (100-800)',
    '2020-08-04_14h06m33s': 'MSMS DHB- MS1 (100-800)',
    '2020-08-04_14h06m47s': 'MSMS DHB- AIF (100-800)',
    '2020-08-11_15h28m01s': 'MSMS DAN- MS1 (100-800)',
    '2020-08-11_15h28m27s': 'MSMS DAN- AIF (100-800)',
    '2020-08-11_15h28m38s': 'MSMS DAN+ MS1 (100-800)',
    '2020-08-11_16h49m43s': 'MSMS DAN+ AIF (100-800)',
}
dataset_aliases2 = {
    '2020-08-03_13h23m33s': 'DHB Pos MS1',
    '2020-08-03_19h56m47s': 'DHB Pos AIF',
    '2020-08-04_14h06m33s': 'DHB Neg MS1',
    '2020-08-04_14h06m47s': 'DHB Neg AIF',
    '2020-08-11_15h28m01s': 'DAN Neg MS1',
    '2020-08-11_15h28m27s': 'DAN Neg AIF',
    '2020-08-11_15h28m38s': 'DAN Pos MS1',
    '2020-08-11_16h49m43s': 'DAN Pos AIF',
}
dataset_polarity = {
    '2020-08-03_13h23m33s': 'positive',
    '2020-08-03_19h56m47s': 'positive',
    '2020-08-04_14h06m33s': 'negative',
    '2020-08-04_14h06m47s': 'negative',
    '2020-08-11_15h28m01s': 'negative',
    '2020-08-11_15h28m27s': 'negative',
    '2020-08-11_15h28m38s': 'positive',
    '2020-08-11_16h49m43s': 'positive',
}

msms_mol_ids = set(pd.read_csv('./spotting/msms_spotted_mols.csv')[lambda df: df.cm_name.notna()].hmdb_id)

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