import json, time, ast
import pandas as pd
from metaspace.sm_annotation_utils import get_config, GraphQLClient
# from metaspace.sm_annotation_utils import SMInstance
from msms_scoring.datasets import dataset_ids

# %% Make DB

for p in ['positive', 'negative']:
    (
        pd.concat([
            pd.read_csv('to_metaspace/cm3_msms_all_pos_v3.csv', sep='\t').assign(polarity='positive'),
            pd.read_csv('to_metaspace/cm3_msms_all_neg_v3.csv', sep='\t').assign(polarity='negative'),
            pd.read_csv('to_metaspace/cm3_msms_all_pos_v2.csv', sep='\t').assign(polarity='positive'),
            pd.read_csv('to_metaspace/cm3_msms_all_neg_v2.csv', sep='\t').assign(polarity='negative'),
            pd.read_pickle('to_metaspace/cm3_msms_all_both.pickle'),
        ])
        [lambda df: df.polarity == p]
        [['id', 'name', 'formula']]
            .drop_duplicates()
            .to_csv(f'to_metaspace/ls_cm3_msms_all_{p[:3]}_v3.csv', sep='\t', index=False)
    )

#%% Reprocess DSs

def update_db(ds_id_in):
    beta_gql = GraphQLClient(get_config(host='https://beta.metaspace2020.eu'))

    result = beta_gql.query(
        """
        query editDatasetQuery($id: String!) {
          dataset(id: $id) {
            id
            name
            metadataJson
            configJson
            isPublic
            inputPath
            group { id }
            submitter { id }
            principalInvestigator { name email }
            molDBs
            adducts
          }
        }
        """,
        {'id': ds_id_in}
    )
    ds = result['dataset']
    config = json.loads(ds['configJson'])
    metadata = json.loads(ds['metadataJson'])

    is_pos = metadata['MS_Analysis']['Polarity'] == 'Positive'
    new_dbs = list(
        set(ds['molDBs'])
        .difference(['ls_cm3_msms_all_pos_v2', 'ls_cm3_msms_all_neg_v2'])
        .union({'ls_cm3_msms_all_pos_v3'} if is_pos else {'ls_cm3_msms_all_neg_v3'})
    )
    new_adducts = ['[M]+'] if is_pos else ['[M]-']
    assert ds['adducts'] == ['[M]+'] or ds['adducts'] == ['[M]-'], (ds['id'], is_pos, ds['adducts'], new_adducts, ds['name'])

    result = beta_gql.query(
        """
        mutation ($id: String!, $input: DatasetUpdateInput!) {
          updateDataset(id: $id, input: $input, reprocess: true, delFirst: true)
        }
        """,
        {
            'id': ds_id_in,
            'input': {
                'molDBs': new_dbs,
                'adducts': new_adducts,
            }
        }
    )
    print(ds_id_in, result)
    # In case you put this in a loop to copy multiple datasets, always sleep 1 second between new datasts
    # or else METASPACE may throw an error

    # Should print and return new dsid on betaserver
    # ds_id_out = dict(result)['createDataset']
    # ds_id_out = ds_id_out.split(":")[1].split(",")[0].replace('"', '')
    # output = (ds_id_in, ds_id_out)
    # print(output)
    # return output

for ds_id in dataset_ids:
    update_db(ds_id)

