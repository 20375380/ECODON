# make sequence for training from database.json
import os
import json

database_path = '/data/yujie/output/database.json'
save_path = '/data/yujie/Ebert/pre_dataset.json'

with open(database_path, 'r') as file:
    data = json.load(file)

pdb_ids = data.keys()
training_set = {}

i=0
for pdb_id in pdb_ids:
    dict_pdb = data[pdb_id]
    keys = dict_pdb.keys()
    amino_seq = dict_pdb['amino seq']
    gene_seq = dict_pdb['gene seq']

    amino_len = len(amino_seq)
    gene_len = len(gene_seq)
    if 3 * amino_len == gene_len:
        for key in keys:
            if key.startswith('structure'):
                structure = dict_pdb[key]
                instance = {
                    'amino seq': amino_seq,
                    'structure': structure,
                    'gene seq': gene_seq
                }
                training_set[i] = instance
                i += 1

with open(save_path, 'w') as file:
    json.dump(training_set, file, indent=i)


