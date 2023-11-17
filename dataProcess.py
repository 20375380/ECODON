# make dataset
import os
import json
import sqlite3
import re


protein_folder_path = "/data/yujie/protein/proteins"
dssp_folder_path = "/data/yujie/protein/structure"
output_json = '/data/yujie/output/database.json'
ff_path = '/data/yujie/oct24th/en.db'

protein_file_names = os.listdir(protein_folder_path)
dssp_file_name = os.listdir(dssp_folder_path)

# 打开数据库文件
conn = sqlite3.connect(ff_path)
cursor = conn.cursor()

with conn:
    ffTable = cursor.execute('select * from finalTable')
df = ffTable.fetchall()

# pdb_pattern = re.compile('([A-Z0-9]{4})_[0-9]+')
data_dict = {}
count = 1

for (pdbid, organism, expression_sys, pseq, mutation, ncbi_id, source, gseq) in df:

    if pdbid not in data_dict.keys():
        data_dict[pdbid] = {
            'organism': organism,
            'expression system': expression_sys,
            'amino seq': pseq,
            'mutation': mutation,
            'ncbi id': ncbi_id,
            'source': source,
            'gene seq': gseq
        }

    count += 1

dict_keys = data_dict.keys()

structure_key = ['structure_1', 'structure_2', 'structure_3', 'structure_4', 'structure_5']

def extract_info_dssp(dssp_file_path):
    '''从dssp文件中提取位置，链，原子，结构信息'''
    with open(dssp_file_path, 'r') as dssp_file:
        lines = dssp_file.readlines()

    # 提取氨基酸序列和对应的结构
    ami_poss = []
    chains = []
    amino_acids = []
    structures = []
    i = 0
    for line in lines:
        i = i + 1
        if i > 28:
            ami_pos = line[7:10].strip()
            chain = line[11]
            amino_acid = line[13:14].strip()
            structure = line[16]
            if amino_acid != '!' and amino_acid != '!*':
                ami_poss.append(ami_pos)
                chains.append(chain)
                amino_acids.append(amino_acid)
                structures.append(structure if structure != ' ' else 'F')
    ami_poss = [int(x) for x in ami_poss]
    # chains = ''.join(chains)
    # amino_acids = ''.join(amino_acids)
    # structures = ''.join(structures)

    return ami_poss, chains, amino_acids, structures

for protein_file_name in protein_file_names:

    pdb_id, _ = os.path.splitext(protein_file_name)  # 9xia
    pdb_id_dssp = pdb_id + '.dssp'

    if pdb_id_dssp in dssp_file_name:

        protein_file_path = os.path.join(protein_folder_path, protein_file_name)
        dssp_file_path = os.path.join(dssp_folder_path, pdb_id_dssp)
        pdb_id_cap = pdb_id.upper()

        entities = []
        for key in dict_keys:
            if key.startswith(pdb_id_cap):
                entities.append(key)

        ami_poss, chains, amino_acids, structures = extract_info_dssp(dssp_file_path)

        chain_set=set(chains)
        with open('/data/yujie/log.txt','w') as file:
            chain_set_str = str(chain_set)
            file.write(chain_set_str)
        for chain in chain_set:
            indices_of_chain = [i for i, value in enumerate(chains) if value == chain]# 找到chains中同一链上的索引
            values_in_amino_acids = [amino_acids[i] for i in indices_of_chain]
            values_in_ami_poss = [ami_poss[i] for i in indices_of_chain]
            values_in_structures = [structures[i] for i in indices_of_chain]

            # if chain == 'B':
            #     with open('/data/yujie/log.txt', 'w') as file:
            #         values_in_ami_poss_str = str(values_in_ami_poss)
            #         file.write(values_in_ami_poss_str)
            #     with open('/data/yujie/STRCT.txt', 'w') as file:
            #         values_in_structures_str = str(values_in_structures)
            #         file.write(values_in_structures_str)

            # 按照氨基酸索引生成结构列表
            for entity_id in entities:
                entity = data_dict[entity_id]
                seq_of_entity = entity['amino seq']
                seq_of_entity_list = [letter for letter in seq_of_entity]
                dssp_amino_seq = values_in_amino_acids[0:9]
                dssp_amino_seq = ''.join(dssp_amino_seq)

                if dssp_amino_seq in seq_of_entity:
                    # 检查entity是否存在其他结构
                    for structure_id in structure_key:
                        if structure_id not in entity.keys():
                            j=0
                            structure_list = []
                            for i,_ in enumerate(seq_of_entity_list):
                                if (i-1) in values_in_ami_poss:
                                    structure_list.append(values_in_structures[j])
                                    j += 1
                                else:
                                    structure_list.append('X')
                            structure_list = ''.join(structure_list)
                            entity[structure_id] = structure_list
                            break

                data_dict[entity_id] = entity# 更新entity


with open(output_json, 'w') as json_file:
    json.dump(data_dict, json_file, indent=count)