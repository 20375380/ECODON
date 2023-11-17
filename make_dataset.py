# create training , testing and validating dataset from json file
# where training:testing:validating=8:1:1
import os
import json
import torch
from torch.utils.data import Dataset

json_path = '/data/yujie/pre_dataset.json'

with open(json_path, 'r') as file:
    data = json.load(file)

word2idx = {f'[{name}]': idx for idx, name in enumerate(['PAD', 'CLS', 'MASK'])}
# {'[PAD]':0, '[CLS]':1, '[MASK]':2}

amino_seq_list = [entity['amino seq'] for entity in data.values()]
struct_list = [entity['structure'] for entity in data.values()]
gene_list = [entity['gene seq'] for entity in data.values()]
for entity in data.values():
    gene_seq = entity['gene seq']
    codon_list = 

codon_list = [gene_list[i:i + 3] for i in range(0, len(gene_seqs), 3)]

amino_list = list(set("".join(amino_seq_list).split()))
struct_list = list(set("".join(struct_list).split()))

def make_data(sentence, n_data):
    batch_data = []
    positive = negative = 0


class proGenDataset(Dataset):
    def __init__(self, amino_seq, structure_seq, gene_seq):
        self.aminoSeq = amino_seq
        self.strucSeq=structure_seq
        self.geneSeq=gene_seq

    def __len__ (self):
        return len(self.aminoSeq)

    def __getitem__(self, index):
        sample = {
            'aminoSeq': torch.tensor(self.data[index], dtype=torch.float32),
            'structSeq': torch.tensor(self.strucSeq[index], dtype=torch.float32),
            'geneSeq': torch.tensor(self.geneSeq[index], dtype=torch.long)
        }
        return sample

def makeData(amino_seq, structure_seq, gene_seq)


