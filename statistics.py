# Count the structure and codons of proteins
import json
from collections import Counter

path = '/data/yujie/Ebert/pre_dataset_corrected.json'
outPath = '/data/yujie/Ebert/statistic.txt'

with open(path, 'r') as file:
    data = json.load(file)

proteins = data.keys()

amino_seqs = ''
structures = ''
gene_seqs = ''

for protein in proteins:
    protein_instance = data[protein]

    amino_seq = str(protein_instance['amino seq'])
    structure = str(protein_instance['structure'])
    gene_seq = str(protein_instance['gene seq'])

    amino_seqs = amino_seqs + amino_seq
    structures = structures + structure
    gene_seqs = gene_seqs + gene_seq

amino_seqs = list(amino_seqs)
structures = list(structures)
codon_seq = [gene_seqs[i:i + 3] for i in range(0, len(gene_seqs), 3)]

statistic_list = list(zip(amino_seqs, structures, codon_seq))
element_counts = Counter(statistic_list)
element_counts = str(element_counts)

with open(outPath,'w') as file:
    file.write(element_counts)

