# utils for model
import random
import datetime
import logging
import torch
import json
from math import sqrt as msqrt

amino_token_list = {
    "F":1, "L":2,"I":3, "M":4, "V":5, "P":6, "T":7, "A":8, "Y":9, "H":10, "Q":11, "N":12,
    "K":13, "D":14, "E":15, "C":16, "S":17, "R":18, "G":19, "W":20, "*":21
}

struct_token_list = {
    "H":1, "B":2, "E":3, "G":4, "I":5, "P":6, "T":7, "S":8, "X":9, "F":10
}

gene_token_list = {
    "ttt":1, "ttc":2, "tta":3, "ttg":4, "ctt":5, "ctc":6, "cta":7, "ctg":8, "att":9, "atc":10,
    "ata":11, "atg":12, "gtt":13, "gtc":14, "gta":15, "gtg":16, "cct":17, "ccc":18,
    "cca": 19, "ccg":20, "act":21, "acc":22, "aca":23, "acg":24, "gct":25, "gcc":26,
    "gca":27 ,"gcg":28, "tat":29, "tac":30, "cat":31, "cac":32, "caa":33, "cag":34,
    "aat":35, "aac":36, "aaa":37, "aag":38, "gat":39, "gac":40, "gaa":41, "gag":42,
    "tgt":43, "tgc":44, "agt":45, "agc":46, "tct":47, "tcc":48, "tca":49, "tcg":50,
    "aga":51, "agg":52, "cgt":53, "cgc":54, "cga":55, "cgg":56, "ggt":57, "ggc":58,
    "gga":59, "ggg":60, "tgg":61, "tta":62, "tag": 63, "tga":64
}

def gelu(self,x):
    return 0.5*x*(1.+torch.erf(x/msqrt(2.)))

def get_pad_mask(tokens, pad_idx=0):
    """
    :param tokens: [batch, seq_len]
    :param pad_idx:
    :return:
    """
    batch, seq_len = tokens.size()
    pad_mask = tokens.data.eq(pad_idx).unsqueeze(1)
    pad_mask = pad_mask.expand(batch, seq_len, seq_len)

    return pad_mask

def pad(ids, n_pads, pad_sym=0):
    return ids.extend([pad_sym for _ in range(n_pads)])

def  make_data(dataset_dir, shuffle=False, start_time=None):
    with open(dataset_dir, 'r') as file:
        data = json.load(file)

    if shuffle == True:
        random.shuffle(data)
        filepath = f"/data/yujie/Ebert/Dataset/shuffled_{start_time}.json"
        with open(filepath, "w") as file:
            json.dump(data, file)

    total_sample = len(data)
    train_size = int(0.8*total_sample)
    valid_size = int(0.1*total_sample)
    test_size = total_sample - train_size - valid_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    trainset=[]
    valset=[]
    testset=[]
    train_aminos=[]
    valid_aminos=[]
    test_aminos=[]
    train_genes=[]
    valid_genes=[]
    test_genes=[]
    train_structs=[]
    valid_structs=[]
    test_structs=[]

    for entry in train_data:
        train_aminos.append(entry.get("amino seq", ''))
        train_genes.append(entry.get("gene seq", ''))
        train_structs.append(entry.get("structure", ''))
    trainset.append(train_aminos)
    trainset.append(train_genes)
    trainset.append(train_structs)

    for entry in valid_data:
        valid_aminos.append(entry.get("amino seq", ''))
        valid_genes.append(entry.get("gene seq", ''))
        valid_structs.append(entry.get("structure", ''))
    valset.append(valid_aminos)
    valset.append(valid_genes)
    valset.append(valid_structs)

    for entry in test_data:
        test_aminos.append(entry.get("amino seq", ''))
        test_genes.append(entry.get("gene seq", ''))
        test_structs.append(entry.get("structure", ''))
    testset.append(test_aminos)
    testset.append(test_genes)
    testset.append(test_structs)

    return trainset, valset, testset

def tokenlize(aminos, genes, structs):
    num_instance = len(aminos)
    max_len_amino = max(len(amino) for amino in aminos)

    token_aminos=[]
    token_genes=[]
    token_structs=[]

    for i in range(num_instance):
        token_amino = []
        token_gene = []
        token_struct = []

        gene = genes[i]
        codons = [gene[j:j + 3] for j in range(0, len(gene), 3)]

        token_amino.append(amino_token_list.get(amino) for amino in
                           aminos[i])
        token_gene.append(gene_token_list.get(codon, codon) for codon in codons)
        token_struct.append(struct_token_list.get(struct, struct) for struct in structs[i])

        pad(token_amino, max_len_amino-len(token_amino))
        pad(token_gene, max_len_amino-len(token_gene))
        pad(token_struct, max_len_amino-len(token_struct))

        token_aminos.append(token_amino)
        token_genes.append(token_gene)
        token_structs.append(token_struct)

    return token_aminos, token_genes, token_structs

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger





