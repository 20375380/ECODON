from torch.utils.data import  Dataset
from utils import tokenlize

class EBertDataset(Dataset):
    def __init__(self, dataset):
        self.aminos=dataset[0]
        self.genes=dataset[1]
        self.structs=dataset[2]
        self.token_aminos, self.token_genes, self.token_structs = tokenlize(
            self.aminos, self.genes, self.structs
        )

    def __len__(self):
        return len(self.aminos)

    def __getitem__(self, idx):
        amino_seq = self.token_aminos[idx]
        struct_seq = self.token_structs[idx]
        gene_seq = self.token_genes[idx]

        return amino_seq, struct_seq, gene_seq




