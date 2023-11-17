# this part for transport parameters
import argparse
import subprocess

import torch.cuda

class Config:
    def __init__(self,paras):
        self.max_seq_len = paras['max_seq_len']
        self.glossary_len = paras['glossary_len']
        self.max_mask = paras['max_mask']
        self.d_k = paras['d_k']
        self.d_v = paras['d_v']
        self.d_embedding = paras['d_embedding']
        self.d_ff = paras['d_ff']
        self.n_amino = paras['n_amino']
        self.n_struct = paras['n_struct']
        self.n_heads = paras['n_heads']
        self.n_layers = paras['n_layers']
        self.p_dropout = paras['p_dropout']
        self.p_mask = paras['p_mask']
        self.p_replace = paras['p_replace']
        self.p_do_nothing = paras['p_do_nothing']
        self.batch_size = paras['batch_size']
        self.epochs = paras['epochs']
        self.lr = paras['lr']

class initailDevice:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.device_memory={}
        for i in range(self.num_gpus):
            self.device_memory[i]=torch.cuda.memory_allocated(i)
        self.sorted_gpus = sorted(self.device_memory.items(), key=lambda x: x[1])
        self.chosen_gpu = self.sorted_gpus[0][0]
