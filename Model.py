import torch
from math import sqrt as msqrt
from utils import gelu,get_pad_mask

import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, config, device):
        super(Embeddings, self).__init__()
        self.amino_emb = nn.Embedding(config.n_amino, config.d_embedding)
        self.struct_emb = nn.Embedding(config.n_struct,config.d_embedding)
        self.pos_emb = nn.Embedding(config.glossary_len,config.d_embedding)
        self.norm = nn.LayerNorm(config.d_embedding)
        self.dropout = nn.Dropout(config.p_dropout)
        self.device = device

    def forward(self, amino_seq, struct_seq):
        '''
        :param amino_seq: [batch, seq_len]
        :param struct_seq: [batch, seq_len]
        :return:  [batch, seq_len]
        '''
        # amino embedding
        amino_enc = self.amino_emb(amino_seq)
        # struct embedding
        struct_enc = self.struct_emb(struct_seq)
        # positional embedding
        pos = torch.arange(amino_seq.shape[1], dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(amino_seq)
        pos_enc = self.pos_emb(pos)

        Enc = self.norm(amino_enc + struct_enc + pos_enc) # Enc=encoding vector
        return self.dropout(Enc)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = config.d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)/ msqrt(self.d_k))
        # score: [batch, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(config.d_embedding, config.d_k*config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_embedding, config.d_k*config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_embedding, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.d_heads*config.d_v, config.d_embedding, bias=False)
        self.batch = config.batch_size
        self.n_heads = config.n_heads
        self.d_k = config.d_k

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch, seq_len, d_embedding]
        :param K:  [batch, seq_len, d_embedding]
        :param V:  [batch, seq_len, d_embedding]
        :param attn_mask:  [batch, seq_len, seq_len]
        :return:
        """
        batch = Q.size(0)
        per_Q = self.W_Q(Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context = ScaledDotProductAttention()(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1,2).contiguous().view(self.batch, -1,
                                                           self.n_heads*self.d_v)
        output = self.fc(context)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(config.d_embedding, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_embedding)
        self.dropout = nn.Dropout(config.p_dropout)
        self.gelu = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(config.d_embedding)
        self.norm2 = nn.LayerNorm(config.d_embedding)

        self.enc_attn = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x, pad_mask):
        """
        :param x: [batch, seq_len, d_embeddings]
        :param pad_mask:
        :return:
        """
        residual = x
        x = self.norm1(x)
        x = self.enc_attn(x, x, x, pad_mask) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x+residual

class Pooler(nn.Module):
    def __init__(self, config):
        super(Pooler, self).__init__()
        self.fc = nn.Linear(config.d_embedding, config.d_embedding)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        :param x: [batch, d_embedding]
        :return:
        """
        x = self.fc(x)
        x = self.tanh(x)
        return x

class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.embedding = Embeddings(config)
        self.encoders = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.pooler = Pooler(config)
        self.next_cls = nn.Linear(config.d_embedding, 2)
        self.gelu = gelu
        shared_weight = self.pooler.fc.weight
        self.gene_classifier = nn.Linear(config.d_embedding, config.max_mask, bias=False)
        self.gene_classifier.weight = shared_weight
        self.d_embedding = config.d_embedding

    def forward(self, amino_seq, struct_seq):
        output = self.embedding(amino_seq, struct_seq)
        enc_self_pad_mask = get_pad_mask(amino_seq)
        for layer in self.encoders:
            output = layer(output, enc_self_pad_mask)
        # output: [batch, seq_len, d_embedding]

        return output




