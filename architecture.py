import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math

from attention import d_model, num_heads, d_ff, dropout
from attention import MultiHeadAttention

#MLP -- feed forward network
class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.w_1(x)
        x = self.dropout(F.relu(x))
        x = self.w_2(x)
        
        return x
    
#encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, mask=None)
        self.MLP = MLP(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        attn_out, _ = self.multi_head_attention(x,x,x, mask)
        x = self.layer_norm1(x + self.dropout(attn_out))
        
        x = self.layer_norm2(x+self.dropout(self.MLP(x)))
        
        return x
        

#encoder
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, mask=None):        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
        

encoder = Encoder(d_model, num_heads, d_ff, dropout=0.1, num_layers=6)
encoder_out = encoder.forward(x, mask=None)

#decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.MLP = MLP(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        
        attn_out, _ = self.masked_attn(x,x,x, tgt_mask)
        x = self.layer_norm1(x + attn_out)
        
        #cross attention
        cross_out, _ = self.multi_head_attention(x, encoder_out, encoder_out, src_mask)
        x = self.layer_norm2(x + cross_out)
        
        x = self.layer_norm3(x + self.dropout(
            self.MLP(x)
        ))
        
        return x

#decoder 
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask, src_mask)
        return x
    
#positional encoding as per the paper
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe 
        return x
    
#embedding tables
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
#transformer itself
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, dropout = 0.1, num_layers=6):
        super().__init__()
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        src= self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, src_mask)
        out = self.linear(decoder_out)
        
        return out
    