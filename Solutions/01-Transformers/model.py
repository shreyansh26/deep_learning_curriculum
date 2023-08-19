import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    # From https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L18
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalAttentionBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, block_size, dropout_rate=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.attn_dropout = nn.Dropout(p=dropout_rate)
        self.out_dropout = nn.Dropout(p=dropout_rate)

        self.qkv_proj = nn.Linear(hidden_size, 3*hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mask_template = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def forward(self, x):
        '''
        Input
            x - (batch_size, seq_len, hidden_size)
        Output
            out - (batch_size, seq_len, hidden_size)
        '''
        batch_size, seq_len, hidden_size = x.shape
        d_head = hidden_size // self.num_heads

        q, k, v = self.qkv_proj(x).split(self.hidden_size, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2) # (batch_size, num_heads, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2) # (batch_size, num_heads, seq_len, d_head)
        v = v.view(batch_size, seq_len, self.num_heads, d_head).transpose(1, 2) # (batch_size, num_heads, seq_len, d_head)

        # att = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)
        att = torch.einsum('b h s d, b h S d -> b h s S', q, k) / math.sqrt(d_head)
        mask = (self.mask_template[:,:,:seq_len,:seq_len] == 0).to(att.device)
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # out = att @ v
        out = torch.einsum('b h s S, b h S d -> b h s d', att, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        
        return out

class MLPBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.fc_2 = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        '''
        Input 
            x - (batch_size, seq_len, hidden_size)
        Output
            out - (batch_size, seq_len, hidden_size)
        '''
        out = self.fc_1(x)
        out = self.gelu(out)
        out = self.fc_2(out)
        out = self.dropout(out)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, block_size, dropout_rate=0.0):
        super().__init__()

        self.ln_1 = LayerNorm(hidden_size, bias=False)
        self.ln_2 = LayerNorm(hidden_size, bias=False)

        self.attn_block = CausalAttentionBlock(num_heads, hidden_size, block_size, dropout_rate)
        self.mlp_block = MLPBlock(hidden_size, dropout_rate)

    def forward(self, x):
        '''
        Input 
            x - (batch_size, seq_len, hidden_size)
        Output
            out - (batch_size, seq_len, hidden_size)
        '''
        x = x + self.attn_block(self.ln_1(x))
        out = x + self.mlp_block(self.ln_2(x))
        return out
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_size, vocab_size, block_size, dropout_rate=0.0):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(block_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.Sequential(*[DecoderBlock(num_heads, hidden_size, block_size, dropout_rate) for _ in range(num_layers)])
        self.ln = LayerNorm(hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        """
        Input 
            x - (batch_size, seq_len)
        Output
            out - (batch_size, seq_len, vocab_size)
        """
        seq_len = x.shape[-1]

        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        pos_embedding = self.pos_embedding(pos)
        token_embedding = self.token_embedding(x)

        x = self.dropout(token_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln(x)
        out = self.lm_head(x)

        return out