
import torch.nn as nn
import torch
import math

from tokenisazation import Tokenisation 

class PositionalEmbedding(nn.Module):
    def __init__ (self, token_ids, out_dim, drop_out):
        super().__init__()
        self.len_token_ids = len(token_ids)
        self.out_dim   = out_dim
        self.drop_out  = nn.Dropout(drop_out)
        pe = torch.zeros(self.len_token_ids,self.out_dim)
        position = torch.arange(0,self.len_token_ids,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.out_dim, 2).float() * (-math.log(10000.0) / out_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 

        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False) 
        return self.drop_out(x)
 
 
text = """hello , I'm Poorna Praneesha!"""
out_dim = 4
drop_out = 0.1  # Example dropout rate
torch.manual_seed(1)
tokenisation = Tokenisation(text, out_dim)  # Assuming this class is defined
token_ids = tokenisation()

positional_embedding = PositionalEmbedding(token_ids, out_dim, drop_out)

# Example tensor input (assuming it comes from the tokenization process)
x = torch.randn(1, len(token_ids), out_dim)

# Forward pass through the positional embedding
output = positional_embedding(x)

print(output.shape)


def InputEmbedding(output,token_ids):
    return output + token_ids

Input_embedding= InputEmbedding(output,token_ids)
print(Input_embedding)

