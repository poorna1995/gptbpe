


import torch
import torch.nn as nn

from tokenisazation import Tokenisation 
from positional_embedding import PositionalEmbedding 



input_embedding = torch.tensor([[[-1.0122, -0.6376, -1.5238,  1.1620],
         [ 0.2838,  2.0191, -0.1473, -0.3376],
         [-0.4041, -0.9644, -1.2488,  2.3606],
         [-0.4826, -4.2042, -1.9961,  0.5841],
         [-1.2461,  1.3416, -0.0549,  1.1404],
         [-0.7705, -1.3585,  1.8694,  2.8551],
         [ 0.9154,  3.1839, -1.0849,  1.6446],
         [ 1.3561,  0.6128,  1.6444,  1.8525],
         [ 2.4496, -0.9380, -1.5300,  1.4513],
         [-0.3746, -1.1202, -0.8350,  1.6626],
         [-0.3054, -2.2298,  1.0575,  4.0346],
         [ 0.7338,  0.0450, -3.1336,  1.8473],
         [-2.1387,  1.1609,  0.9993,  1.8886]]])


import torch.nn.functional as F

def SelfAttention1(input_embedding):
    # print(f'query_x2.shape: {query_x2.shape}')  
    print(f'input_embedding.shape: {input_embedding.shape}')  
    
    # Calculate attention scores
    attn_score = input_embedding @ input_embedding.transpose(1,2)
    print(f'attn_score.shape: {attn_score.shape}') 
    
    # Compute softmax to get attention weights
    attn_weights = F.softmax(attn_score, dim=-1)  # Shape: (1, seq_length)
    print(f'attn_weights.shape: {attn_weights[0]}')  
    contxt_vect =+ attn_weights @ input_embedding
    print(f'contxt_vect.shape: {contxt_vect.shape}') 
    return contxt_vect



output = SelfAttention1(input_embedding)
print(output)

