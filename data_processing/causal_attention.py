# casual attention also ckaled as a masked attention

# what does this do: it restricts the model to only consider previous and current input in a sequence , when processing any given token
# this is contrast to the self attetntion mechanism which allows  access to the entire 
# when computing attention score, the casual attention mechanism ensures that the model only fators in token that occur ar 0r before the current token in the sequence

# we mask ou the attention _weight above the diagonal, and we nomalize the non_masked attentionweights, such tht the attention weight, sum up to 1 in wach row

# to achieve this in gpr like llm for each token processes, we mack out the future tokens whic come after the current tomens



# diff imput va context

# imput_embedding = contains only semantic meaning of the input token, it does not contain any other information 
# like how the other in the sentence, and how much attention should pay of each the word when looking for a particular word

# contect vect - which contains bothe semantic menaning and how much other words should pay attention to that specific input token,
# how that particular words relate to all other words


# wq, wk, wv are the trainable weight matrix of each dimension . these are npt fixed , their parameter are need to be trined


# getting the attention weight and normaling



import torch
import torch.nn as nn
import math



class CasualAttention(nn.Module):
    def __init__(self, out_emb, out_dim, context_len, dropout, qkv_bias = False):
        super().__init__()
        self.w_q     = nn.Linear(out_emb,out_dim,bias = qkv_bias) # 4,3
        self.w_k     = nn.Linear(out_emb,out_dim, bias = qkv_bias)# 4,3
        self.w_v     = nn.Linear(out_emb,out_dim, bias = qkv_bias) # 4,3
        self.dropout = nn.Dropout(dropout) # 0.1
        self.registry_buffer('mask', torch.triu(torch.ones(ontext_len,context_len), diagonal = 1))
        
    
    def forward (self,x): # (batch_size , num_tokens, embedding_dim) @ (batch_size, embedding_dim, output_dim ) ---> (batch_size, num_tokens, output_dim)
        query  = w_q(x)    # 1,11,4 @1,4,3 ---> 1,11,3
        keys   = w_k(x)     # 1,11,4 @1,4,3 ---> 1,11,3
        values = w_v(x)     # 1,11,4 @1,4,3 ---> 1,11,3
        batch_size, num_tokens, out_emb = x.shape
        
        attn_score = query @ keys.transpose(1,2) # (1,11,3)@(1,3,11) --->1,11,11
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weight = torch.softmax(attn_score/math.sqrt(keys.shape[-1]),dim =-1 )
        attn_weight = self.dropout(attn_weight)
        context_vec  = attn_weight @ values # (1,11,11)@(1,11,3)-->1,11,3
        return context_vec
        

torch.manual_seed(1)

input_embedding = torch.tensor([[[-0.2566,  0.0918, -3.3868,  0.0239],
         [ 0.3491, -1.8707, -2.0376, -1.7278],
         [-0.9179, -0.7690, -1.2165,  1.7341],
         [-1.3919, -3.0176, -1.1270,  4.1750],
         [ 1.1861,  2.3395, -0.9940,  0.5740],
         [-0.6740, -0.6381,  2.4336,  0.1265],
         [ 1.1774,  2.1602, -1.5564,  1.9314],
         [ 0.3351,  2.2418,  0.0489,  1.8005],
         [ 3.5250, -2.3766,  1.6663,  1.9996],
         [ 2.7091,  1.7903, -1.6857,  1.6448],
         [-1.7731, -1.9442, -0.6540, -0.6489]]])


out_emb = 4
out_dim = 3

context_len = input_embedding.shape[1]
casual_attn = CasualAttention(out_emb, out_dim,context_len,0.1)

cnotect_weights = casual_attn(input_embedding)
cnotect_weights