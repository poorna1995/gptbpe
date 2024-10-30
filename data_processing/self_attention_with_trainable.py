
import torch
import torch.nn as nn
import math

class  SelfAttention(nn.Module):
  def __init__(self,out_emb,out_dim ):
    super().__init__()
    self.w_q = nn.Parameter(torch.rand(out_emb,out_dim)) # 4,3
    self.w_k = nn.Parameter(torch.rand(out_emb,out_dim)) # 4,3
    self.w_v = nn.Parameter(torch.rand(out_emb,out_dim)) # 4,3
    
  def forward(self, x):
    query = x @ self.w_q # 1,11,4 @  4,3 ---> 1,11,3
    key   = x @ self.w_k # 1,11,4 @  4,3 ---> 1,11,3
    value = x @ self.w_v # 1,11,4 @  4,3 ---> 1,11,3
    
    attn_scores  = query @ key.transpose(1,2)  # 1,11,3 @ 1,3,11---> 1,11,11
    
    # why sqrt : to male the variance of the dot product stable. 
    # the dot prodcut of Q and K  incersea the vairance because multiplying two random numbers increse the variance
    # The increase in variance grows witesnionh the
    # divind th by srt (dimension ) keeps the variance close to 1
    
    scaled_vect  = attn_scores/math.sqrt(key.shape[-1]) # ---> 1,11,11
    attn_weights = torch.softmax(scaled_vect ,dim=-1) # ---> 1,11,11
    
    context_vec  =  attn_weights @ value # -- >1,11,11 @ 1,11,3 --->1,11,3
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


torch.manual_seed(1)

out_emb = 4
out_dim = 3

self_attention = SelfAttention(out_emb,out_dim)
values= self_attention(input_embedding)
print(f'values: {values}')
print(f'values.shape: {values.shape}')






# def SelfAttention(input_embedding,out_emb,out_dim):
#     w_q = nn.Linear(in_features = out_emb,out_features = out_dim, bias=False)
#     w_k = nn.Linear(in_features = out_emb,out_features = out_dim, bias=False)
#     w_v = nn.Linear(in_features = out_emb,out_features = out_dim, bias=False)
    
#     query = w_q(input_embedding) # 4,3 @ 1,11,4 ---> 1,11,3
#     key   = w_k(input_embedding) # 4,3
#     value = w_v(input_embedding) # 4,3
#     print(f'query:{query.shape}')
    
#     attn_score = query @ key.transpose(1,2) # 1,11,3 @ 1,3,11 --> 1,11,11 
#     print(f'attn_score:{attn_score.shape}')
  
#     scaled = attn_score / math.sqrt(out_dim) # 1,11,11 
#     attn_weights = torch.softmax(scaled, dim =-1) # 1,11,11 
#     # 
#     context_vec =  attn_weights @ value  #---> 1,11,11 @ 1,11,3 --->1,11,3
#     return context_vec