
import torch
import torch.nn as nn
import math

# The `SelfAttention` class implements a self-attention mechanism using linear transformations for
# query, key, and value matrices to calculate attention scores and context vectors.
class  SelfAttention(nn.Module):
  def __init__(self,out_emb,out_dim ):
    super().__init__()
    self.w_q = nn.Linear(out_emb,out_dim,bias = False) # 4,3
    self.w_k = nn.Linear(out_emb,out_dim, bias = False)# 4,3
    self.w_v = nn.Linear(out_emb,out_dim, bias = False) # 4,3
    
  def forward(self, x):
    query = self.w_q(x)# 1,11,4 @  4,3 ---> 1,11,3
    key   = self.w_k(x)# 1,11,4 @  4,3 ---> 1,11,3
    value = self.w_v(x)# 1,11,4 @  4,3 ---> 1,11,3
    
    attn_scores  = query @ key.transpose(1,2)  # 1,11,3 @ 1,3,11---> 1,11,11
    
    # why sqrt : to male the variance of the dot product stable. 
    # the dot prodcut of Q and K  incersea the vairance because multiplying two random numbers increse the variance
    # The increase in variance grows witesnionh the
    # divind th by srt (dimension ) keeps the variance close to 1
    
    scaled_vect  = attn_scores/math.sqrt(key.shape[-1]) # ---> 1,11,11
    attn_weights = torch.softmax(scaled_vect ,dim=-1) # ---> 1,11,11
    
    context_vec  =  attn_weights @ value # -- >1,11,11 @ 1,11,3 --->1,11,3
    return context_vec
    

# why do we use the terms : keys, query, and value

# query : analogos to search query in a datanase, it represents the current token the model focus on
# key :  the attention mechanism, each item in inut sequence has a key. keys to used to match with the query
#  value : it represnrs the actial content or representaoin of the input items. once the model determine which keys 
# (which parts of the input) are most relvant to th equery (current focus items), it retrives the corresponding values









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




# calsula attention

queries = self_attention.w_q(input_embedding) # 1,11,4 @  4,3 ---> 1,11,3
keys    = self_attention.w_k(input_embedding)
values    = self_attention.w_v(input_embedding)
attn_scores = queries @ keys.transpose(1,2)
attn_weights = torch.softmax(attn_scores,dim=-1) # 1,11,3 @ 1,3,11---> 1,11,11
print(f' casual masked attn_weights: {attn_weights}')

context_len = attn_scores.shape[1]
mask_simple = torch.tril(torch.ones(context_len,context_len))

print(mask_simple)
casual_masked = attn_weights * mask_simple #---> 1,11,11 # 11,11 => 11,11
print(casual_masked)

row_sum  = casual_masked.sum(dim=-1, keepdim =True)
casual_masked_norm = casual_masked/row_sum
print(casual_masked_norm)


# attnt_score ---> 1. apply softmax ----> attention weit(normalied) --->mask with 0's above diagonal -->mask with rows -->masked attention weight (normalised)
# 





mask = torch.triu(torch.ones(context_len,context_len),diagonal =1)
masked = attn_scores.masked_fill(mask.bool(),-torch.inf)
print(masked)

attn_weight = torch.softmax(masked/math.sqrt(out_dim), dim=-1)
print(attn_weight)


# masking transformwers sets scores for futue tokens to a large negative value, makeing their influden in the softmax

# the softmax funct then recalucates attention weights only amoing the unmasked tokens
# this process ensures no information leakage from lasked tokens, focing the model soley on the intended data


# we could now use the modified attention to cpmute the context vectors via context_vect = attn_weights  @values
# however, in the next section, we first cover another mino


context_vec = attn_weight @ values #  --->1,11,11 @ 1,11,3 ---> 1,11,3
print(f' context vectors: {context_vec}')



# maskin additioanl attention weights with drop out 
# dropout is a deep learning technique where randonly selected hidden layer units are ignored during training
# this prvents overfitting and imporves generalization performance


# dropout is attn is applied in 2 specic areas
# 1. after calculating attn_score - this is mrore common
# lers stay we have atten weight - which casual attention implented(all future token have beedn masked) 1. we create a dropout mask - random posiiotn to be dropped (random zeroed out if the dropout - 0.5 and all the other values wiil be rescale such amount, it drop out 50% of the hidden layer units are ignored during training
# this prvents overfitting and imporves generalization performance


# when applying drouput to an attention weight matrix witha a rate of 50% half of the elments in the matrux are randomly set to zero
# to compensate for the reduction in active elemetns, the values aof the remaining elemin the matrix are scaled up by a factor of 1/0.5 =2
# this sclain is crucial to mainintain the overall balance of the attention weights, ensuring that th average influence mechanism remain consostent during inference trainign

# 2. after applicyin attention weight to vale vect


torch.manual_seed(1)
dropout = nn.Dropout(0.5)
print(dropout(attn_weight))


# m







