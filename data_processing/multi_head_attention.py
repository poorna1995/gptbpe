# the multi-head refere to divinding the attention mechanisms inot muliple heads( eacah operate independlty)


# implemention f mha with weight splits
# 1. instead of mainataing 2 separate clases of mha and casual attention . we combined both of these into a single multihead attention class

# num of head is specified = head _dimens = out_dim/n_head



# step1 : atart witht the input 
# batch_size = batch sizee
# num_token, = 
# out_embed( = every token that is represented in a vector dimension)
# eg : (1,3,6)

# Text ==" the cat sleeps" --->[1,3,6]
# the  --> 0 ( tokenid) --> [ 1,0,2,0,3.0,4.0,5.0,6.0](embedded vector) 
# cat  ---> 12 ( tokenid) -> [ 1,0,2,0,3.0,4.0,5.0,6.0](embedded vector)
# sleep ---> 18( tokenid) -> [ 1,0,2,0,3.0,4.0,5.0,6.0](embedded vector)

#step 2 : decide : out_dim = 6 , num_head = 2 ---> head_dim = out_dim/num_head (6/2 =3)

# step 3 : intialise trainable weight matrics for q,k,v( wq,wk,wv) 

# wq = out_em*out_dim = 6*6
# wk = out_em*out_dim = 6*6
# wv = out_em*out_dim = 6*6


# step4 : calculate the querys, keys, values ( input@ wq,input@ wk,input@ wv)
# query ---> [batch_size, num_token, out_embed] (1, 3, 6) @[batch_size,out_em,out_dim](1, 6, 6) ---> [batch_size,num_token, out_dim ](1,3,6)
# key ---> [batch_size, num_token, out_embed] (1, 3, 6) @[batch_size,out_em,out_dim](1, 6, 6) ---> [batch_size,num_token, out_dim ](1,3,6)
# values ---> [batch_size, num_token, out_embed] (1, 3, 6) @[batch_size,out_em,out_dim](1, 6, 6) ---> [batch_size,num_token, out_dim ](1,3,6)


#step unroll last dimension of keys, queires, and values to include num_heads and head_dim
# head_dim = out_dim/n_head (6/2 =3)
#  [batch_size, num_tokens,out_dim ] ---> [batch_size, num_tokens, num_head, head_dim]
# 

# step 6: group matrices by "'number of heads"
# [batch_size, num_tokens, num_head, head_dim] ---->  [batch_size, num_head, num_tokens, head_dim]

# step7 find the attention score
# [batch-size, num_heads, num_token, head_dim] @ [batch-size,num_heads,head_dim,num_token] --->  [batch-size, num_head, num_token, num_token ]

# step 8 - attn weight --> repalce all the value sof the upper layer with negative infinity ---> divide by sqrt(head-dim)---> softmax --> dropout




# <div class="alert alert-block alert-info">

# Step 1: Reduce the projection dim to match desired output dim

# Step 2: Use a Linear layer to combine head outputs

# Step 3: Tensor shape: (b, num_tokens, d_out)

# Step 4: We implicitly split the matrix by adding a `num_heads` dimension. Then we unroll last dim: (b,
# num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

# Step 5: Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)

# Step 6: Compute dot product for each head

# Step 7: Mask truncated to the number of tokens

# Step 8: Use the mask to fill attention scores

# Step 9: Tensor shape: (b, num_tokens, n_heads, head_dim)

# Step 10: Combine heads, where self.d_out = self.num_heads * self.head_dim

# Step 11: Add an optional linear projection
# </div>










import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__ (self, out_emb, out_dim, context_length, num_heads, dropout, bias = False):
        super().__init__()
        assert out_dim%num_heads ==0
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim//num_heads
        self.w_q  = nn.Linear(in_features = out_emb ,out_features = out_dim, bias = False) # [out_emb,out_dim ]
        self.w_k  = nn.Linear(in_features = out_emb ,out_features = out_dim, bias = False) # [out_emb,out_dim ]
        self.w_v  = nn.Linear(in_features = out_emb ,out_features = out_dim, bias = False) # [out_emb,out_dim ]
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
    
    # x = [batch_size, num_token, out_embed]
    def forward(self, x):
        batch_size, num_token, out_embed = x.shape
        query = self.w_q(x) # [batch_size, out_emb, out_dim] @ [batch_size, num_token, out_embed] --->[batch_size, num_token, out_dim]
        key   = self.w_q(x) # [batch_size, out_emb, out_dim] @ [batch_size, num_token, out_embed] --->[batch_size, num_token, out_dim]
        value = self.w_q(x) # [batch_size, out_emb, out_dim] @ [batch_size, num_token, out_embed] --->[batch_size, num_token, out_dim]
        
        # unwrap the dimension 
        
        query  = query.view(batch_size,num_token ,self.num_heads, self.head_dim) # [batch_size, num_token, num_head,head_dim]
        key    = key.view(batch_size,num_token ,self.num_heads, self.head_dim)   # [batch_size, num_token, num_head,head_dim
        value  = value.view(batch_size,num_token ,self.num_heads, self.head_dim) # [batch_size, num_token, num_head,head_dim
        
        
        query = query.transpose(1,2) # [batch_size, num_head, num_token,head_dim]
        key = key.transpose(1,2)  # [batch_size, num_head, num_token,head_dim]
        value = value.transpose(1,2) # [batch_size, num_head, num_token,head_dim]
    
        # attention score ( batch_size, numb_head, num_token, num-toke) = q.k
        attn_score = query @ key.transpose(2,3) #[batch_size, num_head, num_token,head_dim] @[batch_size, num_head,head_dim, num_token] ---> [batch_size, num_head,num_token, num_token]
        
        attn_score.masked_fill_(self.mask.bool(), -torch.inf)
        
        # softmax
        attn_weight = torch.softmax(attn_score/ math.sqrt(out_dim), dim = -1)
        attn_weight = self.dropout(attn_weight) # [batch_size, num_head,num_token, num_token]
        
        context_vec = (attn_weight @ value).transpose(1, 2)  # [batch_size, num_head,num_token, num_token] @ # [batch_size, num_head,num_token, num_token] --->[batch_size,num_token,num_head num_token] 
        
        context_vec = context_vec.contiguous().view(batch_size,num_token,self.out_dim)
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
out_dim =3
context_length = input_embedding.shape[1]
num_heads = 3
dropout = 0.1

multi_head_attention = MultiHeadAttention( out_emb, out_dim, context_length, num_heads, dropout, bias = False)

output = multi_head_attention(input_embedding)
print(output.shape)

