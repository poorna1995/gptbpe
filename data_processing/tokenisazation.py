import torch
import tiktoken
import torch.nn as nn

class Tokenisation(nn.Module):
    
    def __init__(self, text,out_dim):
        super(Tokenisation,self).__init__()
        self.tokenaiser = tiktoken.get_encoding("cl100k_base")
        self.ids = self.tokenaiser.encode(text)
        self.vocab = max(self.ids) +1 
        self.embeddings = nn.Embedding(self.vocab,out_dim)
    def forward(self):
        token_tensor= torch.tensor(self.ids)
        print(f'token:{token_tensor}')
        embedding_vect = self.embeddings(token_tensor)
        return embedding_vect 


 
text = """hello , I'm Poorna Praneesha!"""
print(len(text))
out_dim = 4

tokenisation= Tokenisation(text,out_dim)
token_id = tokenisation.forward()

print(token_id)
print(token_id.shape)



