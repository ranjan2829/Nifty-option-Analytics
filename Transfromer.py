from _typeshed import BytesPath
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
#The MultiHeadAttention code initializes the module with
#input parameters and linear transformation layers.
#It calculates attention scores, reshapes the input tensor into
#multiple heads, and combines the attention outputs from all heads.
#The forward method computes the multi-head self-attention,
#allowing the model to focus on some
#different aspects of the input sequence.

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_heads==0 ,"d-model must be divisible by num_threads"
        self.d_model=d_model
        self.num_heads=num_heads

        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)#dimensionaltiy
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)

        def Scaled_dot_product_attention(self,Q,K,V,mask=None):
            attention_scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
            if mask is not None:
                attention_scores=attention_scores.masked_fill(mask==0,-1e9)
            attention_probs=torch.softmax(attention_scores,dim=-1)
            output=torch.matmul(attention_probs,V)
            return output


        def spilt_heads(self,x):
            batch_size,seq_length,d_model=x.size()
            return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)
        def combine_heads(self,x):
            batch_size,_,seq_length,d_k=x.size()
            return x.transpose(1,2).contigious().view(batch_size,seq_length,self.d_model)

        def forward(self,Q,K,V,mask=None):
            Q=self.split_heads(self.W_q(Q))
            K=self.split_heads(self.W_k(K))
            V=self.split_heads(self.W_v(V))

            attention_output=self.scaled_dot_product_attention(Q,K,V,mask)
            output=self.W_o(self.combine_heads(attention_output))
            return output
