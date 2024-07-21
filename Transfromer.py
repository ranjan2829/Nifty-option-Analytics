import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_heads==0 ,"d-model must be divisible by num_threads"
        self.d_model=d_model
        self.num_heads=num_heads

        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.linear(d_model,d_model)
        self.W_o=nn.linear(d_model,d_model)

        def Scaled_dot_product_attention(self,Q,K,V,mask=None):
            attention_scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
