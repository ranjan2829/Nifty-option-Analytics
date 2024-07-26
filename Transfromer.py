
import torch
from torch._C import TracingState
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
from torch.optim.optimizer import Optimizer
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
class PostionalWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PostionalWiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)

        self.relu=nn.Relu()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

#The PositionWiseFeedForward class extends PyTorch’s nn.Module and implements a position-wise feed-forward network.
#The class initializes with two linear transformation layers and a ReLU activation function.
#The forward method applies these transformations and activation function sequentially to compute the output.
#This process enables the model to consider the position of input elements while making predictions.

class PositonalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_length):

        super(PositonalEncoding,self).__init__()

        pe=torch.zeros(max_seq_length,d_model)
        position=torch.arange(0,max_seq_length,dtype=torch.float).unsequeeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        self.register_buffer('pe',pe.unsequeeze(0))

        def forward(self,x):
            return x+self.pe[:,x.size(1)]
class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(Encoder,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PostionalWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)


    def forward(self,x,mask):

        attn_out=self.self_attn(x,x,x,mask)
        x=self.norm1(x+self.dropout(attn_out))
        ff_out=self.feed_forward(x)

        x=self.norm2(x+self.dropout(ff_out))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads)
        self.cross_Attn=MultiHeadAttention(d_model,num_heads)
        self.feed_forward=PostionalWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout)
    def forward(self,x,enc_output,srx_mask,tgt_mask):
        attn=self.self_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout(attn))
        attn=self.cross_Attn(x,enc_output,enc_output,srx_mask)

        x=self.norm2(x+self.dropout(attn))

        ff=self.feed_forward(x)

        x=self.norm3(x+self.dropout(ff))

        return x
        #The forward method computes the decoder layer output by performing the following steps:

        #Calculate the masked self-attention output and add it to the input tensor, followed by dropout and layer normalization.
        #Compute the cross-attention output between the decoder and encoder outputs, and add it to the normalized masked self-attention output, followed by dropout and layer normalization.
        #Calculate the position-wise feed-forward output and combine it with the normalized cross-attention output, followed by dropout and layer normalization.
        #Return the processed tensor.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout):

        super(Transformer,self).__init__()

        self.encoder_embedding=nn.Embedding(src_vocab_size,d_model)
        self.decoder_embedding=nn.Embedding(tgt_vocab_size,d_model)
        self.positional_encoding=PositonalEncoding(d_model,max_seq_length)

        self.encoder_layer=nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layer=nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc=nn.Linear(d_model,tgt_vocab_size)
        self.dropout=nn.Dropout(dropout)

    def generate_mask(self,src,tgt):
        src_mask=(src!=0).unsueeze(1).unsequeeze(2)
        tgt_mask=(tgt!=0).unsequeeze(1).unsequeeze(3)
        seq_length=tgt.size(1)
        nopeak_mask=(1-torch.triu(torch.ones(1,seq_length,seq_length),diagonal=1)).bool()
        tgt_mask=tgt_mask & nopeak_mask
        return src_mask,tgt_mask


    def forward(self,src,tgt):
        src_mask,tgt_mask=self.generate_mask(src,tgt)
        src_embedded=self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded=self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        enc_output=src_embedded
        for enc_layer in self.encoder_layer:
            enc_output=enc_layer(enc_output,src_mask)

        dec_output=tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output=dec_layer(dec_output,enc_output,src_mask,tgt_mask)
        output=self.fc(dec_output)

        return output
src_vocab_size=5000
tgt_vocab_size=5000

d_model=512
num_heads=8

num_layers=6
d_ff=2048
max_seq_length=100

dropout=0.1

transformer=Transformer(src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout)
src_data=torch.randint(1,src_vocab_size,(64,max_seq_length))
tgt_data=torch.randint(1,tgt_vocab_size,(64,max_seq_length))

criterion=nn.CrossEntropyLoss(ignore_index=0)

optimizer=optim.Adam(transformer.parameters(),lr=0.0001,betas=(0.9,0.98),eps=1e-9)

for epoch in range(100):
    optimizer.zero_grad()
    output=transformer(src_data,tgt_data[:,:-1])
    loss=criterion(output.contiguous().view(-1,tgt_vocab_size),tgt_data[:,1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()

    print(f"Epoch : {epoch+1},loss : {loss.item()}")
