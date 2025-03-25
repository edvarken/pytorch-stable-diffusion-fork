import torch
from torch import nn
from torch.nn import functional as F
import math
from datetime import datetime

#Transformer formula: query*transpose(keys) and then divided by the square root of d_models then apply SoftMax
class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True): # embeddings are the number of channels or features a pixel has (each pixel has many channels)
        super().__init__()
         # input projection bias: this is the Q = input*W_q, K = input*W_k, V = input*W_v
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # the W_q, W_k, W_v matrices are learnable params trained during backprop, and are concatenated into 1 big matrix.
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # d_head is the dimensionality of the query/key space for each head.

    def forward(self, x: torch.Tensor, causal_mask=False):
        #x: (Batch_Size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        selfattn_block_static_mms1_start_t = datetime.now()
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 3 * Dim) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # query, key, value matrices are constructed now, this already needed 3 matrix matrix multiplications
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)
        print("selfattn_block_static_mms1_t", (datetime.now() - selfattn_block_static_mms1_start_t))

        selfattn_block_dynamic_mm1_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2) # the @ symbol is used as the matrix multiplication operator, introduced in Python 3.5
        print("selfattn_block_dynamic_mm1_t", (datetime.now() - selfattn_block_dynamic_mm1_start_t))
        
        selfattn_block_softmax_start_t = datetime.now()
        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is made up of ones
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # fill up with -infinity
            weight.masked_fill_(mask, -torch.inf)

        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = weight / math.sqrt(self.d_head)
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)
        print("selfattn_block_softmax_t", (datetime.now() - selfattn_block_softmax_start_t))

        selfattn_block_dynamic_mm2_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output =  weight @ v
        print("selfattn_block_dynamic_mm2_t", (datetime.now() - selfattn_block_dynamic_mm2_start_t))

        selfattn_block_static_mm2_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)
        print("selfattn_block_static_mm2_t", (datetime.now() - selfattn_block_static_mm2_start_t))

        # (Batch_Size, Seq_Len, Dim)
        return output
    

class CrossAttention(nn.Module): # very similar to Self Attention, except that the query come from one side(latent), and the keys and the values from another side(text)
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        # d_cross= dimension of the embedding of the keys and values
        # d_embed= dimension of the queries, = channels here so 320 for level 1 I think?
        super().__init__() # we'll define 3 separate matrices instead of one big matrix.
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias) # W-q matrix
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias) # W-k matrix
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias) # W-v matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x is our query (LATENT)
        # y is our keys and values (=context or the 'prompt')
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q) 
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        crossattn_block_static_mms1_start_t = datetime.now()
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x) # multiply queries by W-q matrix
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y) # mulitply keys by W-k matrix
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y) # multiply values by W-v matrix

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        print("crossattn_block_static_mms1_t", (datetime.now() - crossattn_block_static_mms1_start_t))
        
        crossattn_block_dynamic_mm1_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2) # matrix multiplication
        print("crossattn_block_dynamic_mm1_t", (datetime.now() - crossattn_block_dynamic_mm1_start_t))
        
        crossattn_block_softmax_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head) # divide weight by the square root of the dimension of each head
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        print("crossattn_block_softmax_t", (datetime.now() - crossattn_block_softmax_start_t))
        
        crossattn_block_dynamic_mm2_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        print("crossattn_block_dynamic_mm2_t", (datetime.now() - crossattn_block_dynamic_mm2_start_t))
        
        crossattn_block_static_mm2_start_t = datetime.now()
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous() # transpose the output
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape) # reshape the output
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)
        print("crossattn_block_static_mm2_t", (datetime.now() - crossattn_block_static_mm2_start_t))

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output

