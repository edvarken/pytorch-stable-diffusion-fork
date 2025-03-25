import torch
from torch import nn
from torch.nn import functional as F
import math

#Transformer formula: query*transpose(keys) and then divided by the square root of d_models then apply SoftMax
class SelfAttention(nn.Module): # channels= d_embed = 320,640,1280
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True): # embeddings are the number of channels or features a pixel has (each pixel has many channels)
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # input projection bias: constructs Q, K and V matrices
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # 320/8 = 40, 640/8 = 80, 1280/8 = 160

    def forward(self, x: torch.Tensor, causal_mask=False):
        #x: (Batch_Size, Seq_Len, Dim)

        input_shape = x.shape
        print("input_shape:", input_shape)
        batch_size, sequence_length, d_embed = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        print("intermediate_shape:", intermediate_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 3 * Dim) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # query, key, value
        print("##################")
        print(q.shape)
        print(k.shape) 
        print(v.shape) 
        print("##################")

        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)
        print("######after transpose############")
        print(q.shape) 
        print(k.shape) 
        print(v.shape)
        print("##################")

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2) # the @ symbol is used as the matrix multiplication operator, introduced in Python 3.5
        print("weight.shape:", weight.shape)
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

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output =  weight @ v
        print("output.shape:", output.shape)

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output

# add the init main function
if __name__ == "__main__":
    # add a random torch tensor:
    d_embed = 320 # channels = d_embed = 320,640,1280
    n_heads = 8
    d_head = d_embed // n_heads # d_head = 40, 80, 160
    x = torch.randn(1, 4096, 320) # (Batch_Size, Seq_Len, d_embed)
    # now pass through the self attention layer
    self_attention_output = SelfAttention(n_heads=8, d_embed=320)
    print(self_attention_output)
    y = self_attention_output(x)