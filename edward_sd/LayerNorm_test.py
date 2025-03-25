import torch
from torch import nn
from torch.nn import functional as F
import math


# add the init main function
if __name__ == "__main__":
    # add a random torch tensor:
    d_embed = 3 # channels = d_embed = 3
    channels = d_embed



    x = torch.Tensor([[1, 2, 3],[5, 3, 18]]) # (Seq_Len=2, d_embed=3)
    # do the LayerNorm
    print(x)
    layernorm_1 = nn.LayerNorm(channels, eps=0) # normalize over the last dimension only: the channels=d_embed=3 in this case(which would be 320 in level1 of real SD1.5's UNET)
    groupnorm_1 = nn.GroupNorm(1, channels, eps=0)
    output = layernorm_1(x)
    print(output)
    output = groupnorm_1(x)
    print(output) # is the same:)




    # m = nn.GLU()
    # input = torch.randn(4, 2)
    # print(input)
    # output = m(input)
    # print(output)