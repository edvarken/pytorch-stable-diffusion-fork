import torch
from torch import nn
from torch.nn import functional as F
import math


# add the init main function
if __name__ == "__main__":
    # add a random torch tensor:
    d_embed = 4 # channels = d_embed = 3
    channels = d_embed
    x = torch.Tensor([[1.0, 2.0, 3.0, 4.0], [9.5, 11.0, 3.0, 19.0]]) # (Seq_Len=2, d_embed=4)
    # do the GroupNorm
    print(x)
    print()
    # normalize over the last dimension only? NO: the channels=d_embed=4 in this case(which would be 320 in level1 of real SD1.5's UNET)
    groupnorm_1 = nn.GroupNorm(2, channels, eps=0,)
    output = groupnorm_1(x)
    print(output)
    print()

    layernorm_1 = nn.LayerNorm(channels, eps=0) 
    output = layernorm_1(x)
    print(output)




    # m = nn.GLU()
    # input = torch.randn(4, 2)
    # print(input)
    # output = m(input)
    # print(output)