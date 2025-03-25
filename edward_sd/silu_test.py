import torch
from torch import nn
from torch.nn import functional as F
import math


# add the init main function
if __name__ == "__main__":
    # add a random torch tensor:
    channels = 1280 # channels = d_embed = 3
    height = 8
    width = height

    x = torch.Tensor([[1, 2, 3],[5, 3, 18]]) # (Seq_Len=2, d_embed=3)
    print(x)
    output = F.silu(x)
    print(output)
    print("##################")
    # y = torch.randn(channels, height, width)
    # output_y1 = F.silu(y)
    # output_y2 = nn.SiLU()(y)
    # print(output_y1-output_y2)
    for i in range(-127, 129): # -127 to 128 inclusive
        # print(i, i*(1/(1+math.exp(-i)))) # to how many decimal places is the output of the SiLU function accurate?
        # print(i, "error:", i*(1/(1+math.exp(-i))) - F.silu(torch.tensor(i, dtype=torch.float64)).item()) # to how many decimal places is the output of the SiLU function accurate?
        # print()
        print(i, i*nn.functional.relu6(torch.tensor(i+3)).item()/6) # only -2 ,-1, 1 and 2 are different, so just a LUT in hardware?
