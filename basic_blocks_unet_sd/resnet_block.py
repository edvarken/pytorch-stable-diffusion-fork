import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# from attention import SelfAttention, CrossAttention
verbose = False
NUM_PRETRAINING_STEPS = 1000
CURRENT_UNET_ITERATION = 0

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # the time tensor is broadcasted to the same shape as the feature tensor and added to it
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)


def set_inference_timesteps(num_inference_steps=50):
    # 999, 998, 997, ..., 0 one thousand numbers but we want only 50 of them so space them every 20
    # 999, 999-20, 999-40, ..., 0 = 50 steps
    step_ratio = NUM_PRETRAINING_STEPS // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    timesteps = torch.from_numpy(timesteps)
    return timesteps


def get_time_embedding(timestep):
    # produces 160 numbers
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) # see Positional Encoding(PE) Formula slide 11
    # multiply with timestep so we create a shape of size (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # concatenated along the last dimension by using dim=-1


timesteps = set_inference_timesteps(50)
if verbose:
    print(timesteps)
time_embedding = TimeEmbedding(320)
timestep = time_embedding(get_time_embedding(timestep=timesteps[CURRENT_UNET_ITERATION]))
# timestep dimension: (1, 1280)

my_res_block = UNET_ResidualBlock(320, 320).type(torch.float8_e5m2fnuz)
if verbose:
    print(my_res_block)
input_latent = torch.randn((1, 320, 64, 64))

# output_latent = my_res_block(input_latent, timestep)
# feature(=latent) dimension: (Batch_Size=1, In_Channels=320, Height=64, Width=64)

# ONNX EXPORT ##
onnx_program = torch.onnx.export(
    my_res_block,                  # model to export
    (input_latent, timestep),        # inputs of the model,
    "./resnet_block_withoutWeightsfloat8.onnx",        # filename of the ONNX model
    export_params=False, 
    do_constant_folding=True,
    input_names=["input_latent", "timestep"]  # Rename inputs for the ONNX model
    # dynamo=True             # True or False to select the exporter to use -> 
    # report=True
    # The TorchDynamo-based ONNX exporter is the newest (and Beta) exporter for PyTorch 2.1 and newer
)
print(onnx_program) # = None


################################################################################################
#  we cannot use this stuff: only for newer versions of PyTorch and we are using a (conda-forged) pytorch=2.3.1
## could go to docker linux where I have torch=2.6.0, then we can use dynamo=True and .save() function

# onnx_program.optimize() # what does this do?
# onnx_program.save("./resnet_block_withoutweights.onnx",
    # include_initializers=False,
    # keep_initializers_as_inputs=False
# )


