import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from attention import SelfAttention, CrossAttention
verbose = False
    
class UNET_TransformerBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768): # always 8 heads, 40,80,160 embeddings
        super().__init__()
        channels = n_head * n_embd # 8*[40,80,160] = 320,640,1280
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False) # channels = 320,640,1280, n_head is always 8
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x) # (1, 8*8=64, 1280)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

my_transfo_block = UNET_TransformerBlock(8, 40) # 8 heads, 40=embedding dimension of each head in MHA
if verbose:
    print(my_transfo_block)
input_latent = torch.randn((1, 320, 64, 64))
# TODO: context = clip(prompt)
context = torch.randn((1, 77, 768)) # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
output_latent = my_transfo_block(input_latent, context)
# feature(=latent) dimension: (Batch_Size=1, In_Channels=320, Height=64, Width=64)

# ONNX EXPORT ##
onnx_program = torch.onnx.export(
    my_transfo_block,                  # model to export
    (input_latent, context),        # inputs of the model,
    "./transformer_block_withoutWeights2.onnx",        # filename of the ONNX model
    export_params=False, 
    do_constant_folding=True,
    input_names=["input_latent", "CLIP(prompt)"]  # Rename inputs for the ONNX model
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


