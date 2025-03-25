import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor: # the -> torch.Tensor is just a return annotation, meaning the function returns a torch.Tensor
        # x: (Batch_Size, Features, Height, Width)

        residue = x

        n, c, h, w = x.shape # returns the sizes of x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(n, c, h*w)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        x = self.attention(x) # shape remains the same

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2) # transpose back

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(n, c, h ,w)

        x+= residue

        return x



class VAE_ResidualBlock(nn.Module): # made up of normalizations and convolutions
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # this Conv changes amount of channels

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # this Conv does not change amount of channels

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, in_channels, Height, Width)

        residue = x

        x = self.groupnorm_1(x) # does not change size of tensor

        x = F.silu(x) # does not change size of tensor

        x = self.conv_1(x) # does not change size of tensor, it does change the amount of channels

        x = self.groupnorm_2(x) # does not change size of tensor

        x = F.silu(x) # does not change size of tensor

        x = self.conv_2(x) # does not change size of tensor, does not change amount of channels

        return x + self.residual_layer(residue) # make sure we add 2 tensors of same shape: if in_channels != out_channels, do a conv2d(residue) before adding it.


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # we start with 4 channels to 4 channels (no change)

            nn.Conv2d(4, 512, kernel_size=3, padding=1), # go from 4 to 512 channels now!!

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # if height = width = 512, then we now have dimension of 512/8 by 512/8 which is 64by64
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # to increase the dimensions
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512,256), # we reduce the amount of features now..
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 512, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256,128), # we reduce the amount of features again
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # after the last upsampling we are already back to original size Height, Width
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #input of the decoder is our latent: x: (Batch_Size, 4, Heigth/8, Width.8)
        #reverse the scaling, which we did last in the encoder.
        x = x / 0.18215

        # now run it through the decoder
        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x