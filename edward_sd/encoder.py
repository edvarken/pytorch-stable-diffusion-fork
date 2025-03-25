import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock # very similar to Resnet's residual block

class VAE_Encoder(nn.Sequential): # (parentheses after a class definition are used to indicate inheritance)

    # each pixel represents more information but the number of pixels decreases with each step...
    def __init__(self):
        super().__init__( # these are the blocks that will make up our Encoder.
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width) 
            # output has same height and width as the original image here, because we have a padding=1 at both sides and a kernel size=3:) see https://ezyang.github.io/convolution-visualizer/
            nn.Conv2d(3, 128, kernel_size=3, padding=1), # 3input channels(r,g,b), 128 output channels, kernel size=3, padding size=1, you could also do a stride where you skip some pixels

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Width/2) -> (Batch_Size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4, Width/4) -> (Batch_Size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # now run Attention over each pixel, attention will relate a sequence of pixels to each other...
            # so each pixel is not independent of each other, global over the image(e.g. very first and last pixel), not only locally due to the kernel convolutions...
            VAE_AttentionBlock(512), # image remains the same size # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512), # 32 groups, 512 channels or features still.

            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 512, Height/8, Width/8)
            nn.SiLU(),

            # the following two Convolutional layers are called the 'Bottleneck'
            #The number of output channels is reduced to 8, representing a bottleneck that compresses the information from the input image into a much lower-dimensional space.
            # (Batch_Size, 512, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), # decrease the number of features or channels to only 8, this is the bottleneck of the encoder!!

            # (Batch_Size, 8, Height/8, Width/8) -> (Batch_Size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)

        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor: # encoder and sampling part of that Latent space's distribution
        # x: (Batch_Size, Channels, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_left, Padding_right, Padding_top, Padding_bottom)
                x = F.pad(x, (0, 1, 0, 1)) # layer of padding on right side and bottom side only for the layers where stride=2 is happening.
            x = module(x) # apply the layer to tensor x and update x with it.

        # log_variance and mean are the OUTPUTS of this forward method!

        # (Batch_Size, 8, Height, Height/8, Width/8) -> two tensors of shape (Batch_Size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # if value too big, make it stay within these values
        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8)
        log_variance =  torch.clamp(log_variance, -30, 20)

        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8) (does not change size of the tensor)
        variance = log_variance.exp()

        # (Batch_Size, 4, Height/8, Width/8) -> (Batch_Size, 4, Height/8, Width/8) (does not change size of the tensor)
        stdev = variance.sqrt()

        # we now know the mean and variance of the multivariate Gaussian distribution that is the 'latent space'

        # Z = N(0, 1) -> N(mean, variance)=X?
        # X = mean + stdev * Z    (=formula from probability and statistics)
        # this basically means sampling from the distribution X

        x = mean + stdev * noise # where noise is the input Gaussian I think?

        # Scale the output by a constant (historical reason?)
        x *= 0.18215

        return x
