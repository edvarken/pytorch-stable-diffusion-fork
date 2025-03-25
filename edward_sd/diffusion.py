import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from datetime import datetime

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        start_time_embedding_t = datetime.now()
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        time_embedding_t = datetime.now() - start_time_embedding_t
        print("time_embedding_t", time_embedding_t)
        return x


class UNET_ResidualBlock(nn.Module): # very similar to the residual block we built for the variational autoencoder
    # in this residual block, we are relating the LATENT with the time embedding so that the output will depend on both noise AND timestep
    def __init__(self, in_channels, out_channels, n_time=1280): # time embedding dimension is 1280 deep.
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # 320 320 for the first level
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = False
        if in_channels == out_channels: # connect directly with residual connection
            self.conv1 = False
            self.residual_layer = nn.Identity()
        else: # otherwise create a convolution to connect them(to convert input size to be same as output size), so we can add the two tensors
            self.conv1 = True
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width) this is the LATENT representation/vector
        # time: (1, 1280) this is the TIME embedding

        residue = feature # builds the residual connection
        residual_block_gn1_start_t = datetime.now()
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature) # normalization
        print("residual_block_gn1_t", (datetime.now() - residual_block_gn1_start_t))
        
        residual_block_silu1_start_t = datetime.now()
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature) # activation function
        print("residual_block_silu1_t", (datetime.now() - residual_block_silu1_start_t))
        
        residual_block_firstconv3_start_t = datetime.now()
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature) 
        print("residual_block_firstconv3_t", (datetime.now() - residual_block_firstconv3_start_t))
        
        residual_block_silu2_start_t = datetime.now()
        # (1, 1280) -> (1, 1280)
        time = F.silu(time) # is this correct?
        print("residual_block_silu2_t", (datetime.now() - residual_block_silu2_start_t))

        residual_block_FC_start_t = datetime.now()
        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        print("residual_block_FC_t", (datetime.now() - residual_block_FC_start_t))
        
        residual_block_add1_start_t = datetime.now()
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1) # add time embedding
        print("residual_block_add1_t", (datetime.now() - residual_block_add1_start_t))
        
        residual_block_gn2_start_t = datetime.now()
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        print("residual_block_gn2_t", (datetime.now() - residual_block_gn2_start_t))
        
        residual_block_silu3_start_t = datetime.now()
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        print("residual_block_silu3_t", (datetime.now() - residual_block_silu3_start_t))
        
        residual_block_secondconv3_start_t = datetime.now()
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        print("residual_block_secondconv3_t", (datetime.now() - residual_block_secondconv3_start_t))

        residual_block_conv1_or_add_start_t = datetime.now()
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        add_residue = self.residual_layer(residue)
        if self.conv1:
            print("residual_block_conv1_t", (datetime.now() - residual_block_conv1_or_add_start_t))
        residual_block_add2_start_t = datetime.now()
        addition = merged + add_residue  
        print("residual_block_add2_t", (datetime.now() - residual_block_add2_start_t))
        return addition


class UNET_AttentionBlock(nn.Module): # this attention block will combine the text prompt with the current latent(cross-attention) and self-attention as well
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd # = 8 * 40 = 320 or 8 * 80 = 640 or 8 * 160 = 1280
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6) #GroupNorm, using 32 groups, and a small epsilon value
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0) # CONV1

        self.layernorm_1 = nn.LayerNorm(channels) # LayerNorm: only calculated over the last dimension of the Tensor: channels
        # Layernorm(channels= is a single integer): If a single integer is used, it is treated as a singleton list, 
        # and this module will normalize over the last dimension which is expected to be of that specific size.
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False) # ATTN1: no bias like in the 'vanilla' transformer
        self.layernorm_2 = nn.LayerNorm(channels) # residual add, then LayerNorm
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False) # ATTN 2: cross attention
        self.layernorm_3 = nn.LayerNorm(channels) # residual add, then LayerNorm
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2) # static MatMult, then GeGLU activation function
        self.linear_geglu_2 = nn.Linear(4 * channels, channels) # still GeGLU

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0) # residual add, then CONV1, and a final residual add
    
    def forward(self, x, context): # the forward method
        # x: (Batch_Size, Features, Height, Width) # x is our LATENT representation
        # context: (Batch_Size, Seq_Len, Dim) # context is our TEXT PROMPT

        residue_long = x # called LONG residual connection since it will be applied at the end.
        
        transfo_block_gn1_start_t = datetime.now()
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x) # normalization does not change the size of the tensor
        print("transfo_block_gn1_t", (datetime.now() - transfo_block_gn1_start_t))
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        transfo_block_firstconv1_start_t = datetime.now()
        x = self.conv_input(x) # this also does not change the size of the tensor
        print("transfo_block_firstconv1_t", (datetime.now() - transfo_block_firstconv1_start_t))
        n, c, h, w = x.shape # batch_size, number of channel or features, height, width
        

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        

        # Normalization + Self-Attention with skip connection
        # (Batch_Size, Height * Width, Features)
        residue_short = x # short residual connection that we'll apply right after the attention
        
        transfo_block_ln1_start_t = datetime.now()
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        print("transfo_block_ln1_t", (datetime.now() - transfo_block_ln1_start_t))
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        transfo_block_add1_start_t = datetime.now()
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        print("transfo_block_add1_t", (datetime.now() - transfo_block_add1_start_t))
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        transfo_block_ln2_start_t = datetime.now()
        # Normalization + Cross-Attention with skip connection
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        print("transfo_block_ln2_t", (datetime.now() - transfo_block_ln2_start_t))
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context) # this is Cross-attention
        
        transfo_block_add2_start_t = datetime.now()
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        print("transfo_block_add2_t", (datetime.now() - transfo_block_add2_start_t))
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        transfo_block_ln3_start_t = datetime.now()
        # feed-forward layer
        # Normalization + FFN with GeGLU and skip connection
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        print("transfo_block_ln3_t", (datetime.now() - transfo_block_ln3_start_t))
        
        transfo_block_MM1_geglu_start_t = datetime.now()
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) # herein is also the static MM that happens
        print("transfo_block_MM1_geglu_t", (datetime.now() - transfo_block_MM1_geglu_start_t))
        
        transfo_block_gelu_start_t = datetime.now()
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate) # GELU on only the gate and then element-wise multiplication happens
        print("transfo_block_gelu_t", (datetime.now() - transfo_block_gelu_start_t))
        
        transfo_block_MM2_geglu_start_t = datetime.now()
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x) # herein is also the static MM that happens
        print("transfo_block_MM2_geglu_t", (datetime.now() - transfo_block_MM2_geglu_start_t))
        
        transfo_block_add3_start_t = datetime.now()
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        print("transfo_block_add3_t", (datetime.now() - transfo_block_add3_start_t))

        # reverse the previous transposition so
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w)) # reverse the multiplication of height and width

        transfo_block_secondconv1_start_t = datetime.now()
        # Final skip connection between initial input and output of the block :)
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_output(x)
        print("transfo_block_secondconv1_t", (datetime.now() - transfo_block_secondconv1_start_t))

        transfo_block_add4_start_t = datetime.now()
        addition = x + residue_long
        print("transfo_block_add4_t", (datetime.now() - transfo_block_add4_start_t))
        return addition


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        upsample_block_conv3_start_t = datetime.now()
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # this is the same operation as we did in the decoder.py nn.Upsample(scale_factor=2)
        upsample_block_conv3_start_t = datetime.now()
        x = self.conv(x)
        print("upsample_block_conv3_t", (datetime.now() - upsample_block_conv3_start_t))
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                attention_block_start_t = datetime.now()
                x = layer(x, context)
                attention_block_t = datetime.now() - attention_block_start_t
                print("attention_block_t", attention_block_t)
            elif isinstance(layer, UNET_ResidualBlock):
                residual_block_start_t = datetime.now()
                x = layer(x, time)
                residual_block_t = datetime.now() - residual_block_start_t
                print("residual_block_t", residual_block_t)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), # we start with the LATENT representation which is height/8 and width/8
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), # attention Block of 8 heads and 40 is the embedding size.
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16) 
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)), # image gets smaller due to this CONVOLUTION

            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)), # image gets smaller due to this CONVOLUTION
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)), # image gets smaller due to this CONVOLUTION, number of features=1280
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) (residual connections don't change the size)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)), # this image has 1280 channels=features
        ])


        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), # this attentionblock performs CROSS-attention
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )


        self.decoders = nn.ModuleList([ # decoder will reduce the number of features but increases the image size
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), # why do we expect 2560 input features? since we have a skip connection at the decoders so we each time have
            # double the amount of input features for the decoders, since they all have skip connections
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)), # Upsample!
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)), # increase size of image
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), # the dimension=320 is the same as the output of the UNET
            # give this 320 to the self.final() layer to build the original image size...
        ])
    
    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x): # the input x has sizes:
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x) # the convolution here reduces the size of the features from in_channels=320 to out_channels=4
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4) # go from 320 channels or features to 4 channels or features
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)  # 4 is the size of our embedding
        # context: (Batch_Size, Seq_Len, Dim) # this is the prompt!
        # time: (1, 320) # it's a vector of size 320

        # (1, 320) -> (1, 1280) # convert the time into an embedding, just like the positional encoding of the transformer model(a number multiplied for sines and cosines)
        # info of time is kind of actually also info about position.
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8) # size back to original UNET size of 4, because the UNET takes in latents, predict how much noise is in it,
        # remove the noise and then use that again as input to the UNET model so output dimension must match input dimension...
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output