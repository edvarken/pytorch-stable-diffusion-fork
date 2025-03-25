import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

"""
 The encoder is not used because the image generation starts from pure noise, 
 and the process involves denoising this noise to create an image based on the prompt.
 If you were starting from an existing image, the encoder would be necessary to convert that image into the latent space
 before applying the diffusion process.
"""

def generate(
    prompt: str,
    uncond_prompt: str,  # can be empty string or if not empty: also called 'negative prompt' to go away from a certain object e.g. you dont want a cat on a sofa, then you put sofa in negative prompt
    input_image=None,
    strength=0.8,  # if we start from an image to generate an image, how much attention we want to pay to the starting image, if strength=1 we start from pure noise!
    do_cfg=True,  # do classifier free guidance
    cfg_scale=7.5,  # value between 1-14, tells us how much we want to pay attention to the prompt=condition
    sampler_name="ddpm",  # we will only use one scheduler/sampler, this defines it
    n_inference_steps=50,  # number of inference steps.
    models={},  # are the pretrained models
    seed=None,  # how we want to initialize
    device=None,  # where we want to create our tensor
    idle_device=None,  # if we load a model on CUDA, and if we then don't need it, we move it to the CPU
    tokenizer=None,
):  # given all this info, our pipeline will generate one image

    begin_pipeline_generate_time = datetime.now()
    with torch.no_grad():  # this disables grad, since we are inferencing the model, not training it with gradient descent.
        
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)

        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        begin_clip_time = datetime.now()

        clip = models["clip"]
        clip.to(device) # move the clip model to the device
        
        # with classifier free guidance we inference the model twice, first by specyfing the condition=prompt
        # another time without specifying any condition=prompt
        # then we apply a weigth w = the cfg_scale, so this tells us how much we want the model to pay attention to the conditioned output
        # convert it into a list of tokens
        tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids # sequence length is 77 because it is the max length for our prompt
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        # (Batch_Size=1only, Seq_Len, Dim) = (1, 77, 768)
        context = clip(tokens)
        # we are finished using clip so we can move it to the idle device, very useful if you have a very limited GPU and want to offload the models after using them
        to_idle(clip) # offload the models back to the CPU

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator) # we give it the noise generator as input
            sampler.set_inference_timesteps(n_inference_steps) # see slides Architexture (Text-To-Image) we need to tell the scheduler how many denoising steps to do
            # e.g. during inference we can only do 50 denoising steps, while during the training we used 1000 denoising steps in the forward pass
            # for some other samplers that work on differential equations we can do even less inferencing steps.
        else:
            raise ValueError(f"Unkown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH) # this is the latents that will run through the UNET

        # if we have aprompt we can take care of the prompt by either running classifier free guidance(combining the output of the model with and without the prompt)

        # start with pure random noise(~~ as if strength was equal to 1)
        latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device) # e.g. our CUDA device
        # when we train the model we had a maximum of 1000 timesteps, but when inferencing we could only do with 50

        # if max=1000, minimum is 1; if maximum is 999, then minimum is 0
        # each time step represents a NOISE LEVEL: 1000/20 = 50 timesteps
        # 1000 980 960 940 920 900 880 860 840 820 ... 0
        # so less and less noise is added to the image
        # we start with a very noisy image and we remove the noise step by step
        # we have a UNET that predicts the noise in the latent space, and we remove it
        # we do this 50 times, and then we have a denoised latent space, which we can give to the decoder to generate the image
        clip_t = datetime.now() - begin_clip_time
        print("clip_t", clip_t)
        UNET_begin_time = datetime.now()
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            UNET_iteration_begin_time = datetime.now()
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device) # equal to the positional encoding used in the transfomer model, here we use sines and cosines to define the timestep

            # (Batch_Size, 4, Latents_Height=64, Latents_Width=64) is the shape of the input of the encoder of the VAE, which has 4 channels
            model_input = latents

            if do_cfg: # classifier free guidance means we look at mix of latent with the prompt and the latent without the prompt
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2* Batch_Size, 4, Latents_Height, Latents_Width) # we're making 2 copies of the latent, one with the prompt, one without
                # copying means multiplying the Batch_Size times 2
                model_input = model_input.repeat(2, 1, 1, 1) #  BATCH SIZE = 2!

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2) # chunk splits along the 0th dimension by default
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            #now comes the CLUE part: we have a model that is able to predict the noise in the CURRENT latent
            # e.g. we're doing text to image: we start with some random noise, tranfsorm into latents, and according to some scheduler's timesteps we keep denoising it (less and less)
            # we predict the noise in the latents
            # how can we remove the noise? this is done by the scheduler
            # at each step we ask the UNET how much noise is in the LATENT (NOT image yet, UNET works still in latent space) and remove it, again and again until we finish all 50 timesteps.
            # then we take the latent, give it to the decoder which will build the IMAGE.

            # remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)
            print(f"UNET_iteration_{timestep}_t:", (datetime.now() - UNET_iteration_begin_time))


        UNET_total_t = datetime.now() - UNET_begin_time
        print("UNET_total_t:", UNET_total_t)
        print("average_UNET_iteration_time:", (UNET_total_t / n_inference_steps))
        to_idle(diffusion)
        decoder_begin_time = datetime.now()
        # we now have our denoise latent after the 50 timesteps, now load the decoder
        decoder = models["decoder"]
        decoder.to(device) # put decoder on the GPU
        # run the latent through the decoder
        images = decoder(latents)
        to_idle(decoder)
        print("decoder_t:", (datetime.now() - decoder_begin_time))
        end_pipeline_generate_time = datetime.now()
        print("TOTAL_T:", (end_pipeline_generate_time - begin_pipeline_generate_time))

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x+= new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # produces 160 numbers
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) # see Positional Encoding(PE) Formula slide 11
    # mulitply with timestep so we create a shape of size (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # concatenated along the last dimension by using dim=-1



