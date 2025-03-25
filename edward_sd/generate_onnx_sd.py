import torch
print(torch.__version__)  # Check if it is installed correctly
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ddpm import DDPMSampler

import model_loader
# from PIL import Image
from transformers import CLIPTokenizer

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False
n_inference_steps = 50
sampler_name = "ddpm"
seed = 42
prompt = "A cat stretching on the floor, photorealistic"
uncond_prompt = "cat ears" # you can use it as a negative prompt
do_cfg = True # if we set this to False, it will just generate the image without looking at the text prompt?
cfg_scale = 7



if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.backends.mps.is_built() or torch.backend.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"using device {DEVICE}") # always uses cpu, no metal performance shaders(mps) idk why

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt" # Exponential Moving Average (EMA) is a model averaging technique that maintains 
# an exponentially weighted moving average of the model parameters during training. The averaged parameters are used for model evaluation.
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

unet_model = models["diffusion"]
clip_model = models["clip"]
VAE_encoder_model = models["encoder"]
VAE_decoder_model = models["decoder"]


tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids # sequence length is 77 because it is the max length for our prompt
tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
context = clip_model(tokens)

generator = torch.Generator(device=DEVICE) # gebruik ik hier niet

sampler = DDPMSampler(generator) # we give it the noise generator as input
sampler.set_inference_timesteps(n_inference_steps) # see slides Architexture (Text-To-Image) we need to tell the scheduler how many denoising steps to do
# e.g. during inference we can only do 50 denoising steps, while during the training we used 1000 denoising steps in the forward pass
# for some other samplers that work on differential equations we can do even less inferencing steps.
timesteps = tqdm(sampler.timesteps)

latents_shape = (1, 4, 64, 64) # this is the latents that will run through the UNET
input_tensor = torch.randn(latents_shape, device=torch.device(DEVICE)) # je zou hier nog een generator adhv een seed kunnen meegeven.

timestep=0
def get_time_embedding(timestep):
    # produces 160 numbers
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) # see Positional Encoding(PE) Formula slide 11
    # mulitply with timestep so we create a shape of size (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # concatenated along the last dimension by using dim=-1



# for i, timestep in enumerate(timesteps):
time_embedding = get_time_embedding(timestep)

# model_output = unet_model(input_tensor, context, time_embedding)

# ONNX EXPORT ##
onnx_program = torch.onnx.export(
    unet_model,                  # model to export
    (input_tensor, context, time_embedding),        # inputs of the model,
    # "../generated_onnx/run2/unet_model.onnx",        # filename of the ONNX model
    # input_names=["input"]  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use -> 
    # report=True
    # The TorchDynamo-based ONNX exporter is the newest (and Beta) exporter for PyTorch 2.1 and newer
)
onnx_program.optimize()
onnx_program.save("../generated_onnx/run2/unet_model.onnx")


# # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
# with torch.onnx.enable_fake_mode():
#     # When initialized in fake mode, the model's parameters are fake tensors
#     # They do not take up memory so we can initialize large models
#     my_nn_module = unet_model
#     arg1 = input_tensor,
#     onnx_program = torch.onnx.export(my_nn_module, (arg1,), dynamo=True)
#     # Saving model WITHOUT initializers (only the architecture)
#     onnx_program.save(
#     "my_model_without_initializers.onnx",
#     include_initializers=False,
#     keep_initializers_as_inputs=True,
# )
# # Saving model WITH initializers after applying concrete weights
# onnx_program.apply_weights({"weight": torch.tensor(42.0)})
# onnx_program.save("my_model_with_initializers.onnx")












###### INFERENCE ######
# ## TEXT TO IMAGE

# prompt = "A cat stretching on the floor, photorealistic"
# uncond_prompt = "cat ears" # you can use it as a negative prompt
# do_cfg = True # if we set this to False, it will just generate the image without looking at the text prompt?
# cfg_scale = 7

# ## IMAGE TO IMAGE

# input_image = None
# image_path = "../images/dog.jpg"
# # input_image = Image.open(image_path)
# strength = 0.9

# sampler = "ddpm"
# num_inference_steps = 50
# seed = 42

# output_image = pipeline.generate(
#     prompt=prompt,
#     uncond_prompt=uncond_prompt,
#     input_image=input_image,
#     strength=strength,
#     do_cfg=do_cfg,
#     cfg_scale=cfg_scale,
#     sampler_name=sampler,
#     n_inference_steps=num_inference_steps,
#     seed=seed,
#     models=models,
#     device=DEVICE,
#     idle_device="cpu",
#     tokenizer=tokenizer
# )

# Image.fromarray(output_image)
