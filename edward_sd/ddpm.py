# this is the actual scheduler that does the 50 denoising steps in the UNET
import torch
import numpy as np

class DDPMSampler: # not called scheduler since we dont want to confuse with the beta scheduler

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.0085, beta_end: float = 0.0120):
        # forward process makes image more noisy according to Markov chain of Gaussian distributions
        # the noise we add varies acording to a variance schedule Beta1, Beta2, Beta3, ..., BetaT (T=final timestep)
        # beta_start = 0.0085, beta_end = 0.0120 choice made by authors of Stable Diffusion
        # 1000 numbers between beta_start and beta_end

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # alpha_bar is the product of alpha going from 1 up to t timestep t is somewhere between timestep 1 and 1000
        # alpha = 1 - beta

        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0*alpha_1, alpha_0*alpha_1*alpha_2, ...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps # this number is 1000 training steps for the pre-trained SD1.5 model we use
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy()) # [::-1] reverses it so it goes from 999 to 1

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps

        # 999, 998, 997, ..., 0 one thousand numbers but we want only 50 of them so space them every 20
        # 999, 999-20, 999-40, ..., 0 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)


    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t


    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # computed using formula (7) of the DDPM paper.
        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * current_beta_t
        variance = torch.clamp(variance, min=1e-20) # to make sure it doesn't reach zero
        return variance


    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor): # model output is the predicted noise of the UNET, and corresponds to epsilon_theta of x_t, t (so at timestep t)
        # see paper DDPM equation (11)
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one # if no previous step just return 1
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Compute coefficients for pred_original_sample x_0 and coefficient for current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        variance = 0
        if t > 0: # only add noise for all timesteps but the last. so if t=0 do not add noise
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise # this is actually already the standard deviation * noise
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample




    # adds noise to an image according to the equation (4) of the DDPM paper.
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        # calculate mean and variance
        # alpha bar is cumulative product of all alphas
        alpha_cumprod = self.alpha_cumprod.to(defice=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5 # square root of alpha_cumprod[timesteps]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1) # keep adding dimensions until the tensor original_samples and the tensor sqrt_alpha_prod have the same dimensions
        
        # this is the standard deviation
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5 # timesteps ==  timestep T so the last timestep 
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # again keep adding dimension until they have the same dimension
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise
        return noisy_samples