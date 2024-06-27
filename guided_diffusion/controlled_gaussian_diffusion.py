"""
This code started out as a PyTorch port of the following:
https://github.com/HJ-harry/MCG_diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

In addition to the original code, for our Diff-Unmix model, we have incorporated spectral unmixing-based guidance into the sampling process to control the generation procedure.
"""

import enum
import math
import gc
from re import T

import torch.nn as nn
import numpy as np
import torch as th
th.cuda.empty_cache()
from torch.autograd import grad
import torch.nn.functional as nF
from functools import partial
import torch.nn.parameter as Para
from tensorly import unfold, fold
import tensorly as tl
# 
# from .core import imresize
from os.path import join as join
from scipy.stats import norm
import torchvision.transforms as transforms
# from math import sqrt, log
# from .resizer import Resizer
    
th.autograd.set_detect_anomaly(True)

mse = th.nn.MSELoss().cuda()

# To control the sampling process of the selected pre-trained diffusion model

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    """

    def __init__(
        self,
        *,
        betas
    ):

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., self.alphas_cumprod))
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # fuse the outputs pf noise and condition
        self.conv1_1 = nn.Conv2d(6, 3, 1, 1, 0, bias=True) 

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None
    ):
        B = x.shape[0]
        noise_level = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t.item())+1]]).repeat(B, 1).to(x.device)   
        
        model_output = model(x, noise_level) 

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        pred_xstart = process_xstart(   
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
        model_mean, _, posterior_log_variance = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {
            "mean": model_mean,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



    def determine_input_stage(self, sqrt_alphas_cumprod_prev, A_y, denoised):
        min_lh = 999
        min_t = -1
        for t in range(sqrt_alphas_cumprod_prev.shape[0]-500):
            noise = A_y - sqrt_alphas_cumprod_prev[t] * denoised
            noise_mean = th.mean(noise)
            noise = noise - noise_mean

            mu, std = norm.fit(noise.cpu().detach().numpy())

            diff = np.abs(np.sqrt(1 - sqrt_alphas_cumprod_prev[t]**2) - std)

            if diff < min_lh:
                min_lh = diff
                min_t = t

        return min_lh, min_t, std


    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        noise = th.randn_like(x)
        nonzero_param = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_param * th.exp(0.5 * out["log_variance"]) * noise 
        return {"pred_xstart": out["pred_xstart"], "mean": out["mean"], "log_variance": out["log_variance"], "sample": sample}

    def p_sample_loop(
        self,
        model,
        shape,
        Rr, 
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_condition=None,
        param=None,
        save_root=None,
        progress=True
    ):
        finalX = None
        # finalA_hat = None

        dstep = 1000        
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            Rr,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            progress=progress,
            param=param,
            save_root=save_root
        ):
            finalX = sample
            # finalA_hat = accumulated_A_hat
             
        return finalX

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        Rr,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_condition=None,
        device=None,
        progress=False,
        param=None,
        save_root=None   # use it for output intermediate predictions
        ):
        Bb, Cc, Hh, Ww = shape
        Rr = Rr
        device = th.device("cuda")
        # estimate coefficient Steps
        A_y = model_condition["A_y"]
        E_y = model_condition["E_y"]
        y = model_condition["y"]
        A_c = model_condition["A_c"]
        A_y = th.unsqueeze(A_y, 0)
        E_y = th.unsqueeze(E_y, 0)
        y = th.unsqueeze(y, 0)
        A_c = th.unsqueeze(A_c, 0)
        normalize3 = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

        ## timestep matching
        match_state = 1
        if match_state:
            min_lh, min_t, std = self.determine_input_stage(self.sqrt_alphas_cumprod_prev, A_y, A_c)
            self.num_timesteps = min_t
            print("Original diffusion timestep: 1000")
            print("Matched timestep:", min_t)
        noise = A_y
        img = noise if noise is not None else th.randn((Bb, Rr, Hh, Ww), device=device)
        indices = list(range(self.num_timesteps))[::-1] 
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        use_grad = 1
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            # re-instantiate requires_grad for backpropagation
            img = img.requires_grad_()

            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn
            )
            Ahat_1 = ((out["pred_xstart"] +1)/2).to(device)
            [b, c, w, h] = Ahat_1.shape

            if use_grad:
                # spectral unmixing based diffusion guidance
                norm1 = th.norm(th.mm(E_y.reshape(28, 3), Ahat_1.reshape(3, 256*256)).reshape(1, 28, 256, 256) - th.mm(E_y.reshape(28, 3), A_c.reshape(3, 256*256)).reshape(1, 28, 256, 256)).to(device)
                # gradient
                norm_gradX = grad(outputs=norm1, inputs=img)
                # correction with gradient, note: guidance scale is a sensitive hyperparameter 
                guidance_scale = th.tensor(5.0)
                out["sample"] = out["sample"] - guidance_scale * norm_gradX[0] # data varied
                # save GPU memory
                del Ahat_1, norm1, norm_gradX
            gc.collect()

            yield out["sample"]
            img = out["sample"]
            img.detach_()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            th.arange(n_timestep + 1, dtype=th.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class Param(th.nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.E = Para.Parameter(data=data)
    
    def forward(self,):
        return self.E
