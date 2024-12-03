import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from tqdm import tqdm
import math


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class VarianceSchedule(nn.Module):
    """
    Variance schedule for diffusion process.
    Parameters
    ----------
    num_steps: int, number of steps in the diffusion process. (Markov chain length)
    mode: str, 'linear' or 'cosine', the mode of the variance schedule.
    beta_1: float, the initial value of beta.
    beta_T: float, the final value of beta.
    cosine_s: float, the cosine annealing start value.

    Attributes
    ----------
    betas: Tensor, [T+1], the beta values.
    alphas: Tensor, [T+1], the alpha values. alpha = 1 - beta
    alpha_bars: Tensor, [T+1], the cumulative sum of alpha. alpha_bar_t = sum_{i=0}^{t-1} alpha_i
    sigmas_flex: Tensor, [T+1], the flexible part of the variance schedule. sigma_t = sqrt(beta_t)
    sigmas_inflex: Tensor, [T+1], the inflexible part of the variance schedule. sigma_t = sqrt(beta_t)
    """
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    

class Diffusion_utils(nn.Module):
    def __init__(self, var_sched: VarianceSchedule):
        super().__init__()
        self.var_sched = var_sched


    def get_loss(self, x_0, context, t=None, model: nn.Module=None):
        """
        Diffusion loss.
        Based on Denoising Diffusion Probabilistic Models
        equation (14) in
        https://arxiv.org/abs/2006.11239
        Loss = ||\epsilon - \epsilon_theta(\sqrt(\alpha_bar_t x0) + \sqrt(1 - \alpha_bar_t \epsilon)
                                          , t)||^2
        """
        batch_size = x_0.shape[0]   # (B, N, c, d)
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        t = torch.tensor(t).to(x_0.device)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1).cuda()       # (B, 1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1).cuda()   # (B, 1, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, c, d)

        e_theta = model(c0 * x_0 + c1 * e_rand, y=context, t=t)
        loss = F.mse_loss(e_theta, e_rand, reduction='mean')
        return loss

    def sample(self, shape, context, sample, bestof, model: nn.Module,
               flexibility=0.0, ret_traj=False, sampling="ddpm", step=1):
        """
        Sample from the diffusion model.
        DDPM: Denoising Diffusion Probabilistic Models
        https://arxiv.org/abs/2006.11239
        DDIM: Denoising Diffusion Implicit Models
        https://arxiv.org/abs/2010.02502
        
        shape: tuple, (channels, frames, height, width)
        """
        traj_list = []
        for _ in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, shape[0], shape[1], shape[2], shape[3]]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, shape[0], shape[1], shape[2], shape[3]]).to(context.device)
            traj = {self.var_sched.num_steps: x_T}
            stride = step

            for t in tqdm(range(self.var_sched.num_steps, 0, -stride)):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]    # next: closer to 1
                # pdb.set_trace()
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t] * batch_size]
                e_theta = model(x_t, y=context, t=torch.tensor([t] * batch_size).to(x_t.device))
                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
        return torch.stack(traj_list)
  