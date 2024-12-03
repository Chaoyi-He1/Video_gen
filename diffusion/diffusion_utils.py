import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from . import VarianceSchedule
import pdb
from tqdm import tqdm


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
                e_theta = model(x_t, y=context, t=t)
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
  