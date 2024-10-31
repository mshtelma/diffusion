# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Utils for working with diffusion schedulers and performing sampling."""

import torch


def shift_noise_schedule(noise_scheduler, base_dim: int = 64, shift_dim: int = 64):
    """Shifts the function SNR(t) for a noise scheduler to correct for resolution changes.

    Implements the technique from https://arxiv.org/abs/2301.11093

    Args:
        noise_scheduler (diffusers.SchedulerMixin): The noise scheduler to shift.
        base_dim (int): The base side length of the schedule resolution.
        shift_dim (int): The new side length of the schedule resolution.

    Returns:
        diffusers.SchedulerMixin: The shifted noise scheduler.
    """
    # First, we need to get the original SNR(t) function
    alpha_bar = noise_scheduler.alphas_cumprod
    SNR = alpha_bar / (1 - alpha_bar)
    # Shift the SNR acorrording to the resolution change
    SNR_shifted = (base_dim / shift_dim)**2 * SNR
    # Get the new alpha_bars
    alpha_bar_shifted = torch.where(SNR_shifted == float('inf'), torch.tensor(1.0), SNR_shifted / (1 + SNR_shifted))
    # Get the new alpha values
    alpha_shifted = torch.empty_like(alpha_bar_shifted)
    alpha_shifted[0] = alpha_bar_shifted[0]
    alpha_shifted[1:] = alpha_bar_shifted[1:] / alpha_bar_shifted[:-1]
    # Get the new beta values
    beta_shifted = 1 - alpha_shifted
    # Update the noise scheduler
    noise_scheduler.alphas = alpha_shifted
    noise_scheduler.betas = beta_shifted
    noise_scheduler.alphas_cumprod = alpha_bar_shifted
    return noise_scheduler


class ClassifierFreeGuidance:
    """Implements classifier free guidance given a conditional and unconditional output.

    Args:
        guidance_scale (float): The scale of the guidance.
    """

    def __init__(self, guidance_scale: float):
        self.guidance_scale = guidance_scale

    def perform_guidance(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """A function that performs classifier free guidance given a conditional and unconditional output.

        Args:
            cond_output (torch.Tensor): The conditional output.
            uncond_output (torch.Tensor): The unconditional output.

        Returns:
            torch.Tensor: The guided output.
        """
        return cond_output + self.guidance_scale * (cond_output - uncond_output)


class RescaledClassifierFreeGuidance:
    """Implements rescaled classifier free guidance from https://arxiv.org/abs/2305.08891.

    Args:
        guidance_scale (float): The scale of the guidance.
        rescaled_guidance_scale (float): The rescaled guidance scale. Default: ``0.7``.
    """

    def __init__(self, guidance_scale: float, rescaled_guidance_scale: float = 0.7):
        self.guidance_scale = guidance_scale
        self.rescaled_guidance_scale = rescaled_guidance_scale

    def perform_guidance(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """A function that performs rescaled classifier free guidance given a conditional and unconditional output.

        Args:
            cond_output (torch.Tensor): The conditional output.
            uncond_output (torch.Tensor): The unconditional output.

        Returns:
            torch.Tensor: The guided output.
        """
        # First get the CFG output
        cfg_output = cond_output + self.guidance_scale * (cond_output - uncond_output)
        # Then rescale it
        std_pos = torch.std(cond_output, dim=tuple(range(1, cond_output.dim())), keepdim=True)
        std_cfg = torch.std(cfg_output, dim=tuple(range(1, cfg_output.dim())), keepdim=True)
        cfg_output_rescaled = cfg_output * (std_pos / std_cfg)
        return cfg_output_rescaled * self.rescaled_guidance_scale + cfg_output * (1 - self.rescaled_guidance_scale)


class AdaptiveProjectedGuidance:
    """Implements adaptive projected guidance (APG) from https://arxiv.org/abs/2410.02416 as an alternative to CFG.

    Args:
        guidance_scale (float): The scale of the guidance.
        rescaling (float): The rescaling factor (r in the paper). Default: ``2.5``.
        beta (float): The negative momentum value (beta in the paper). Default: ``0.75``.
    """

    def __init__(self, guidance_scale: float, rescaling: float = 2.5, beta: float = 0.75):
        self.guidance_scale = guidance_scale
        self.rescaling = rescaling
        self.beta = beta
        # Initialize the momentum term
        self.dt_beta = 0.0

    def perform_guidance(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """A function that performs adaptive projected guidance given a conditional and unconditional output.

        Args:
            cond_output (torch.Tensor): The conditional output.
            uncond_output (torch.Tensor): The unconditional output.

        Returns:
            torch.Tensor: The guided output.
        """
        # Here just use the perpendicular component
        dt = cond_output - uncond_output
        dot_product = torch.sum(dt * cond_output, dim=tuple(range(1, dt.dim())), keepdim=True)
        norm_squared = torch.sum(uncond_output * cond_output, dim=tuple(range(1, cond_output.dim())),
                                 keepdim=True) + 1.0e-6
        dt_perp = dt - cond_output * dot_product / norm_squared
        dt = dt_perp
        dt = dt * torch.minimum(torch.full_like(dt, 1.0),
                                2.5 / torch.sqrt(torch.sum(dt * dt, tuple(range(1, dt.dim())), keepdim=True)))
        self.dt_beta = dt - 0.75 * self.dt_beta
        return cond_output + (self.guidance_scale - 1) * self.dt_beta


class SLERPGuidance:
    """Implements spherical linear interpolation (SLERP) as an alternative to CFG.

    Args:
        guidance_scale (float): The scale of the guidance.
    """

    def __init__(self, guidance_scale: float):
        self.guidance_scale = guidance_scale

    def perform_guidance(self, cond_output: torch.Tensor, uncond_output: torch.Tensor) -> torch.Tensor:
        """A function that performs spherical linear interpolation given a conditional and unconditional output.

        Args:
            cond_output (torch.Tensor): The conditional output.
            uncond_output (torch.Tensor): The unconditional output.

        Returns:
            torch.Tensor: The guided output.
        """
        cond_norm = torch.sqrt(torch.square(cond_output).sum(dim=tuple(range(1, cond_output.dim())), keepdim=True))
        uncond_norm = torch.sqrt(
            torch.square(uncond_output).sum(dim=tuple(range(1, uncond_output.dim())), keepdim=True))
        cond = cond_output / cond_norm
        uncond = uncond_output / uncond_norm
        angle = torch.arccos((cond * uncond).sum(dim=tuple(range(1, cond.dim())), keepdim=True))
        unit_pred = torch.sin((1 - self.guidance_scale) * angle) / torch.sin(angle) * uncond + torch.sin(
            self.guidance_scale * angle) / torch.sin(angle) * cond
        return unit_pred * cond_norm
