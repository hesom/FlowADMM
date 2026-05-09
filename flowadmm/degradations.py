# This file includes code derived from the upstream PnP-Flow repository:
# https://github.com/annegnx/PnP-Flow
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn.functional as F

from flowadmm.utils import (
    bicubic_filter,
    create_downsampling_matrix,
    downsample,
    gaussian_2d_kernel,
    upsample,
)


class Degradation:
    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        return None


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        inv_nu = 1.0 / (nu + 1e-10)
        inv_sy2 = 1.0 / (sigma_y ** 2)
        return (inv_nu * x_hat + inv_sy2 * y) / (inv_nu + inv_sy2)


class BoxInpainting(Degradation):
    def __init__(self, half_size_mask):
        self.half_size_mask = half_size_mask
        self._mask = None

    def _get_mask(self, x):
        if self._mask is None or self._mask.shape != x.shape:
            d = x.shape[2] // 2
            mask = torch.ones_like(x)
            mask[:, :, d - self.half_size_mask:d + self.half_size_mask, d - self.half_size_mask:d + self.half_size_mask] = 0
            self._mask = mask
        return self._mask

    def H(self, x):
        return self._get_mask(x) * x

    def H_adj(self, x):
        return self._get_mask(x) * x

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        mask = self._mask
        if mask is None:
            return None
        inv_nu = 1.0 / (nu + 1e-10)
        inv_sy2 = 1.0 / (sigma_y ** 2)
        rhs = inv_nu * x_hat + inv_sy2 * mask * y
        denom = inv_nu + inv_sy2 * mask
        return rhs / denom


class RandomInpainting(Degradation):
    def __init__(self, p):
        self.p = p
        self._mask = None

    def _get_mask(self, x):
        if self._mask is None or self._mask.shape[0] != x.shape[0] or self._mask.shape[2:] != x.shape[2:]:
            np.random.seed(42)
            mask = torch.from_numpy(
                np.random.binomial(n=1, p=1 - self.p, size=(x.shape[0], x.shape[2], x.shape[3]))
            ).to(x.device)
            self._mask = mask.unsqueeze(1).float()
        return self._mask

    def H(self, x):
        return self._get_mask(x) * x

    def H_adj(self, x):
        return self._get_mask(x) * x

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        mask = self._mask
        if mask is None:
            return None
        inv_nu = 1.0 / (nu + 1e-10)
        inv_sy2 = 1.0 / (sigma_y ** 2)
        rhs = inv_nu * x_hat + inv_sy2 * mask * y
        denom = inv_nu + inv_sy2 * mask
        return rhs / denom


class GaussianDeblurring(Degradation):
    def __init__(
        self,
        sigma_blur,
        kernel_size,
        mode="fft",
        num_channels=3,
        dim_image=128,
        device="cuda",
    ) -> None:
        self.mode = mode
        self.sigma = sigma_blur
        self.kernel_size = kernel_size
        self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
        filt = torch.zeros((1, num_channels, dim_image, dim_image), device=device)
        filt[..., :kernel_size, :kernel_size] = self.kernel
        self.filter = torch.roll(
            filt, shifts=(-(kernel_size - 1) // 2, -(kernel_size - 1) // 2), dims=(2, 3)
        )
        self.device = device
        self._h_f = torch.fft.fft2(self.filter)

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding="same", groups=x.shape[1])
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x.to(self.device)) * self._h_f))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding="same", groups=x.shape[1])
        return torch.real(
            torch.fft.ifft2(torch.fft.fft2(x.to(self.device)) * torch.conj(self._h_f))
        )

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        if self.mode != "fft":
            return None
        inv_nu = 1.0 / (nu + 1e-10)
        inv_sy2 = 1.0 / (sigma_y ** 2)
        x_f = torch.fft.fft2(x_hat.to(self.device))
        y_f = torch.fft.fft2(y.to(self.device))
        rhs_f = inv_nu * x_f + inv_sy2 * torch.conj(self._h_f) * y_f
        denom_f = inv_nu + inv_sy2 * (torch.abs(self._h_f) ** 2)
        return torch.real(torch.fft.ifft2(rhs_f / denom_f))


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda") -> None:
        self.sf = sf
        self.mode = mode
        if mode == "bicubic":
            self.filter = torch.nn.Parameter(bicubic_filter(sf), requires_grad=False).to(device)
            filt = torch.zeros((1, 3, dim_image, dim_image), device=device)
            filt[..., : self.filter.shape[-1], : self.filter.shape[-1]] = self.filter
            self.filter = torch.roll(
                filt,
                shifts=(-(self.filter.shape[-1] - 1) // 2, -(self.filter.shape[-1] - 1) // 2),
                dims=(2, 3),
            )
        self.downsampling_matrix = create_downsampling_matrix(dim_image, dim_image, sf, device)

    def H(self, x):
        if self.mode is None:
            return downsample(x, self.sf)
        x_blur = torch.real(torch.fft.ifft2(torch.fft.fft2(x) * torch.fft.fft2(self.filter)))
        return downsample(x_blur, self.sf)

    def H_adj(self, x):
        if self.mode is None:
            return upsample(x, self.sf)
        x_up = upsample(x, self.sf)
        return torch.real(
            torch.fft.ifft2(torch.fft.fft2(x_up) * torch.conj(torch.fft.fft2(self.filter)))
        )

    def wiener_solve(self, x_hat, y, sigma_y, nu):
        if self.mode is not None:
            return None
        sf = self.sf
        inv_nu = 1.0 / (nu + 1e-10)
        inv_sy2 = 1.0 / (sigma_y ** 2)
        rhs = inv_nu * x_hat
        rhs[..., ::sf, ::sf] = rhs[..., ::sf, ::sf] + inv_sy2 * y
        denom = torch.full_like(x_hat, inv_nu)
        denom[..., ::sf, ::sf] = denom[..., ::sf, ::sf] + inv_sy2
        return rhs / denom
