import torch
import math


class GaussianRF(object):
    def __init__(
        self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None
    ):
        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))

        k_max = size // 2

        if dim == 1:
            k = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )

            self.sqrt_eig = (
                size
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi**2) * (k**2) + tau**2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, 1)

            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers

            self.sqrt_eig = (
                (size**2)
                * math.sqrt(2.0)
                * sigma
                * ((4 * (math.pi**2) * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2.0))
            )
            self.sqrt_eig[0, 0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            ).repeat(size, size, 1)

            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)

            self.sqrt_eig = (
                (size**3)
                * math.sqrt(2.0)
                * sigma
                * (
                    (4 * (math.pi**2) * (k_x**2 + k_y**2 + k_z**2) + tau**2)
                    ** (-alpha / 2.0)
                )
            )
            self.sqrt_eig[0, 0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        # Generate random coefficients (real and imaginary parts)
        coeff = torch.randn(N, *self.size, 2, device=self.device)

        # Multiply by sqrt of eigenvalues
        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]

        # Convert to complex tensor
        coeff_complex = torch.view_as_complex(coeff)

        # Perform inverse FFT
        if self.dim == 1:
            u = torch.fft.irfft(coeff_complex, n=self.size[0], norm="backward")
        elif self.dim == 2:
            u = torch.fft.irfft2(coeff_complex, s=self.size, norm="backward")
        elif self.dim == 3:
            u = torch.fft.irfftn(coeff_complex, s=self.size, norm="backward")

        return u
