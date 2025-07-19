import torch
import torch.nn.functional as F
import numpy as np


class LpLoss(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    """central_diff_2d computes derivatives
    df(x,y)/dx and df(x,y)/dy for f(x,y) defined
    on a regular 2d grid using finite-difference

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j] = f(x_i, y_j)
    h : float or list
        discretization size of grid for each dimension
    fix_x_bnd : bool, optional
        whether to fix dx on the x boundaries, by default False
    fix_y_bnd : bool, optional
        whether to fix dy on the y boundaries, by default False

    Returns
    -------
    dx, dy
        tuple such that dx[:, i,j]= df(x_i,y_j)/dx
        and dy[:, i,j]= df(x_i,y_j)/dy
    """
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2)) / (2.0 * h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1)) / (2.0 * h[1])

    if fix_x_bnd:
        dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / h[0]
        dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / h[0]

    if fix_y_bnd:
        dy[..., :, 0] = (x[..., :, 1] - x[..., :, 0]) / h[1]
        dy[..., :, -1] = (x[..., :, -1] - x[..., :, -2]) / h[1]

    return dx, dy


class PILoss:
    def __init__(self, dataset_name, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == "Burgers":
            self.loss = BurgersEqnLoss(**kwargs)
        elif self.dataset_name == "Darcy":
            self.loss = DarcyEqnLoss(**kwargs)
        elif self.dataset_name == "Advection":
            self.loss = AdvectionEqnLoss(**kwargs)
        elif self.dataset_name == "NavierStokesIncompressible":
            self.loss = NavierStokesEqnLoss(**kwargs)

    def __call__(self, y_pred, **kwargs):
        return self.loss(y_pred, **kwargs)


class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(
        self, visc=0.01, method="fdm", loss=F.mse_loss, domain_length=1.0, **kwargs
    ):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u, reduction="mean"):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # du/dt and du/dx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # d^2u/dxx
        dudxx = (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / dx**2
        # fix boundary
        dudxx[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / dx**2
        dudxx[..., -1] = (u[..., -1] - 2 * u[..., -2] + u[..., -3]) / dx**2

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(right_hand_side, dudt, reduction=reduction)

    def __call__(self, y_pred, reduction="mean", **kwargs):
        if self.method == "fdm":
            return self.fdm(u=y_pred, reduction=reduction)
        raise NotImplementedError()


class DarcyEqnLoss(object):
    """
    Computes loss for Darcy's equation.
    """

    def __init__(self, loss=F.mse_loss, **kwargs):
        super().__init__()
        self.loss = loss

    def fdm(self, u, a, domain_length=1, reduction="mean"):
        # remove extra channel dimensions
        a = a[:, 0, :, :]

        u = u[:, 0, :, :]

        # compute the left hand side of the Darcy Flow equation
        # note: here we assume that the input is a regular grid
        n = u.size(1)
        dx = domain_length / (n - 1)
        dy = dx
        ux, uy = central_diff_2d(u, [dx, dy], fix_x_bnd=False, fix_y_bnd=False)
        a_ux = a * ux
        a_uy = a * uy

        a_uxx, _ = central_diff_2d(a_ux, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)
        _, a_uyy = central_diff_2d(a_uy, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        left_hand_side = -(a_uxx + a_uyy)
        left_hand_side = left_hand_side[:, 2:-2, 2:-2]

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        loss = self.loss(left_hand_side, forcing_fn, reduction=reduction)

        del ux, uy, a_ux, a_uy, a_uxx, a_uyy
        return loss

    def __call__(self, y_pred, x, **kwargs):
        return self.fdm(y_pred, x)


class AdvectionEqnLoss(object):
    def __init__(
        self, velocity=1.0, method="fdm", loss=F.mse_loss, domain_length=1.0, **kwargs
    ):
        super().__init__()
        self.velocity = velocity
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u, reduction="mean"):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # Compute derivatives
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # right hand side (advection equation)
        right_hand_side = -self.velocity * dudx

        # compute the loss of the left and right hand sides
        return self.loss(right_hand_side, dudt, reduction=reduction)

    def __call__(self, y_pred, reduction="mean", **kwargs):
        if self.method == "fdm":
            return self.fdm(u=y_pred, reduction=reduction)
        raise NotImplementedError()


class ICLoss(object):
    """
    Computes loss for initial value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def initial_condition_loss(self, y_pred, x, reduction="mean"):
        x = x.squeeze(1)
        y_pred = y_pred.squeeze(1)
        boundary_true = x[:, 0, :]
        boundary_pred = y_pred[:, 0, :]
        return self.loss(boundary_pred, boundary_true, reduction=reduction)

    def __call__(self, y_pred, x, reduction="mean", **kwargs):
        return self.initial_condition_loss(y_pred, x, reduction=reduction)


class NavierStokesEqnLoss(object):
    def __init__(self, loss=F.mse_loss, method="fdm", resolution=64):
        super().__init__()
        self.loss = loss
        self.method = method
        self.resolution = resolution

    def fdm(self, w, v=1 / 40, t_interval=1.0, reduction="mean"):
        batchsize = w.size(0)
        nx = w.size(3)
        ny = w.size(4)
        nt = w.size(2)
        device = w.device
        w = w.reshape(batchsize, nx, ny, nt)

        w_h = torch.fft.fft2(w, dim=[1, 2])
        # Wavenumbers in y-direction
        k_max = nx // 2
        N = nx
        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(N, 1)
            .repeat(1, N)
            .reshape(1, N, N, 1)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(1, N)
            .repeat(N, 1)
            .reshape(1, N, N, 1)
        )
        # Negative Laplacian in Fourier space
        lap = k_x**2 + k_y**2
        lap[0, 0, 0, 0] = 1.0
        f_h = w_h / lap

        ux_h = 1j * k_y * f_h
        uy_h = -1j * k_x * f_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        ux = torch.fft.irfft2(ux_h[:, :, : k_max + 1], dim=[1, 2])
        uy = torch.fft.irfft2(uy_h[:, :, : k_max + 1], dim=[1, 2])
        wx = torch.fft.irfft2(wx_h[:, :, : k_max + 1], dim=[1, 2])
        wy = torch.fft.irfft2(wy_h[:, :, : k_max + 1], dim=[1, 2])
        wlap = torch.fft.irfft2(wlap_h[:, :, : k_max + 1], dim=[1, 2])

        dt = t_interval / (nt - 1)
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

        Du1 = wt + (ux * wx + uy * wy - v * wlap)[..., 1:-1]  # - forcing
        forcing = (
            self.get_forcing(self.resolution)
            .repeat(Du1.shape[0], 1, 1, Du1.shape[-1])
            .to(Du1.device)
        )
        return self.loss(Du1, forcing, reduction=reduction)

    def get_forcing(self, S):
        x1 = (
            torch.tensor(
                np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float
            )
            .reshape(S, 1)
            .repeat(1, S)
        )
        x2 = (
            torch.tensor(
                np.linspace(0, 2 * np.pi, S, endpoint=False), dtype=torch.float
            )
            .reshape(1, S)
            .repeat(S, 1)
        )
        return -4 * (torch.cos(4 * (x2))).reshape(1, S, S, 1)

    def __call__(self, y_pred, reduction="mean", **kwargs):
        if self.method == "fdm":
            return self.fdm(y_pred, reduction=reduction)
        raise NotImplementedError()


class WeightedSumLoss(object):
    """
    Computes an average or weighted sum of given losses.
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if not len(weights) == len(losses):
            raise ValueError("Each loss must have a weight.")
        self.losses = list(zip(losses, weights))

    def __call__(self, *args, **kwargs):
        weighted_losses = [
            float(weight) * loss(*args, **kwargs) for loss, weight in self.losses
        ]
        if "reduction" in kwargs and kwargs["reduction"] == "none":
            return torch.cat([i.flatten() for i in weighted_losses])
        else:
            return sum(weighted_losses)

    def __str__(self):
        description = "Combined loss: "
        for loss, weight in self.losses:
            description += f"{loss} (weight: {weight}) "
        return description
