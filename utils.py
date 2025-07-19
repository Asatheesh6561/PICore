from neuraloperator.neuralop.models import *
from neuraloperator.neuralop.data.datasets import (
    load_darcy_2d,
    load_ns_incom_2d,
    load_spherical_swe,
    load_advection_1d,
    load_burgers_1d,
    load_ns_com,
)
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import TensorDataset
import os


def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)


def get_model(args):
    model_name = args.model.model_name
    if model_name == "FNO":
        if type(args.dataset.n_modes) == tuple:
            n_modes = args.dataset.n_modes
        else:
            n_modes = tuple([args.dataset.n_modes] * args.dataset.dim)
        model = FNO(
            n_modes=n_modes,
            in_channels=args.dataset.in_channels,
            out_channels=args.dataset.out_channels,
            hidden_channels=args.model.hidden_channels[args.dataset.dim],
            projection_channel_ratio=args.model.projection_channel_ratio,
            factorization=args.model.factorization,
        )
    # elif model_name == "SFNO":
    #     model = SFNO(
    #         n_modes=(
    #             args.dataset.train_resolution
    #             if type(args.dataset.train_resolution) == tuple
    #             else (args.dataset.train_resolution, args.dataset.train_resolution)
    #         ),
    #         in_channels=args.dataset.in_channels,
    #         out_channels=args.dataset.out_channels,
    #         hidden_channels=args.hidden_channels,
    #         projection_channel_ratio=args.projection_channel_ratio,
    #         factorization=args.factorization,
    #     )
    elif model_name == "UNO":
        uno_n_modes = [list(i) * args.dataset.dim for i in args.model.uno_n_modes]
        uno_scalings = [list(i) * args.dataset.dim for i in args.model.uno_scalings]
        uno_out_channels = [i for i in args.model.uno_out_channels]
        model = UNO(
            in_channels=args.dataset.in_channels,
            out_channels=args.dataset.out_channels,
            hidden_channels=args.model.hidden_channels[args.dataset.dim],
            projection_channels=args.model.projection_channels[args.dataset.dim],
            uno_out_channels=uno_out_channels,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            channel_mlp_skip=args.model.channel_mlp_skip,
            n_layers=args.model.n_layers,
            domain_padding=args.model.domain_padding,
        )
    elif model_name == "CNO":
        model = CNO(
            size=args.dataset.train_resolution,
            in_channels=args.dataset.in_channels,
            out_channels=args.dataset.out_channels,
            n_layers=args.model.n_layers,
            dim=args.dataset.dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_dataset(args):
    dataset_name = args.dataset.dataset_name
    # if dataset_name == "SWE":
    #     train_loader, test_loaders = load_spherical_swe(
    #         n_train=args.dataset.train_size,
    #         n_tests=[args.dataset.test_size] * len(args.dataset.test_resolution),
    #         batch_size=1,
    #         test_batch_sizes=[args.dataset.test_batch_size]
    #         * len(args.dataset.test_resolution),
    #         train_resolution=args.dataset.train_resolution,
    #         test_resolutions=args.dataset.test_resolution,
    #     )
    if dataset_name == "Darcy":
        train_loader, test_loaders, data_processor = load_darcy_2d(
            data_root=args.dataset.data_path,
            n_train=args.dataset.train_size,
            n_test=[args.dataset.test_size] * len(args.dataset.test_resolution),
            batch_size=1,
            test_batch_sizes=[args.dataset.test_batch_size]
            * len(args.dataset.test_resolution),
            train_resolution=args.dataset.train_resolution,
            test_resolutions=args.dataset.test_resolution,
        )
    elif dataset_name == "NavierStokesIncompressible":
        train_loader, test_loaders, data_processor = load_ns_incom_2d(
            data_root=args.dataset.data_path,
            n_train=args.dataset.train_size,
            n_test=[args.dataset.test_size] * len(args.dataset.test_resolution),
            batch_size=1,
            test_batch_sizes=[args.dataset.test_batch_size]
            * len(args.dataset.test_resolution),
            train_resolution=args.dataset.train_resolution,
            test_resolutions=args.dataset.test_resolution,
        )
    elif (
        dataset_name == "NavierStokesCompressible1d"
        or dataset_name == "NavierStokesCompressible2d"
    ):
        train_loader, test_loaders, data_processor = load_ns_com(
            data_root=args.dataset.data_path,
            n_train=args.dataset.train_size,
            n_test=[args.dataset.test_size] * len(args.dataset.test_resolution),
            batch_size=1,
            test_batch_sizes=[args.dataset.test_batch_size]
            * len(args.dataset.test_resolution),
            train_resolution=args.dataset.train_resolution,
            test_resolutions=args.dataset.test_resolution,
        )
    elif dataset_name == "Burgers":
        train_loader, test_loaders, data_processor = load_burgers_1d(
            data_root=args.dataset.data_path,
            n_train=args.dataset.train_size,
            n_test=[args.dataset.test_size] * len(args.dataset.test_resolution),
            batch_size=1,
            test_batch_sizes=[args.dataset.test_batch_size]
            * len(args.dataset.test_resolution),
            train_resolution=args.dataset.train_resolution,
            test_resolutions=args.dataset.test_resolution,
        )
    elif dataset_name == "Advection":
        train_loader, test_loaders, data_processor = load_advection_1d(
            data_root=args.dataset.data_path,
            n_train=args.dataset.train_size,
            n_test=[args.dataset.test_size] * len(args.dataset.test_resolution),
            batch_size=1,
            test_batch_sizes=[args.dataset.test_batch_size]
            * len(args.dataset.test_resolution),
            train_resolution=args.dataset.train_resolution,
            test_resolutions=args.dataset.test_resolution,
        )

    data_processor = data_processor.to(args.device)

    if args.attack == "mask":
        train_loader = generate_masked_train_loader(train_loader, args.mask_percentage)
    if args.attack == "blur":
        train_loader = generate_blurred_train_loader(train_loader, args.blur_std)

    return train_loader, test_loaders, data_processor


def mse_loss(y_pred, y, reduction="mean", **kwargs):
    loss = (y_pred - y) ** 2
    loss = loss.mean(dim=tuple(range(1, loss.dim())))
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def rmse_loss(y_pred, y, reduction="mean", **kwargs):
    loss = torch.sqrt((y_pred - y) ** 2)
    loss = loss.mean(dim=tuple(range(1, loss.dim())))
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def nrmse_loss(y_pred, y, reduction="mean", **kwargs):
    mse = (y_pred - y) ** 2
    rmse = torch.sqrt(torch.mean(mse, dim=tuple(range(1, mse.dim()))))
    y_norm = torch.sqrt(torch.mean(y**2, dim=tuple(range(1, y.dim()))))
    nrmse = rmse / y_norm

    if reduction == "mean":
        return torch.mean(nrmse)
    elif reduction == "sum":
        return torch.sum(nrmse)
    elif reduction == "none":
        return nrmse
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def pretrain(args, model, train_loader, optimizer, train_loss, data_processor):
    if os.path.exists(args.pretrain_model_save_path):
        model.load_state_dict(
            torch.load(args.pretrain_model_save_path, weights_only=False)
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=args.dataset.pretrain_batch_size,
            shuffle=True,
        )
        for epoch in tqdm(range(args.pretrain_epochs), desc="Pretraining epochs"):
            model.train()
            mean_loss = 0
            for batch_idx, data in enumerate(dataloader):
                optimizer.zero_grad()
                data = data_processor.preprocess(
                    {k: v.to(args.device) for k, v in data.items()}
                )
                inputs, targets = (
                    data["x"].to(args.device),
                    data["y"].to(args.device),
                )
                outputs = model(inputs)
                loss = train_loss(x=inputs, y_pred=outputs, y=targets)
                mean_loss += loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
                optimizer.step()
            print(f"Pretraining epoch {epoch} loss: {mean_loss / len(dataloader)}")
        save_model(model, args.pretrain_model_save_path)
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


# def calculate_gradients(model, train_loader, train_loss, device):
#     model.eval()
#     gradients = []
#     for batch_idx, data in tqdm(enumerate(train_loader)):
#         inputs, targets = data["x"].to(device), data["y"].to(device)
#         outputs = model(inputs)
#         if train_loss == nrmse_loss:
#             loss = train_loss(outputs, targets)
#         else:
#             loss = train_loss(inputs, outputs, targets)
#         model.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         grad = list(model.parameters())[-2].grad.view(-1)
#         gradients.append(grad.cpu().numpy().astype(np.float16))
#         del grad
#     gradients = torch.tensor(np.array(gradients)).to(device)
#     return gradients


def stocastic_greedy_selection(
    similarity_matrix, k, sample_size_fraction=0.1, gamma_selection="uniform"
):
    S = []
    selected_set = set()
    n = similarity_matrix.shape[0]
    all_indices = np.arange(n)

    max_similarities = np.full(n, -np.inf)

    for _ in tqdm(range(k), desc="Stochastic Greedy Selection"):
        candidate_pool = np.random.choice(
            [i for i in all_indices if i not in selected_set],
            size=max(1, int(sample_size_fraction * n)),
            replace=False,
        )

        sims_to_candidates = similarity_matrix[:, candidate_pool]
        new_max_sims = np.maximum(max_similarities[:, None], sims_to_candidates)
        marginal_gains = np.sum(new_max_sims, axis=0) - np.sum(max_similarities)

        best_idx_in_pool = np.argmax(marginal_gains)
        best_idx = candidate_pool[best_idx_in_pool]
        S.append(best_idx)
        selected_set.add(best_idx)

        max_similarities = np.maximum(max_similarities, similarity_matrix[:, best_idx])

    gamma = get_gamma(similarity_matrix, S, gamma_selection)

    return S, gamma


def get_gamma(similarity_matrix, S, type):
    if type == "uniform":
        gamma = {i: 1 for i in S}
    elif type == "bincount":
        sim_to_selected = similarity_matrix[:, S]
        closest = np.argmax(sim_to_selected, axis=1)
        counts = np.bincount(closest, minlength=len(S))
        gamma = {S[i]: int(counts[i]) for i in range(len(S))}
    else:
        raise ValueError(f"Invalid type: {type}")
    return gamma


def generate_masked_train_loader(train_dataloader, mask_percentage):
    dataset = train_dataloader.dataset
    for i in range(len(dataset.x)):
        mask = (torch.rand_like(dataset.x[i]) > mask_percentage).float()
        dataset.x[i] = dataset.x[i] * mask

    masked_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return masked_loader


def generate_blurred_train_loader(train_dataloader, sigma):
    dataset = train_dataloader.dataset
    for i in range(len(dataset.x)):
        blur = gaussian_blur(
            dataset.x[i].unsqueeze(0), dim=len(dataset.x[i].shape) - 1, sigma=sigma
        )
        dataset.x[i] = blur

    blurred_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return blurred_loader


def gaussian_blur(x, dim, sigma=0.5):
    B, C = x.shape[:2]

    def get_kernel1d(sigma, kernel_size):
        center = kernel_size // 2
        x = torch.arange(kernel_size) - center
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel.to(x.device)

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create Gaussian kernel
    kernel_1d = get_kernel1d(sigma, kernel_size).to(x.device)

    if dim == 1:
        kernel = kernel_1d.view(1, 1, -1).repeat(C, 1, 1)
        return torch.nn.functional.conv1d(x, kernel, padding=kernel_size // 2, groups=C)

    elif dim == 2:
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, *kernel_2d.shape).repeat(C, 1, 1, 1)
        return torch.nn.functional.conv2d(x, kernel, padding=kernel_size // 2, groups=C)

    elif dim == 3:
        kernel_3d = (
            kernel_1d[:, None, None]
            * kernel_1d[None, :, None]
            * kernel_1d[None, None, :]
        )
        kernel = kernel_3d.view(1, 1, *kernel_3d.shape).repeat(C, 1, 1, 1, 1)
        return torch.nn.functional.conv3d(x, kernel, padding=kernel_size // 2, groups=C)

    else:
        raise ValueError("dim must be 1, 2, or 3")


class SimCLR(nn.Module):
    def __init__(self, input_shape, projection_dim=256, temperature=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.dim = len(input_shape) - 1
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self._build_encoder(input_shape)
        self.projector = nn.Sequential(
            nn.Linear(self._encoder_out_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )
        self.criterion = nn.CrossEntropyLoss()

    def _build_encoder(self, input_shape):
        C = input_shape[0]
        layers = []
        conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        norm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[self.dim]
        for i in range(4):
            out_channels = 32 * (2**i)
            layers += [
                conv(
                    C if i == 0 else 32 * (2 ** (i - 1)),
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                norm(out_channels),
                nn.ReLU(),
            ]
        return nn.Sequential(*layers)

    def _encoder_out_dim(self):
        dummy = torch.zeros(1, *self.input_shape)
        out = self.encoder(dummy)
        return int(torch.flatten(out, 1).shape[1])

    def _augment(self, x):
        return gaussian_blur(x, self.dim, sigma=0.5)

    def forward(self, x):
        x1, x2 = self._augment(x), self._augment(x)
        z1 = self.projector(torch.flatten(self.encoder(x1), 1))
        z2 = self.projector(torch.flatten(self.encoder(x2), 1))
        return self._contrastive_loss(z1, z2)

    def _contrastive_loss(self, z1, z2):
        z1, z2 = torch.nn.functional.normalize(
            z1, dim=1
        ), torch.nn.functional.normalize(z2, dim=1)
        N = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        labels = torch.arange(N).to(z1.device)
        labels = torch.cat([labels + N, labels])
        mask = torch.eye(2 * N, dtype=torch.bool).to(z1.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        logits = sim_matrix
        targets = labels
        return self.criterion(logits, targets)

    def train_simclr(self, args, dataloader, lr=1e-3, weight_decay=1e-6):
        self.to(self.device)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        dataloader = DataLoader(
            dataloader.dataset, batch_size=args.pretrain_batch_size, shuffle=True
        )
        for epoch in tqdm(range(args.simclr_epochs)):
            self.train()
            for x in dataloader:
                x = x["x"].to(self.device)
                optimizer.zero_grad()
                loss = self(x)
                loss.backward()
                optimizer.step()
