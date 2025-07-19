import torch
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from utils import *
from scipy.optimize import nnls
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import time
import copy
from torch.utils.data import DataLoader


class BaseCoresetSelection(ABC):
    def __init__(
        self, args, train_loader, model, train_loss, optimizer, data_processor
    ):
        self.args = args
        self.train_loader = train_loader
        self.model = model
        self.train_loss = train_loss
        self.optimizer = optimizer
        self.data_processor = data_processor
        self.num_samples = int(
            len(self.train_loader.dataset) * self.args.subset_percentage / 100
        )

    @abstractmethod
    def select_coreset(self):
        pass

    def calculate_gradients(self, model, train_loader, train_loss, device):
        model.eval()
        gradients = []
        for batch_idx, data in tqdm(enumerate(train_loader)):
            data = self.data_processor.preprocess(
                {k: v.to(self.args.device) for k, v in data.items()}
            )
            inputs, targets = data["x"].to(device), data["y"].to(device)
            outputs = model(inputs)
            loss = train_loss(x=inputs, y_pred=outputs, y=targets)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad = list(model.parameters())[-2].grad.view(-1)
            gradients.append(grad.cpu().numpy().astype(np.float16))
            del grad
        gradients = torch.tensor(np.array(gradients)).to(device)
        return gradients

    def _compute_hessian_gradients(
        self, model, train_loader, train_loss, device, num_samples=10
    ):
        model.eval()
        gradients = []

        for batch_idx, data in tqdm(
            enumerate(train_loader), desc="Hessian Gradient Approximation"
        ):
            data = self.data_processor.preprocess(
                {k: v.to(self.args.device) for k, v in data.items()}
            )
            inputs, targets = data["x"].to(device), data["y"].to(device)

            outputs = model(inputs)
            loss = train_loss(x=inputs, y_pred=outputs, y=targets)

            last_param = list(model.parameters())[-2].to(device)
            grad = torch.autograd.grad(loss, last_param, create_graph=True)[0].view(-1)

            diag_estimate = torch.zeros_like(grad)
            for _ in range(num_samples):
                z = torch.randint(0, 2, grad.shape, device=device).float() * 2 - 1
                g_dot_z = torch.dot(grad, z)
                hvp = torch.autograd.grad(g_dot_z, last_param, retain_graph=True)[
                    0
                ].view(-1)
                diag_estimate += z * hvp
            diag_estimate /= num_samples

            inv_hessian_grad = grad / (diag_estimate + 1e-8)

            gradients.append(inv_hessian_grad.detach().cpu().numpy().astype(np.float32))

        gradients = torch.tensor(np.array(gradients)).to(device)
        return gradients

    def calculate_influence_scores(
        self, model, train_loader, train_loss, device, num_samples=10
    ):
        model.eval()
        influence_scores = []

        for batch_idx, data in tqdm(
            enumerate(train_loader), desc="Influence Function Calculation"
        ):
            data = self.data_processor.preprocess(
                {k: v.to(self.args.device) for k, v in data.items()}
            )
            inputs, targets = data["x"].to(device), data["y"].to(device)

            outputs = model(inputs)
            loss = train_loss(x=inputs, y_pred=outputs, y=targets)

            last_param = list(model.parameters())[-2].to(device)
            grad = torch.autograd.grad(loss, last_param, create_graph=True)[0].view(-1)

            diag_estimate = torch.zeros_like(grad)
            for _ in range(num_samples):
                z = torch.randint(0, 2, grad.shape, device=device).float() * 2 - 1
                g_dot_z = torch.dot(grad, z)
                hvp = torch.autograd.grad(g_dot_z, last_param, retain_graph=True)[
                    0
                ].view(-1)
                diag_estimate += z * hvp
            diag_estimate /= num_samples

            inv_hessian_grad = grad / (diag_estimate + 1e-8)

            influence_scores.append(
                -torch.dot(grad.view(-1), inv_hessian_grad.view(-1)).item()
            )

        influence_scores = torch.tensor(influence_scores).to(device)
        return influence_scores


class RandomSelection(BaseCoresetSelection):
    def select_coreset(self):
        coreset_start = time.time()
        indices = (
            torch.randperm(len(self.train_loader))[: self.num_samples].cpu().numpy()
        )
        gamma = {i.item(): len(self.train_loader) // len(indices) for i in indices}
        coreset_end = time.time()
        return list(indices), gamma, coreset_end - coreset_start


class CRAIGSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        similarity_matrix = self._compute_similarity_matrix()
        S, gamma = stocastic_greedy_selection(
            similarity_matrix,
            self.num_samples,
            self.args.sample_size_fraction,
            "bincount",
        )
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start

    def _compute_similarity_matrix(self):
        gradients = self.calculate_gradients(
            self.model,
            self.train_loader,
            self.train_loss,
            self.args.device,
        )
        norms = gradients.pow(2).sum(dim=1, keepdim=True)
        dists_squared = norms + norms.T - 2 * gradients @ gradients.T
        dists_squared = torch.clamp(dists_squared, min=0.0)

        similarity_matrix = -torch.sqrt(dists_squared)
        similarity_matrix.fill_diagonal_(0.0)

        return similarity_matrix.cpu().numpy()


class GradMatchSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        gradients = torch.tensor(
            self.calculate_gradients(
                self.model,
                self.train_loader,
                self.train_loss,
                self.args.device,
            ),
            device=self.args.device,
        )
        indices, weights = self._orthogonal_matching_pursuit(
            gradients.T, torch.mean(gradients, dim=0), self.num_samples
        )
        S = list(indices)
        gamma = {indices[i]: weights[i].item() for i in range(len(S))}
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start

    def _orthogonal_matching_pursuit(self, A, b, budget: int, lam: float = 1.0):
        """approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to:
        https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
        A: design matrix of size (d, n)
        b: measurement vector of length d
        budget: selection budget
        lam: regularization coef. for the final output vector
        Returns:
        vector of length n
        """
        with torch.no_grad():
            d, n = A.shape
            if budget <= 0:
                budget = 0
            elif budget > n:
                budget = n

            # Ensure consistent float32 precision
            A = A.to(torch.float32)
            b = b.to(torch.float32)

            x = np.zeros(n, dtype=np.float32)
            resid = b.clone()
            indices = []
            boolean_mask = torch.ones(n, dtype=bool, device=self.args.device)
            all_idx = torch.arange(n, device=self.args.device)

            for i in tqdm(range(budget)):
                projections = torch.matmul(A.T, resid)
                index = torch.argmax(projections[boolean_mask])
                index = all_idx[boolean_mask][index]

                indices.append(index.item())
                boolean_mask[index] = False

                if len(indices) == 1:
                    A_i = A[:, index].to(torch.float32)
                    x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                    A_i = A[:, index].view(1, -1).to(torch.float32)
                else:
                    A_i = torch.cat(
                        (A_i, A[:, index].view(1, -1).to(torch.float32)), dim=0
                    )

                    # Ensure all matrix operations use float32
                    temp = torch.matmul(
                        A_i, torch.transpose(A_i, 0, 1)
                    ).float() + lam * torch.eye(
                        A_i.shape[0], device=self.args.device, dtype=torch.float32
                    )

                    lstsq_out = torch.linalg.lstsq(
                        temp, torch.matmul(A_i, b).view(-1, 1).float()
                    )

                    x_i = torch.tensor(
                        lstsq_out.solution,
                        device=self.args.device,
                        dtype=torch.float32,
                    )

                resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)

            # Ensure final calculation uses float32 precision
            x_i = nnls(
                temp.cpu().numpy().astype(np.float32),
                torch.matmul(A_i, b).view(-1).cpu().numpy().astype(np.float32),
            )[0]

        return indices, x_i


class GraNdSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        gradients = torch.tensor(
            self.calculate_gradients(
                self.model,
                self.train_loader,
                self.train_loss,
                self.args.device,
            ),
            device=self.args.device,
        )
        normed_gradients = torch.norm(gradients, dim=1)
        sorted_indices = torch.argsort(normed_gradients, descending=True)
        S = list(sorted_indices[i].item() for i in range(self.num_samples))
        gamma = {sorted_indices[i].item(): 1 for i in range(len(S))}
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start


class EL2NSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        loss_values = torch.zeros(len(self.train_loader.dataset))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.train_loader)):
                data = self.data_processor.preprocess(
                    {k: v.to(self.args.device) for k, v in data.items()}
                )
                inputs, targets = data["x"], data["y"]
                inputs, targets = inputs.to(self.args.device), targets.to(
                    self.args.device
                )
                outputs = self.model(inputs)
                loss_norm = torch.norm(
                    self.train_loss(
                        x=inputs, y_pred=outputs, y=targets, reduction="none"
                    ),
                    2,
                )
                loss_values[batch_idx] = loss_norm.cpu()

        _, selected_indices = torch.topk(loss_values, self.num_samples)
        coreset_end = time.time()
        return (
            list(selected_indices.numpy()),
            {i: 1 for i in selected_indices.numpy()},
            coreset_end - coreset_start,
        )


class AdacoreSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        similarity_matrix = self._compute_similarity_matrix()
        S, gamma = stocastic_greedy_selection(
            similarity_matrix,
            self.num_samples,
            self.args.sample_size_fraction,
            "bincount",
        )
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start

    def _compute_similarity_matrix(self):
        gradients = self._compute_hessian_gradients(
            self.model,
            self.train_loader,
            self.train_loss,
            self.args.device,
        )
        norms = gradients.pow(2).sum(dim=1, keepdim=True)
        dists_squared = norms + norms.T - 2 * gradients @ gradients.T
        dists_squared = torch.clamp(dists_squared, min=0.0)

        similarity_matrix = -torch.sqrt(dists_squared)
        similarity_matrix.fill_diagonal_(0.0)

        return similarity_matrix.cpu().numpy()


class InfluenceSelection(BaseCoresetSelection):
    def select_coreset(self):
        self.model = pretrain(
            self.args,
            self.model,
            self.train_loader,
            self.optimizer,
            self.train_loss,
            self.data_processor,
        )
        coreset_start = time.time()
        influence_scores = self.calculate_influence_scores(
            self.model,
            self.train_loader,
            self.train_loss,
            self.args.device,
        )
        top_indices = torch.topk(influence_scores, self.num_samples, largest=True)
        weights = influence_scores.cpu().numpy()
        coreset_end = time.time()
        return (
            top_indices.indices.tolist(),
            weights,
            coreset_end - coreset_start,
        )


class KMeansSelection(BaseCoresetSelection):
    def select_coreset(self):
        coreset_start = time.time()
        input_data = []
        for batch in self.train_loader:
            inputs = batch["x"]
            input_data.append(inputs.cpu().numpy())
        input_data = np.concatenate(input_data, axis=0)
        input_data = input_data.reshape(len(input_data), -1)
        kmeans = KMeans(
            n_clusters=self.num_samples, random_state=self.args.seed, n_init="auto"
        ).fit(input_data)
        selected_indices = self._select_closest_to_centers(
            input_data, kmeans.cluster_centers_
        )
        gamma = self._compute_cluster_weights(kmeans.labels_, selected_indices)
        coreset_end = time.time()
        return selected_indices, gamma, coreset_end - coreset_start

    def _select_closest_to_centers(self, data, centers):
        selected = []
        for i in tqdm(range(len(centers)), desc="Selecting coreset"):
            distances = np.linalg.norm(data - centers[i], axis=1)
            closest = np.argmin(distances)
            selected.append(closest)
        return selected

    def _compute_cluster_weights(self, labels, selected_indices):
        cluster_counts = np.bincount(labels)
        total = cluster_counts.sum()
        gamma = {
            selected_indices[i]: cluster_counts[i] / total
            for i in range(len(selected_indices))
        }
        return gamma


class CosSimilaritySelection(BaseCoresetSelection):
    def select_coreset(self):
        coreset_start = time.time()
        input_data = []
        for batch in self.train_loader:
            inputs = batch["x"]
            input_data.append(inputs.cpu().numpy())
        input_data = np.concatenate(input_data, axis=0)
        input_data = input_data.reshape(len(input_data), -1)
        similarity_matrix = -cdist(input_data, input_data, "cosine")

        S, gamma = stocastic_greedy_selection(
            similarity_matrix,
            self.num_samples,
            self.args.sample_size_fraction,
            "bincount",
        )
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start


class HerdingSelection(BaseCoresetSelection):
    def select_coreset(self):
        coreset_start = time.time()
        input_data = []
        for batch in self.train_loader:
            inputs = batch["x"]
            input_data.append(inputs.cpu().numpy())
        input_data = np.concatenate(input_data, axis=0)
        input_data = input_data.reshape(len(input_data), -1)

        mean_feature = np.mean(input_data, axis=0)
        S = []
        selected_vectors = []
        residual = mean_feature.copy()

        for _ in tqdm(range(self.num_samples), desc="Herding selection"):
            scores = input_data @ residual
            idx = np.argmax(scores)

            while idx in S:
                scores[idx] = -np.inf
                idx = np.argmax(scores)

            S.append(idx)
            selected_vectors.append(input_data[idx])
            residual = mean_feature - np.mean(selected_vectors, axis=0)

        coreset_end = time.time()
        return S, {i: 1 for i in S}, coreset_end - coreset_start


class SimCLRSelection(BaseCoresetSelection):
    def select_coreset(self):
        coreset_start = time.time()
        input_shape = tuple(
            [self.args.in_channels]
            + [self.args.dataset.train_resolution] * self.args.dim
        )
        self.SimCLR = SimCLR(input_shape)
        self.SimCLR.train_simclr(self.args, self.train_loader)
        all_reps = []
        with torch.no_grad():
            for batch in self.train_loader:
                inputs = batch["x"].to(self.args.device)
                outputs = self.SimCLR.projector(self.SimCLR.encoder(inputs).flatten())
                all_reps.append(outputs.cpu())
        all_reps = torch.stack(all_reps, dim=0)
        similarity_matrix = -cdist(all_reps, all_reps, "cosine")
        S, gamma = stocastic_greedy_selection(
            similarity_matrix,
            self.num_samples,
            self.args.sample_size_fraction,
            "uniform",
        )
        coreset_end = time.time()
        return S, gamma, coreset_end - coreset_start


class CoresetSelection:
    def __init__(
        self, args, train_loader, model, train_loss, optimizer, data_processor
    ):
        self.args = args
        self.train_loader = train_loader
        self.model = model
        self.train_loss = train_loss
        self.optimizer = optimizer
        self.data_processor = data_processor

    def coreset_selection(self):
        coreset_algorithms = {
            "random": RandomSelection,
            "craig": CRAIGSelection,
            "gradmatch": GradMatchSelection,
            "el2n": EL2NSelection,
            "graNd": GraNdSelection,
            "adacore": AdacoreSelection,
            "influence": InfluenceSelection,
            "kmeans": KMeansSelection,
            "cosine": CosSimilaritySelection,
            "simclr": SimCLRSelection,
            "herding": HerdingSelection,
        }
        if self.args.coreset_algorithm not in coreset_algorithms:
            raise ValueError(
                f"Unknown coreset selection algorithm: {self.args.coreset_algorithm}"
            )

        selector = coreset_algorithms[self.args.coreset_algorithm](
            self.args,
            self.train_loader,
            self.model,
            self.train_loss,
            self.optimizer,
            self.data_processor,
        )

        coreset = selector.select_coreset()
        torch.cuda.empty_cache()
        return coreset
