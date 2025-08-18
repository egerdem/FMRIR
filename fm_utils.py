from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from torchvision import datasets, transforms
import wandb
# import matplotlib
# matplotlib.use('Qt5Agg', force=True)   # or 'TkAgg'
# import matplotlib.pyplot as plt
import random

# early stopping taken from: https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/utils.py#L72

class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):

        metrics_val = metrics.cpu().item()
        if self.best is None:
            self.best = metrics_val
            return False

        if np.isnan(metrics_val):
            return True

        if self.is_better(metrics_val, self.best):
            self.num_bad_epochs = 0
            self.best = metrics_val
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta

class OldSampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
        """
        pass


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """

    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples)  # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)  # (num_samples, c, h, w)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        pass


class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)


class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        return torch.ones_like(t)


class LinearBeta(Beta):
    """
    Implements beta_t = 1-t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        return - torch.ones_like(t)


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        alpha_t = self.alpha(t)  # (num_samples, 1, 1, 1)
        beta_t = self.beta(t)  # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t)  # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t)  # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - dt: time, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + dt (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

    @torch.no_grad()
    def simulate_trajectory(self, x: torch.Tensor, max_timesteps: int, y: torch.Tensor):
        """
        Simulates the ODE and returns the state at each timestep.
        """
        ts = torch.linspace(0, 1, max_timesteps + 1).to(x.device)
        trajectory = [x.clone()]
        
        for i in range(max_timesteps):
            t_current = ts[i]
            t_next = ts[i+1]
            h = t_next - t_current
            
            # Reshape t for drift_coefficient
            t_reshaped = t_current.view(1, 1, 1, 1).expand(x.shape[0], -1, -1, -1)

            x = self.step(x, t_reshaped, h, y=y)
            trajectory.append(x.clone())
            
        return torch.stack(trajectory)


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.sde.drift_coefficient(xt, t, **kwargs) * h + self.sde.diffusion_coefficient(xt, t,
                                                                                                     **kwargs) * torch.sqrt(
            h) * torch.randn_like(xt)


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    # **NEW: Abstract method for validation loss**
    @torch.no_grad()
    @abstractmethod
    def get_valid_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_iterations: int, device: torch.device, lr: float,
              valid_sampler: Optional[Sampleable] = None,
              save_path: str = "model.pt",
              checkpoint_path: str = "checkpoints",
              validation_interval: Optional[int] = None,
              checkpoint_interval: Optional[int] = None,
              start_iteration: int = 0,
              config: dict = None,
              early_stopping_patience: int = 3000,
              resume_checkpoint_path: Optional[str] = None,
              resume_checkpoint_state: Optional[dict] = None,
              **kwargs):

        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)

        # --- State Tracking ---
        best_val_loss = float("inf")
        best_iteration = start_iteration

        # NEW: Initialize the EarlyStopping monitor
        early_stopper = EarlyStopping(patience=early_stopping_patience)

        # Unified resume logic: load from an explicit checkpoint path/state if provided
        checkpoint = None
        if resume_checkpoint_state is not None:
            checkpoint = resume_checkpoint_state
            print("Resuming from provided in-memory checkpoint state")
        elif resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
            print(f"Loading checkpoint from {resume_checkpoint_path}")
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)

        if checkpoint is not None:
            # Restore model weights if present
            model_state = checkpoint.get('model_state_dict')
            if model_state is not None:
                self.model.load_state_dict(model_state)

            # Restore trainer-specific state if present (e.g., y_null)
            if hasattr(self, 'y_null') and checkpoint.get('y_null') is not None:
                self.y_null.data = checkpoint['y_null'].to(device)

            # Restore optimizer if present
            if checkpoint.get('optimizer_state_dict') is not None:
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored from checkpoint.")

                print(f"Overwriting optimizer LR with new command-line value: {lr}")
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

            # Adopt iteration and best metrics if available
            iter_value = checkpoint.get('iteration', None)
            if isinstance(iter_value, (int, float)):
                start_iteration = int(iter_value)
                print("starting from iteration", start_iteration)
            else:
                start_iteration = checkpoint["config"]["training"].get("num_iterations") + 1
                print("starting from iteration", start_iteration)
            if start_iteration is None or not isinstance(start_iteration, int) or start_iteration <= 0:
                assert start_iteration >= 0, "start_iteration must be a non-negative integer"

            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            best_iteration = checkpoint.get('best_iteration', best_iteration)
            print(f"Resumed state. start_iteration={start_iteration}, best_val_loss={best_val_loss}, best_iteration={best_iteration}")

        batch_size = kwargs.get('batch_size')

        # **NEW: Get total dataset size for epoch calculation**
        # dataset_size = len(self.path.p_data.spectrograms)
        dataset_size = len(self.path.p_data)

        pbar = tqdm(range(start_iteration, num_iterations))
        for iteration in pbar:
            self.model.train()
            opt.zero_grad()

            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()

            # **MODIFICATION: Calculate and display the current epoch number**
            current_epoch = (iteration + 1) * batch_size / dataset_size
            wandb.log({"train_loss": loss.item(), "epoch": current_epoch, "iteration": iteration})

            # **NEW: Validation loop**
            if valid_sampler and (iteration + 1) % validation_interval == 0:
                self.model.eval()
                val_loss = self.get_valid_loss(valid_sampler=valid_sampler, **kwargs)
                pbar.set_description(
                    f'Epoch: {current_epoch:.4f}, Iter: {iteration}, Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}')
                # **NEW: Log validation loss to wandb**
                wandb.log({"val_loss": val_loss.item(), "epoch": current_epoch, "iteration": iteration})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(
                        f"** [Iter {iteration}] New best val. loss found for: train loss: {loss.item():.5f} and val loss: {best_val_loss:.5f}. Saving model. **")
                    # Save best model state for inference
                    best_model_state = {
                        'model_state_dict': self.model.state_dict(),
                        'y_null': getattr(self, 'y_null', None),
                        'best_val_loss': best_val_loss,
                        'best_iteration': iteration,
                        'config': config,
                        'wandb_run_id': config.get('wandb_run_id'),
                        'is_best': True  # Flag to indicate this is best model
                    }
                    # Stable pointer to current best (guard against interruptions)
                    torch.save(best_model_state, save_path)
                    best_iteration = iteration  # Update global tracking

                # NEW: Check for early stopping
                if early_stopper.step(val_loss):
                    print(f"--- Early stopping triggered at iteration {iteration} with val_loss: {val_loss}---")
                    flag_save = True
                    break  # Exit the training loop

            else:
                pbar.set_description(f'Epoch: {current_epoch:.2f}, Iter: {iteration}, Loss: {loss.item():.5f}')

            # --- Periodic Checkpointing Logic ---
            if (iteration + 1) % checkpoint_interval == 0:
                print(f"\n--- Saving checkpoint at iteration {iteration + 1} ---")
                ckpt_save_path = os.path.join(checkpoint_path, f"ckpt_{iteration + 1}.pt")
                # Save checkpoint for resuming training (latest state)
                checkpoint_state = {
                    'iteration': iteration + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_iteration': best_iteration,
                    'y_null': getattr(self, 'y_null', None),
                    'config': config,
                    'wandb_run_id': config.get('wandb_run_id'),
                    'is_best': False  # Flag to indicate this is latest checkpoint
                }
                torch.save(checkpoint_state, ckpt_save_path)

        # --- Save final checkpoint ---
        final_iteration = iteration + 1
        if final_iteration == num_iterations or flag_save:

            final_ckpt_path = os.path.join(checkpoint_path, f"ckpt_final_{final_iteration}.pt")
            print(f"\n--- Saving final checkpoint at iteration {final_iteration} to {final_ckpt_path} ---")
            final_checkpoint_state = {
                'iteration': final_iteration,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
                'best_iteration': best_iteration,
                'y_null': getattr(self, 'y_null', None),
                'config': config,
                'wandb_run_id': config.get('wandb_run_id'),
                'is_best': False,
                'is_final': True  # Flag to indicate this is the final state
            }
            torch.save(final_checkpoint_state, final_ckpt_path)

            # Additionally, save a "last" checkpoint alias for easy resume
            # ckpt_last_versioned = os.path.join(checkpoint_path, f"ckpt_last_{final_iteration}.pt")
            # ckpt_last_alias = os.path.join(checkpoint_path, "ckpt_last.pt")
            # torch.save(final_checkpoint_state, ckpt_last_versioned)
            # torch.save(final_checkpoint_state, ckpt_last_alias)

            # Leave only model.pt as the stable best artifact (no versioned copy)

        self.model.eval()
        print(f"--- Training finished. Best validation loss was {best_val_loss:.5f} at iteration {best_iteration}. ---")


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """

    def __init__(self):
        super().__init__()
        # Try to handle SSL certificate issues
        import ssl
        import urllib.request

        # Create unverified SSL context as a workaround
        ssl._create_default_https_context = ssl._create_unverified_context

        try:
            self.dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
            )
        except Exception as e:
            print(f"Error downloading MNIST: {e}")
            print("Please download MNIST manually or check your SSL certificates")
            raise e

        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


class SpectrogramSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the RIR Spectrogram dataset.
    Splits data into train/valid/test based on hardcoded source indices.
    """

    def __init__(self, data_path: str, mode: str, src_splits: Dict, transform: Optional[callable] = None):
        super().__init__()
        import os

        self.transform = transform
        self.mode = mode
        self.src_splits = src_splits

        # Check if a pre-processed file exists
        # processed_file = os.path.join(os.path.dirname(data_path), 'processed_spectrograms.pt')
        processed_file = os.path.join(data_path, f'processed_{self.mode}.pt')

        re_process = True
        if os.path.exists(processed_file):
            print(f"Loading pre-processed {self.mode} data from {processed_file}")
            data = torch.load(processed_file)
            if 'sample_info' in data:
                self.spectrograms = data['spectrograms']
                self.coords = data['coords']
                self.sample_info = data['sample_info']
                re_process = False
            else:
                print(f"Cached file {processed_file} is outdated. Re-processing.")

        if re_process:
            print(f"Processing {self.mode} data from .npz files...")

            all_spectrograms = []
            all_coords = []
            all_sample_info = []

            source_indices = range(*self.src_splits[self.mode])

            for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
                # Construct file path based on source index
                npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")

                if not os.path.exists(npz_file):
                    print(f"Warning: File not found {npz_file}, skipping.")
                    continue

                with np.load(npz_file) as data:
                    specs = data['spec']  # Shape: (1331, 16, 16)
                    source_pos = data['posSrc']  # Shape: (3,)
                    mic_pos = data['posMic']  # Shape: (1331, 3)

                    # Log-magnitude conversion
                    log_specs = 10 * np.log10(specs + 1e-8)

                    for i in range(log_specs.shape[0]):
                        all_spectrograms.append(torch.tensor(log_specs[i], dtype=torch.float32))
                        # Create the 6D coordinate vector [xs, ys, zs, xm, ym, zm]
                        coord_vec = np.concatenate([source_pos, mic_pos[i]])
                        all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
                        all_sample_info.append(torch.tensor([src_id, i], dtype=torch.int32))

            if not all_spectrograms:
                raise ValueError(f"No data loaded for mode '{self.mode}'. Check file paths and splits.")

            # Stack all spectrograms and coordinates into tensors
            self.spectrograms = torch.stack(all_spectrograms)
            self.coords = torch.stack(all_coords)
            self.sample_info = torch.stack(all_sample_info)

            # Save the processed tensors for faster loading next time
            torch.save({'spectrograms': self.spectrograms, 'coords': self.coords, 'sample_info': self.sample_info},
                       processed_file)
            print(f"Saved processed {self.mode} data to {processed_file}")

        self.dummy = nn.Buffer(torch.zeros(1))
        print(
            f"Loaded {len(self.spectrograms) / 1331} * {1331} = {len(self.spectrograms)} spectrograms for {self.mode} set.")
        print(f"Spectrogram tensor shape: {self.spectrograms.shape}")
        print(f"Coordinate tensor shape: {self.coords.shape}")

    def __len__(self):
        return len(self.spectrograms)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, 1, H, W)
            - labels: shape (batch_size, 6) for coordinates
        """
        if num_samples > len(self.spectrograms):
            # Sample with replacement if requesting more than available
            indices = torch.randint(0, len(self.spectrograms), (num_samples,))
        else:
            indices = torch.randperm(len(self.spectrograms))[:num_samples]

        samples = self.spectrograms[indices]
        labels = self.coords[indices]

        # Apply transformations if they exist
        if self.transform:
            samples = self.transform(samples)

        # Add channel dimension and move to device
        return samples.unsqueeze(1).to(self.dummy.device), labels.to(self.dummy.device)

    def get_item_by_idx(self, item_idx: int):
        """Gets a single item (spectrogram, coords, info) by its flat index."""
        sample = self.spectrograms[item_idx]
        label = self.coords[item_idx]
        info = self.sample_info[item_idx] if self.sample_info is not None else None

        if self.transform:
            # Apply transform to a single sample. We need to add a batch dim and remove it.
            sample = self.transform(sample.unsqueeze(0)).squeeze(0)

        return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device), info.unsqueeze(
            0).to(self.dummy.device) if info is not None else None

    def find_sample_index(self, src_id: int, mic_id: int):
        """Finds the flat index for a given source and mic ID."""
        if self.sample_info is None:
            return None
        # self.sample_info is a tensor of shape [N, 2] where each row is (src_id, mic_id)
        results = (self.sample_info[:, 0] == src_id) & (self.sample_info[:, 1] == mic_id)
        indices = torch.where(results)[0]
        return indices[0].item() if len(indices) > 0 else None

class ATFSliceSampler(torch.nn.Module, Sampleable):
    """
    Loads and serves 2D spatial slices of ATF magnitudes.

    Each sample is a tensor of shape (64, 11, 11), representing the
    64 frequency bins for an 11x11 grid of microphones at a single height.
    """
    def __init__(self, data_path: str, mode: str, src_splits: dict, transform: Optional[callable] = None,
                 freq_up_to: Optional[int] = None):
        super().__init__()
        self.transform = transform
        self.mode = mode
        self.src_splits = src_splits
        self.freq_up_to = freq_up_to

        processed_file = os.path.join(data_path, f'processed_atf_{self.mode}.pt')

        if os.path.exists(processed_file):
            print(f"Loading pre-processed ATF {self.mode} data from {processed_file}")
            data = torch.load(processed_file)
            self.slices = data['slices']
            self.coords = data['coords']
            self.sample_info = data.get('sample_info')
        else:
            print(f"Processing ATF {self.mode} data from .npz files...")
            source_indices = range(*src_splits[self.mode])
            all_slices = []
            all_coords = []
            # **NEW: Create a list to store metadata**
            all_sample_info = []

            for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
                npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
                with np.load(npz_file) as data:
                    atf_mags = data['atf_mag_algn']   # Shape: (1331, 64)
                    mic_pos = data['posMic']          # Shape: (1331, 3)
                    source_pos = data['posSrc']       # Shape: (3,)

                    unique_z = np.unique(mic_pos[:, 2])

                    for z_val in unique_z:
                        slice_indices = np.where(mic_pos[:, 2] == z_val)[0]
                        mic_pos_slice = mic_pos[slice_indices]
                        atf_mags_slice = atf_mags[slice_indices]

                        unique_x = sorted(np.unique(mic_pos_slice[:, 0]))
                        unique_y = sorted(np.unique(mic_pos_slice[:, 1]))
                        nx, ny = len(unique_x), len(unique_y)

                        if nx * ny != len(mic_pos_slice):
                            print(f"Warning: Skipping slice for src_id {src_id} at z={z_val} due to irregular grid.")
                            continue

                        x_map = {val: i for i, val in enumerate(unique_x)}
                        y_map = {val: i for i, val in enumerate(unique_y)}

                        # Pre-allocate for full frequency dimension; we'll crop later if requested
                        grid_slice = torch.zeros((64, ny, nx), dtype=torch.float32)
                        for i in range(len(mic_pos_slice)):
                            ix, iy = x_map[mic_pos_slice[i, 0]], y_map[mic_pos_slice[i, 1]]
                            grid_slice[:, iy, ix] = torch.tensor(atf_mags_slice[i])

                        all_slices.append(grid_slice)
                        coord_vec = np.concatenate([source_pos, [z_val]])
                        all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
                        all_sample_info.append(torch.tensor([src_id, z_val], dtype=torch.float32))

            self.slices = torch.stack(all_slices)
            self.coords = torch.stack(all_coords)
            self.sample_info = torch.stack(all_sample_info)
            torch.save({'slices': self.slices,
                        'coords': self.coords,
                       'sample_info': self.sample_info
                        }, processed_file)
            print(f"Saved processed ATF {self.mode} data to {processed_file}")

        self.dummy = torch.nn.Buffer(torch.zeros(1))
        # Optionally crop frequency channels after loading/processing
        if self.freq_up_to is not None:
            if self.freq_up_to < self.slices.shape[1]:
                self.slices = self.slices[:, :self.freq_up_to, :, :]

        print(f"Loaded {len(self.slices)} ATF slices for {self.mode} set.")
        print(f"Slice tensor shape: {self.slices.shape}")
        print(f"Coordinate tensor shape: {self.coords.shape}")


    def __len__(self):
        return len(self.slices)


    def plot(self, ind: int = 5, sample_idx: int = None):
        """
        Plots a 2D spatial slice of ATF magnitudes for a given sample and frequency.
        'ind' corresponds to the frequency index.
        """
        if sample_idx is None:
            sample_idx = random.randint(0, len(self) - 1)

        # The user used 'ind', which we interpret as frequency index
        freq_idx = ind

        slice_to_plot = self.slices[sample_idx, freq_idx].cpu().numpy()

        # plt.figure(figsize=(8, 6))
        # im = plt.imshow(slice_to_plot, origin='lower', cmap='viridis', aspect='auto')
        # plt.colorbar(im, label="Magnitude")
        # plt.xlabel("X-index")
        # plt.ylabel("Y-index")
        # plt.title(f"ATF Slice - Sample {sample_idx}, Freq Index {freq_idx}")
        # plt.show()


    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: The desired number of samples.
        Returns:
            - samples: Tensor of shape (batch_size, 64, H, W).
            - labels: Tensor of shape (batch_size, 4) for coordinates.
        """
        if num_samples > len(self.slices):
            # Sample with replacement if requesting more samples than available
            indices = torch.randint(0, len(self.slices), (num_samples,))
        else:
            # Sample without replacement for a random, unique batch
            indices = torch.randperm(len(self.slices))[:num_samples]

        samples = self.slices[indices]
        labels = self.coords[indices]

        if self.transform:
            samples = self.transform(samples)

        # The data is already (C, H, W), so we just move it to the correct device
        return samples.to(self.dummy.device), labels.to(self.dummy.device)

    def get_slice_by_id(self, src_id: int, z_height: float):
        """Finds and returns a specific slice by source ID and z-height."""
        if self.sample_info is None:
            raise RuntimeError("Sampler was not initialized with sample_info. Please re-process the data.")

        # Find all entries matching the source ID
        src_matches = self.sample_info[:, 0] == src_id
        # Find all entries matching the z-height (with a small tolerance for float comparison)
        z_matches = torch.isclose(self.sample_info[:, 1], torch.tensor(z_height))

        # Find the index where both conditions are true
        combined_matches = src_matches & z_matches
        indices = torch.where(combined_matches)[0]

        if len(indices) == 0:
            print(f"Warning: No slice found for Source ID {src_id} and Z-Height {z_height}.")
            return None, None

        # Get the first matching index
        item_idx = indices[0].item()

        # Retrieve the data
        sample = self.slices[item_idx]
        label = self.coords[item_idx]

        if self.transform:
            sample = self.transform(sample.unsqueeze(0)).squeeze(0)

        # Return with a batch dimension of 1
        return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device)

class FreqConditionalATFSampler(torch.nn.Module, Sampleable):
    """
    Serves 2D spatial slices of ATF magnitudes, treating each frequency bin
    as a separate sample and adding the frequency index to the conditioning vector.
    """
    def __init__(self, data_path: str, mode: str, src_splits: dict,
                 transform: Optional[callable] = None, freq_up_to: int = 64):
        super().__init__()

        # **NEW: Store the actual frequency values (in Hz)**
        # We assume a fixed fftlen_algn of 128 and fs of 2000 from your generation script
        fftlen_algn = 128
        fs = 2000
        # This creates the same frequency table you have in your screenshot
        self.freq_algn = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
        self.nyquist_freq = fs / 2  # The maximum possible frequency

        self.transform = transform
        self.mode = mode
        self.src_splits = src_splits
        self.num_freqs = freq_up_to

        processed_file = os.path.join(data_path, f'processed_atf_{self.mode}.pt')

        if os.path.exists(processed_file):
            print(f"Loading pre-processed ATF {self.mode} data from {processed_file}")
            data = torch.load(processed_file)
            self.slices = data['slices']
            self.coords = data['coords']
            self.sample_info = data.get('sample_info')
            # self.freq_algn = data['freq_algn']
            # self.nyquist_freq = self.freq_algn[-1]

        else:
            print(f"Processing ATF {self.mode} data from .npz files...")
            source_indices = range(*src_splits[self.mode])
            all_slices = []
            all_coords = []
            all_sample_info = []

            for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
                npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
                with np.load(npz_file) as data:
                    atf_mags = data['atf_mag_algn']   # Shape: (1331, 64)
                    mic_pos = data['posMic']          # Shape: (1331, 3)
                    source_pos = data['posSrc']       # Shape: (3,)
                    # self.freq_algn = data['freq_algn']
                    # self.nyquist_freq = self.freq_algn[-1]

                    unique_z = np.unique(mic_pos[:, 2])

                    for z_val in unique_z:
                        slice_indices = np.where(mic_pos[:, 2] == z_val)[0]
                        mic_pos_slice = mic_pos[slice_indices]
                        atf_mags_slice = atf_mags[slice_indices]

                        unique_x = sorted(np.unique(mic_pos_slice[:, 0]))
                        unique_y = sorted(np.unique(mic_pos_slice[:, 1]))
                        nx, ny = len(unique_x), len(unique_y)

                        if nx * ny != len(mic_pos_slice):
                            print(f"Warning: Skipping slice for src_id {src_id} at z={z_val} due to irregular grid.")
                            continue

                        x_map = {val: i for i, val in enumerate(unique_x)}
                        y_map = {val: i for i, val in enumerate(unique_y)}

                        # Pre-allocate for full frequency dimension; we'll crop later if requested
                        grid_slice = torch.zeros((64, ny, nx), dtype=torch.float32)
                        for i in range(len(mic_pos_slice)):
                            ix, iy = x_map[mic_pos_slice[i, 0]], y_map[mic_pos_slice[i, 1]]
                            grid_slice[:, iy, ix] = torch.tensor(atf_mags_slice[i])

                        all_slices.append(grid_slice)
                        coord_vec = np.concatenate([source_pos, [z_val]])
                        all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
                        all_sample_info.append(torch.tensor([src_id, z_val], dtype=torch.float32))

            self.slices = torch.stack(all_slices)
            self.coords = torch.stack(all_coords)
            self.sample_info = torch.stack(all_sample_info)
            torch.save({'slices': self.slices,
                        'coords': self.coords,
                       'sample_info': self.sample_info
                        }, processed_file)
            print(f"Saved processed ATF {self.mode} data to {processed_file}")

            # --- New Logic for Frequency-Conditional Sampling ---
            # 1. Crop to the desired number of frequencies immediately after loading.

        if freq_up_to > self.slices.shape[1]:
            raise ValueError(
                f"freq_up_to ({freq_up_to}) cannot be larger than the number of available frequency bins ({self.slices.shape[1]}).")

        self.slices = self.slices[:, :freq_up_to, :, :]


        # --- Final Setup ---
        self.dummy = torch.nn.Buffer(torch.zeros(1))

        print(f"\n--- FreqConditionalATFSampler Initialized ({self.mode} mode) ---")
        print(f"  Using {self.num_freqs} frequency bins per slice.")
        print(f"  Number of original spatial slices: {len(self.slices)}")
        print(f"  Total number of samples (slices * freqs): {len(self)}")
        print(f"  Sample shape (before transform): (1, {self.slices.shape[2]}, {self.slices.shape[3]})")
        print(f"  Label shape: ({self.coords.shape[1] + 1},)")
        print("--------------------------------------------------")


    def __len__(self):
        # The total number of samples is num_slices * num_frequencies
        return len(self.slices) * self.num_freqs


    def plot(self, ind: int = 5, sample_idx: int = None):
        """
        Plots a 2D spatial slice of ATF magnitudes for a given sample and frequency.
        'ind' corresponds to the frequency index.
        """
        if sample_idx is None:
            sample_idx = random.randint(0, len(self) - 1)

        # The user used 'ind', which we interpret as frequency index
        freq_idx = ind

        slice_to_plot = self.slices[sample_idx, freq_idx].cpu().numpy()

        # plt.figure(figsize=(8, 6))
        # im = plt.imshow(slice_to_plot, origin='lower', cmap='viridis', aspect='auto')
        # plt.colorbar(im, label="Magnitude")
        # plt.xlabel("X-index")
        # plt.ylabel("Y-index")
        # plt.title(f"ATF Slice - Sample {sample_idx}, Freq Index {freq_idx}")
        # plt.show()

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Generate random indices for the flattened dataset
        indices = torch.randint(0, len(self), (num_samples,))

        # Convert flat indices back to slice and frequency indices
        slice_indices = indices // self.num_freqs
        freq_indices = indices % self.num_freqs

        # Get the single-frequency spatial slices
        # Note: self.slices is (N, C, H, W). We need to gather along N and C.
        samples = self.slices[slice_indices, freq_indices, :, :]

        # Get the corresponding 4D coordinate labels
        coord_labels = self.coords[slice_indices]
        freq_hz_vals = torch.tensor(self.freq_algn[freq_indices.cpu()], dtype=torch.float32)

        # 2. Normalize the Hz values to the range [0, 1]
        normalized_freqs = freq_hz_vals / self.nyquist_freq

        # Create the new 5D conditioning vector: [coords, freq_idx]
        # freq_labels = freq_indices.float().unsqueeze(1).to(coord_labels.device) # old
        # freq_labels = normalized_freqs.unsqueeze(1).to(coord_labels.device)
        freq_labels = normalized_freqs.view(-1, 1).to(coord_labels.device)
        labels = torch.cat([coord_labels, freq_labels], dim=1)

        if self.transform:
            samples = self.transform(samples)

        # Return a single channel (the magnitude) and the new 5D label
        # The channel dim is added here to make it (batch, 1, H, W)
        return samples.unsqueeze(1).to(self.dummy.device), labels.to(self.dummy.device)

    def get_slice_by_id(self, src_id: int, z_height: float, freq_idx: int):
        """Finds and returns a specific slice by source ID and z-height."""
        if self.sample_info is None:
            raise RuntimeError("Sampler was not initialized with sample_info. Please re-process the data.")
        if not (0 <= freq_idx < self.num_freqs):
            raise IndexError(f"freq_idx {freq_idx} is out of bounds for the number of frequencies ({self.num_freqs}).")

        # Find all entries matching the source ID
        src_matches = self.sample_info[:, 0] == src_id
        # Find all entries matching the z-height (with a small tolerance for float comparison)
        z_matches = torch.isclose(self.sample_info[:, 1], torch.tensor(z_height))

        # Find the index where both conditions are true
        combined_matches = src_matches & z_matches
        indices = torch.where(combined_matches)[0]

        if len(indices) == 0:
            print(f"Warning: No slice found for Source ID {src_id} and Z-Height {z_height}.")
            return None, None

        # Get the first matching index
        item_idx = indices[0].item()

        # 2. Retrieve the multi-channel slice and its 4D coordinate
        full_slice = self.slices[item_idx]  # Shape: (num_freqs, 11, 11)
        base_coord = self.coords[item_idx]  # Shape: (4,)

        # 3. Select the specific frequency plane
        sample = full_slice[freq_idx]  # Shape: (11, 11)

        # 2. Look up the actual Hz value and normalize it
        freq_hz_val = self.freq_algn[freq_idx]
        normalized_freq = freq_hz_val / self.nyquist_freq

        # 4. Construct the final 5D conditioning vector
        freq_label = torch.tensor([normalized_freq], dtype=torch.float32, device=base_coord.device)
        label = torch.cat([base_coord, freq_label])  # Shape: (5,)

        # 5. Apply transform and return with a batch dimension of 1
        if self.transform:
            sample = self.transform(sample.unsqueeze(0)).squeeze(0)

        # Return with a batch dimension of 1
        return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device)

# """Part 2: Training for Classifier Free Guidance (CFG) """
class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


#
class CFGVectorFieldODE(ODE):
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0, y_dim: int = 6, y_embed_dim: int = 40):
        self.net = net
        self.guidance_scale = guidance_scale
        # A learned embedding for the unconditional (null) case
        self.y_null = nn.Parameter(torch.randn(y_embed_dim))

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs, y_dim)
        """
        # For CFG, we need both the conditional and unconditional outputs
        guided_vector_field = self.net(x, t, y)

        # Create a batch of null embeddings for the unguided field
        bs = x.shape[0]
        unguided_y = self.y_null.repeat(bs, 1)  # was: unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field


# class CFGTrainer(Trainer):
#     def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, y_dim: int,
#                  **kwargs):
#         assert eta > 0 and eta < 1
#         super().__init__(model, **kwargs)
#         self.eta = eta
#         self.path = path
#         self.y_dim = y_dim
#
#         # A learned embedding for the unconditional (null) case
#         self.y_null = nn.Parameter(torch.randn(1, y_dim))
#
#     #
#     def get_train_loss(self, batch_size: int) -> torch.Tensor:
#         # Step 1: Sample z (spectrograms) and y (coordinates) from p_data#
#         z, y = self.path.p_data.sample(batch_size)  # (bs, c, h, w), y:(bs, 6)
#         # Ensure y is on the correct device
#         y = y.to(z.device)
#         self.y_null = self.y_null.to(z.device)
#
#         # Step 2: With probability eta, replace the coordinate vector with the null embedding
#         is_conditional_mask = (torch.rand(y.shape[0], device=y.device) > self.eta)
#         # Reshape for broadcasting: (bs,) -> (bs, 1)
#         is_conditional_mask = is_conditional_mask.view(-1, 1)
#
#         # Create the final conditioning tensor for the batch
#         y_cond = torch.where(is_conditional_mask, y, self.y_null)
#
#         # Step 3: Sample t (time) and x (noisy spectrogram)
#         t = torch.rand(batch_size, 1, 1, 1).to(z.device)
#         x = self.path.sample_conditional_path(z, t)
#
#         # Step 4: Regress the model's output against the ground truth vector field
#         ut_theta = self.model(x, t, y_cond)
#         ut_ref = self.path.conditional_vector_field(x, z, t)
#
#         error = torch.square(ut_theta - ut_ref)
#         # Flatten error fr
#         # om (bs, c, h, w) to (bs, -1) and sum over dimensions
#         loss_per_sample = error.view(batch_size, -1).sum(dim=1)
#
#         # Apply the mask to compute the loss only on conditional samples
#         masked_loss = loss_per_sample * is_conditional_mask.squeeze()
#
#         # Average the loss over the number of conditional samples
#         # Add a small epsilon to avoid division by zero if no samples are conditional
#         num_conditional_samples = is_conditional_mask.sum()
#         mean_loss = masked_loss.sum() / (num_conditional_samples + 1e-8)
#
#         # error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref))  # (bs,)
#         return mean_loss
#
#     # **NEW: Validation loss implementation**
#     @torch.no_grad()
#     def get_valid_loss(self, valid_sampler: Sampleable, batch_size: int, **kwargs) -> torch.Tensor:
#         # Step 1: Sample z and y from the validation data sampler
#         z, y = valid_sampler.sample(batch_size)
#         y = y.to(z.device)
#
#         # Step 2: For validation, we ONLY use conditional samples. No CFG masking.
#         y_cond = y
#
#         # Step 3: Sample t and x
#         t = torch.rand(batch_size, 1, 1, 1).to(z.device)
#         x = self.path.sample_conditional_path(z, t)
#
#         # Step 4: Calculate loss
#         ut_theta = self.model(x, t, y_cond)
#         ut_ref = self.path.conditional_vector_field(x, z, t)
#         error = torch.square(ut_theta - ut_ref)
#         loss_per_sample = error.view(batch_size, -1).sum(dim=1)
#
#         return loss_per_sample.mean()

class ATFInpaintingTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField,
                 eta: float, M: int, y_dim: int, sigma: float, flag_gaussian_mask: bool, model_mode: str,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.eta = eta
        self.y_null = torch.nn.Parameter(torch.randn(1, y_dim))
        self.m = M
        self.sigma = sigma
        self.FLAG_GAUSSIAN_MASK = flag_gaussian_mask
        self.model_mode = model_mode

        # Flag to print shapes only on the first run
        self.shapes_printed = False

    def _create_sparse_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Helper function to create a sparse mask for a batch of slices."""
        batch_size, _, H, W = z.shape
        mask = torch.zeros(batch_size, 1, H, W, device=z.device)

        # --- EFFICIENT BATCHED MASKING ---
        # Generate random indices for each sample in the batch without a loop
        num_pixels = (H - 1) * (W - 1)
        # We use multinomial to sample M indices for each of the batch_size samples
        indices = torch.multinomial(torch.ones(batch_size, num_pixels), self.m, replacement=False).to(z.device)

        rows = indices // (W - 1)
        cols = indices % (W - 1)

        # Use advanced indexing to set the mask values for the entire batch at once
        batch_indices = torch.arange(batch_size, device=z.device).view(-1, 1)
        mask[batch_indices, 0, rows, cols] = 1

        return mask

    def get_train_loss(self, **kwargs) -> torch.Tensor:

        # 1. Sample a batch of COMPLETE, clean ATF slices 'z'
        batch_size = kwargs.get('batch_size')
        z, y = self.path.p_data.sample(batch_size)

        # 2. Create the sparse mask efficiently
        mask = self._create_sparse_mask(z)
        z_masked = z * mask

        # 3. --- CORRECT INPAINTING PATH ---
        # The path is a straight line from the masked image to the full image, with optional noise.
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        noise = torch.randn_like(z) * self.sigma

        if self.FLAG_GAUSSIAN_MASK:
            z0 = z_masked + (1 - mask) * noise
        # Create the noisy sample on the path between masked and full data
            x_t = (1 - t) * z0 + t * z
            ut_ref = z - z0
        else:
            x_t = (1 - t) * z_masked + t * z + noise #
            # The target vector field is the difference vector
            ut_ref = z - z_masked  # The target velocity is the difference vector

        # x_t = self.path.sample_conditional_path(z_masked, t) # original version


        # --- Concatenate mask as 65th channel for the MODEL INPUT ---
        model_input = torch.cat([x_t, mask], dim=1)  # Shape becomes (bs, 65, 12, 12)

        # --- LABEL MASKING for CFG ---
        is_conditional_mask = (torch.rand(y.shape[0], device=y.device) > self.eta).view(-1, 1)

        y_null_on_device = self.y_null.to(y.device)
        y_cond = torch.where(is_conditional_mask, y, y_null_on_device)

        # --- Loss Calculation ---
        ut_theta = self.model(model_input, t, y_cond)

        if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        elif self.model_mode == 'freq_cond':
            ut_theta_crop = ut_theta[:, :, :-1, :-1]

        ut_ref_crop = ut_ref[:, :, :-1, :-1]

        region_crop = (1.0 - mask)[:, :, :-1, :-1]
        squared_err = torch.square(ut_theta_crop - ut_ref_crop)*region_crop
        error = squared_err.sum() / region_crop.sum()
        # error = torch.mean()

        if not self.shapes_printed:
            print("\\n--- Tensor Shapes (First Training Step) ---")
            print(f"  Input Slice (z):          {z.shape}")
            print(f" Model Input (x_t + mask): {model_input.shape}\\n")
            print(f"  Masked Slice (z_masked):    {z_masked.shape}")
            print(f"  Noisy Sample (x_t):         {x_t.shape}")
            print(f"  Ground Truth Coords (y):    {y.shape}")
            print(f"  Null Embedding (y_null):    {self.y_null.shape}")
            print(f"  Final Condition (y_cond):   {y_cond.shape}")
            print(f"  Model Output (ut_theta):    {ut_theta.shape}")
            print(f"  No. of observations (M): {self.m}")
            print(" cropped loss' shape: ut_theta[:, :, :-1, :-1] ", ut_theta_crop.shape)
            print("------------------------------------------\\n")
            self.shapes_printed = True

        return error

    @torch.no_grad()
    def get_valid_loss(self, valid_sampler: Sampleable, **kwargs) -> torch.Tensor:
        # Validation loss should also simulate the inpainting task
        batch_size = kwargs.get('batch_size')
        z, y = valid_sampler.sample(batch_size)

        # Use the same efficient masking and path logic for validation
        mask = self._create_sparse_mask(z)
        # print(f"Validation mask shape: {mask.shape}, LHS: {mask.sum(dim=(-3, -2, -1)).unique().item()}, "
        #       f"RHS: {z.shape[-2] * z.shape[-1] - 1 - self.m}, z.shape: {z.shape}")
        # assert (1. - mask).sum(dim=(-3, -2, -1)).unique().item() == z.shape[-2] * z.shape[-1] - 1 - self.m

        z_masked = z * mask

        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        noise = torch.randn_like(z) * self.sigma

        if self.FLAG_GAUSSIAN_MASK:
            z0 = z_masked + (1 - mask) * noise
            x_t = (1 - t) * z0 + t * z
            ut_ref = z - z0
        else:
            x_t = (1 - t) * z_masked + t * z + noise  #
            # The target vector field is the difference vector
            ut_ref = z - z_masked  # The target velocity is the difference vector

        model_input = torch.cat([x_t, mask], dim=1)

        ut_theta = self.model(model_input, t, y)  # Use the true label for validation
        # error = torch.mean(torch.square(ut_theta[:, :-1, :-1, :-1] - ut_ref[:, :, :-1, :-1]))

        if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        elif self.model_mode == 'freq_cond':
            ut_theta_crop = ut_theta[:, :, :-1, :-1]

        # Crop output and reference to 11x11 before comparing
        ut_ref_crop = ut_ref[:, :, :-1, :-1]

        region_crop = (1.0 - mask)[:, :, :-1, :-1]
        squared_err = torch.square(ut_theta_crop - ut_ref_crop) * region_crop
        error = squared_err.sum() / region_crop.sum()

        return error

    # def visualize_masking(self, crop, sample_idx: int = 0, freq_idx: int = 5):
    #     """
    #     Samples one slice, applies the inpainting mask, and plots the original
    #     and masked versions side-by-side for a specific frequency index.
    #     """
    #     # 1. Sample a single complete, clean ATF slice 'z' and its condition 'y'
    #     z, y = self.path.p_data.sample(sample_idx)
    #
    #     # Get the padded height and width
    #     _, _, H, W = z.shape
    #
    #     # 2. --- DATA MASKING (Inpainting) ---
    #     # Create a mask that is the same size as the padded 12x12 image
    #     mask = torch.zeros(sample_idx, 1, H, W, device=z.device)
    #
    #     # Get M random pixel locations to keep
    #     indices = torch.randperm((H-1) * (W-1))[:self.m]
    #     # IM changing this to 11x11 since masking the last row and column is not meaningful as we'll discard
    #     rows = indices // (W - 1)
    #     cols = indices % (W - 1)
    #     mask[0, 0, rows, cols] = 1
    #
    #     z_masked = z * mask
    #
    #     # 3. --- Plotting ---
    #     # Detach tensors and move to CPU for numpy/matplotlib
    #     original_slice = z[0, freq_idx].cpu().numpy()
    #     masked_slice = z_masked[0, freq_idx].cpu().numpy()
    #
    #     if crop:
    #         # Crop to the region of interest if needed
    #         original_slice = original_slice[:-1, :-1]
    #         masked_slice = masked_slice[:-1, :-1]
    #
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    #     fig.suptitle(f"Masking Visualization (Frequency Index: {freq_idx})")
    #
    #     # Plot Original
    #     im1 = axes[0].imshow(original_slice, origin='upper', cmap='viridis')
    #     axes[0].set_title(f'Original Slice')
    #     axes[0].set_xlabel("X-index")
    #     axes[0].set_ylabel("Y-index")
    #     fig.colorbar(im1, ax=axes[0], label="Magnitude")
    #
    #     # Plot Masked
    #     im2 = axes[1].imshow(masked_slice, origin='upper', cmap='viridis')
    #     axes[1].set_title(f'Masked Slice ({self.m} points visible)')
    #     axes[1].set_xlabel("X-index")
    #     axes[1].set_ylabel("Y-index")
    #     fig.colorbar(im2, ax=axes[1], label="Magnitude")
    #
    #     # Plot Mask
    #     mask_slice = mask[0, 0].cpu().numpy()
    #     im3 = axes[2].imshow(mask_slice, origin='upper', cmap='gray', vmin=0, vmax=1)
    #     axes[2].set_title(f'Mask (1 = Visible, 0 = Hidden)')
    #     axes[2].set_xlabel("X-index")
    #     axes[2].set_ylabel("Y-index")
    #     fig.colorbar(im3, ax=axes[2], label="Mask binary value")
    #
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    #     plt.show()


# """ Part 3: An Architecture for Spectrograms: Building a U-Net """

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone()  # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x)  # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x)  # (bs, c, h, w)

        # Add back residual
        x = x + res  # (bs, c, h, w)

        return x


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


#
class SpecUNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_dim: int, y_embed_dim: int):
        super().__init__()
        # Initial convolution: (bs, 1, freq, time) -> (bs, c_0, freq, time)
        self.init_conv = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]),
                                       nn.SiLU())

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # **MODIFICATION: Replace nn.Embedding with an MLP for positional encoding**
        # This MLP will take the coordinate vector 'y' and project it to the embedding dimension.
        self.y_embedder = nn.Sequential(
            nn.Linear(y_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, y_embed_dim)
        )

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, freq_bins, time_bins)
        - t: (bs, 1, 1, 1)
        - y: (bs, 6) <- Now a tensor of 6D coordinates , was y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, freq_bins, time_bins)
        """
        # Embed t and y
        t_embed = self.time_embedder(t)

        # **MODIFICATION: The y_embedder is now an MLP**
        y_embed = self.y_embedder(y)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, freq_bins, time_bins)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, t_embed, y_embed)

        # Final convolution
        x = self.final_conv(x)  # (bs, 1, freq_bins, time_bins)

        return x

class ATFUNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_dim: int, y_embed_dim: int,
                 input_channels: int, output_channels: int):
        super().__init__()

        # --- MODIFICATION 1: Change input channels ---
        # The U-Net accepts an image with [freq_channels + 1] channels (freq bins + mask)
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # --- MODIFICATION 2: Adjust y_embedder for 4D conditioning ---
        # The MLP now takes the 4D coordinate vector [xs, ys, zs, zm] as input.
        self.y_embedder = nn.Sequential(
            nn.Linear(y_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, y_embed_dim)
        )

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        # --- MODIFICATION 3: Change output channels ---
        # The final layer outputs [freq_channels + 1] channels (freq bins + optional mask channel)
        self.final_conv = nn.Conv2d(channels[0], output_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, C_in, H, W) where C_in = freq_channels + 1 (freq bins + mask)
        - t: (bs, 1, 1, 1) # same
        - y: (bs, 4) <- New conditioning shape, was (bs, 6), and BEFORE was y: (bs,) for mnist
        Returns:
        - u_t^theta(x|y): (bs, C_out, H, W) where C_out = freq_channels + 1
        """
        # Embed t and y
        t_embed = self.time_embedder(t)

        # **MODIFICATION: The y_embedder is now an MLP**
        y_embed = self.y_embedder(y)

        # Initial convolution
        x = self.init_conv(x)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, t_embed, y_embed)

        # Final convolution
        x = self.final_conv(x)

        return x