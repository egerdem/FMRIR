from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import os
import torch.nn.functional as F
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
from torch.optim.lr_scheduler import CosineAnnealingLR

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

    # def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs): #PREVIOUSLY
    #     return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        # Get the model's output (the "drift"), which has 20 channels
        drift = self.ode.drift_coefficient(xt, t, **kwargs)

        # Case 1: Standard models where input and output shapes match.
        if xt.shape == drift.shape:
            # If shapes match, apply the Euler update to all channels
            x_next = xt + drift * h

        # Case 2: Inpainting models where the input `xt` has one extra channel (the mask).
        # for ATF 2D inpainting model.
        # Separate the current state `xt` into its data and mask components
        elif xt.shape[1] == drift.shape[1] + 1:
            xt_data = xt[:, :-1]  # The first 20 channels (frequencies)
            xt_mask = xt[:, -1:]  # The last channel (the mask)

            # Apply the Euler update ONLY to the data channels
            updated_xt_data = xt_data + drift * h

            # Re-combine the updated data with the original, unchanged mask
            x_next = torch.cat([updated_xt_data, xt_mask], dim=1)

        else:# Case 3: The shapes are incompatible.
            raise ValueError(
                f"Incompatible shapes for Euler. `xt` is {xt.shape} "
                f"output `drift` is {drift.shape}."
            )

        # --- NEW: Optional Data Consistency ("Pasting") Step ---
        if kwargs.get('paste_observations', False):
            z_true = kwargs.get('z_true')
            x0 = kwargs.get('x0')
            obs_indices = kwargs.get('obs_indices')

            if z_true is not None and x0 is not None:
                # Create the mask for pasting
                full_mask_3d = torch.zeros_like(z_true)
                # Flatten view to use flat indices
                full_mask_flat = full_mask_3d.view(1, -1)
                full_mask_flat[:, obs_indices] = 1

                paste_mask = full_mask_flat.view(*z_true.shape)
                # Get the correct value for the known data on the straight noise-to-data path
                t_next = t + h
                known_path_slice = (1 - t_next) * x0 + t_next * z_true

            # Replace the values at the M known locations
                x_next = x_next * (1 - paste_mask) + known_path_slice * paste_mask

        return x_next


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
    def __init__(self, models: Dict[str, nn.Module]):
        super().__init__()
        if not isinstance(models, dict) or not models:
            raise ValueError("`models` must be a non-empty dictionary of nn.Modules.")

        self.models = models
        # For convenience, self.model can point to the primary model
        self.model = next(iter(self.models.values()))
        self.optimizer = None

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    # **NEW: Abstract method for validation loss**
    @torch.no_grad()
    @abstractmethod
    def get_valid_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        # Collect parameters from ALL models provided
        all_params = []
        for model in self.models.values():
            all_params.extend(list(model.parameters()))
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        return self.optimizer

    def train(self, num_iterations: int, device: torch.device, lr: float,
              valid_sampler: Optional[Sampleable] = None,
              save_path: str = "model.pt",
              checkpoint_path: str = "checkpoints",
              validation_interval: Optional[int] = None,
              checkpoint_interval: Optional[int] = None,
              start_iteration: int = 0,
              config: dict = None,
              early_stopping_patience: int = 1800, #was 1000
              resume_checkpoint_path: Optional[str] = None,
              resume_checkpoint_state: Optional[dict] = None,
              **kwargs):

        print("--- Model(s) Summary ---")
        for name, model in self.models.items():
            print(f"  - {name}: {model_size_b(model) / MiB:.3f} MiB")
            model.to(device)

        # Start
        opt = self.get_optimizer(lr)

        # --- NEW: Create the Learning Rate Scheduler ---
        # It will anneal the LR from its starting value down to nearly zero
        # over the total number of training iterations.
        scheduler = CosineAnnealingLR(opt, T_max=num_iterations, eta_min=1e-7)

        # NEW: Initialize the EarlyStopping monitor
        early_stopper = EarlyStopping(patience=early_stopping_patience)
        # --- State Tracking ---
        best_val_loss = float("inf")
        best_iteration = start_iteration

        # Unified resume logic: load from an explicit checkpoint path/state if provided
        # checkpoint = None
        # if resume_checkpoint_state is not None:
        #     checkpoint = resume_checkpoint_state
        #     print("Resuming from provided in-memory checkpoint state")
        # elif resume_checkpoint_path is not None and os.path.exists(resume_checkpoint_path):
        #     print(f"Loading checkpoint from {resume_checkpoint_path}")
        #     checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        #
        # if checkpoint is not None:
        #     # Restore model weights if present
        #     model_state = checkpoint.get('model_state_dict')
        #     if model_state is not None:
        #         self.model.load_state_dict(model_state)
        #
        #     # Restore trainer-specific state if present (e.g., y_null)
        #     if hasattr(self, 'y_null') and checkpoint.get('y_null') is not None:
        #         self.y_null.data = checkpoint['y_null'].to(device)
        #
        #     # Restore optimizer if present
        #     if checkpoint.get('optimizer_state_dict') is not None:
        #         opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #         print("Optimizer state restored from checkpoint.")
        #
        #         print(f"Overwriting optimizer LR with new command-line value: {lr}")
        #         for param_group in opt.param_groups:
        #             param_group['lr'] = lr
        #
        #     # Adopt iteration and best metrics if available
        #     iter_value = checkpoint.get('iteration', None)
        #     if isinstance(iter_value, (int, float)):
        #         start_iteration = int(iter_value)
        #         print("starting from iteration", start_iteration)
        #     else:
        #         start_iteration = checkpoint["config"]["training"].get("num_iterations") + 1
        #         print("starting from iteration", start_iteration)
        #     if start_iteration is None or not isinstance(start_iteration, int) or start_iteration <= 0:
        #         assert start_iteration >= 0, "start_iteration must be a non-negative integer"
        #
        #     best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        #     best_iteration = checkpoint.get('best_iteration', best_iteration)
        #     print(f"Resumed state. start_iteration={start_iteration}, best_val_loss={best_val_loss}, best_iteration={best_iteration}")

        if resume_checkpoint_state:
            print("Resuming from provided in-memory checkpoint state")

            # New, robust way: load multiple model states
            if 'model_states' in resume_checkpoint_state:
                for key, state_dict in resume_checkpoint_state['model_states'].items():
                    if key in self.models:
                        self.models[key].load_state_dict(state_dict)
                        print(f"  - Loaded state for model: '{key}'")
            # Fallback for old, single-model checkpoints
            elif 'model_state_dict' in resume_checkpoint_state:
                print("  - Loading state from legacy 'model_state_dict' key.")
                self.model.load_state_dict(resume_checkpoint_state['model_state_dict'])

            # Restore optimizer and other trainer state
            if 'optimizer_state_dict' in resume_checkpoint_state:
                opt.load_state_dict(resume_checkpoint_state['optimizer_state_dict'])
                print("  - Optimizer state restored.")

            if hasattr(self, 'y_null_token') and 'y_null_token' in resume_checkpoint_state:
                self.set_encoder.y_null_token.data = resume_checkpoint_state['y_null_token'].to(device)
                print("  - y_null_token restored.")

                # --- Robust y_null Loading ---
                # Try to load the new, unified key first
                if 'y_null_token' in resume_checkpoint_state and resume_checkpoint_state['y_null_token'] is not None:
                    y_null_val = resume_checkpoint_state['y_null_token'].to(device)
                    # Check if the current trainer is a 3D one
                    if hasattr(self, 'set_encoder') and hasattr(self.set_encoder, 'y_null_token'):
                        self.set_encoder.y_null_token.data = y_null_val
                        print("  - y_null_token restored.")
                    # Check if the current trainer is a 2D one
                    elif hasattr(self, 'y_null'):
                        self.y_null.data = y_null_val
                        print("  - y_null restored (from 'y_null_token' key).")

                # Fallback for old checkpoints with the legacy 'y_null' key
                elif 'y_null' in resume_checkpoint_state and resume_checkpoint_state['y_null'] is not None:
                    if hasattr(self, 'y_null'):
                        self.y_null.data = resume_checkpoint_state['y_null'].to(device)
                        print("  - y_null restored from legacy 'y_null' key.")

            start_iteration = resume_checkpoint_state.get('iteration', start_iteration)
            best_val_loss = resume_checkpoint_state.get('best_val_loss', best_val_loss)
            best_iteration = resume_checkpoint_state.get('best_iteration', best_iteration)
            print(f"Resumed state. start_iteration={start_iteration}, best_val_loss={best_val_loss:.5f} at iteration {best_iteration}")
            scheduler.last_epoch = start_iteration - 1

        # --- TRAINING LOOP ---
        batch_size = kwargs.get('batch_size')
        # dataset_size = len(self.path.p_data.spectrograms)
        dataset_size = len(self.path.p_data)

        pbar = tqdm(range(start_iteration, num_iterations))
        for iteration in pbar:
            for model in self.models.values():
                model.train()
            # self.model.train()
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            scheduler.step()

            # Calculate and display the current epoch number**
            current_lr = scheduler.get_last_lr()[0]
            current_epoch = (iteration + 1) * batch_size / dataset_size
            wandb.log({"train_loss": loss.item(), "epoch": current_epoch, "iteration": iteration, "learning_rate": current_lr})

            # **NEW: Validation loop**
            if valid_sampler and (iteration + 1) % validation_interval == 0:
                # self.model.eval()
                for model in self.models.values():
                    model.eval()
                val_loss = self.get_valid_loss(valid_sampler=valid_sampler, **kwargs)
                # **NEW: Log validation loss to wandb**
                wandb.log({"val_loss": val_loss.item(), "epoch": current_epoch, "iteration": iteration})
                pbar.set_description(
                    f'Epoch: {current_epoch:.4f}, Iter: {iteration}, Loss: {loss.item():.5f}, Val Loss: {val_loss.item():.5f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(
                        f"** [Iter {iteration}] New best val. loss found for: train loss: {loss.item():.5f} and val loss: {best_val_loss:.5f}. Saving model. **")

                    # Save best model state for inference
                    # Handle different y_null types
                    y_null_to_save = None
                    if hasattr(self, 'set_encoder') and self.set_encoder is not None:
                        y_null_to_save = self.set_encoder.y_null_token
                    elif hasattr(self, 'y_null'):
                        y_null_to_save = self.y_null

                    # 2. Create the state dictionary
                    model_states_to_save = {key: model.state_dict() for key, model in self.models.items()}

                    best_model_state = {
                        # 'model_state_dict': self.model.state_dict(),
                        'model_states': {key: model.state_dict() for key, model in self.models.items()},
                        'optimizer_state_dict': opt.state_dict(),
                        'iteration': iteration + 1,  # Store next iteration for resuming
                        'best_val_loss': best_val_loss,
                        'best_iteration': iteration,
                        'config': config,
                        'wandb_run_id': config.get('wandb_run_id'),
                        'y_null_token': y_null_to_save,
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

                y_null_to_save = getattr(self.models.get('set_encoder'), 'y_null_token', getattr(self, 'y_null', None))

                # Save checkpoint for resuming training (latest state)
                checkpoint_state = {
                    'iteration': iteration + 1,
                    'model_states': {key: model.state_dict() for key, model in self.models.items()},
                    # 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_iteration': best_iteration,
                    'config': config,
                    'wandb_run_id': config.get('wandb_run_id'),
                    'is_best': False,  # Flag to indicate this is latest checkpoint
                    'y_null_token': y_null_to_save,
                }
                torch.save(checkpoint_state, ckpt_save_path)

        # --- Save final checkpoint ---
        final_iteration = iteration + 1
        if final_iteration == num_iterations or flag_save:

            final_ckpt_path = os.path.join(checkpoint_path, f"ckpt_final_{final_iteration}.pt")
            print(f"\n--- Saving final checkpoint at iteration {final_iteration} to {final_ckpt_path} ---")

            y_null_to_save = getattr(self.models.get('set_encoder'), 'y_null_token', getattr(self, 'y_null', None))

            final_checkpoint_state = {
                'iteration': final_iteration,
                # 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val_loss,
                'best_iteration': best_iteration,
                'config': config,
                'wandb_run_id': config.get('wandb_run_id'),
                'is_best': False,
                'is_final': True,  # Flag to indicate this is the final state
                'model_states': {key: model.state_dict() for key, model in self.models.items()},
                'y_null_token': y_null_to_save
            }
            torch.save(final_checkpoint_state, final_ckpt_path)

        self.model.eval()
        print(f"--- Training finished. Best validation loss was {best_val_loss:.5f} at iteration {best_iteration}. ---")
        return best_val_loss

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
        Args:save_path
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
    def __init__(self, data_path: str, mode: str, src_splits: dict, freq_up_to: int,
                 transform: Optional[callable] = None, ):
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


class ATF3DSampler(torch.nn.Module, Sampleable):
    """
        Loads and serves full 3D ATF magnitude cubes.
        Each sample is a tensor of shape [64, 11, 11, 11] (freq, Z, Y, X).
        """
    def __init__(self, data_path: str, mode: str, src_splits: dict, freq_up_to: int, normalize: bool = True):
        super().__init__()
        self.mode = mode
        self.src_splits = src_splits
        self.normalize = normalize
        self.mean = None
        self.std = None
        self.freq_up_to = freq_up_to
        # Use a distinct cache file to avoid clobbering 2D slice caches
        # processed_file = os.path.join(data_path, f'processed_atf3d_{self.mode}.pt')
        processed_file = os.path.join(data_path, f'processed_atf3d_{self.mode}_freqs{self.freq_up_to}.pt')

        if os.path.exists(processed_file):
            print(f"Loading pre-processed ATF-3D {self.mode} data from {processed_file}")
            data = torch.load(processed_file)
            self.cubes = data['cubes']
            self.source_coords = data['source_coords']
            self.grid_xyz = data['grid_xyz']
            # self.sample_info = data.get('sample_info')
            if self.normalize:
                self.mean = data.get('mean')
                self.std = data.get('std')

        else:
            print(f"Processing ATF-3D {self.mode} data from .npz files...")
            source_indices = range(*src_splits[self.mode])
            all_cubes = []
            all_source_coords = []
            all_sample_info = []

            # --- Grid construction (assuming it's constant for all sources) ---
            # --- Colleague's Fix 1: Establish a canonical grid order and permutation ---
            first_npz_file = os.path.join(data_path, f"data_s{source_indices[0] + 1:04d}.npz")
            with np.load(first_npz_file) as data:
                mic_pos = data['posMic']  # shape (1331, 3)
                x, y, z = mic_pos[:, 0], mic_pos[:, 1], mic_pos[:, 2]

                # Sort by z, then y, then x to get a canonical C-style row-major order
                perm = np.lexsort((x, y, z))

                unique_x, unique_y, unique_z = sorted(np.unique(x)), sorted(np.unique(y)), sorted(np.unique(z))
                self.nx, self.ny, self.nz = len(unique_x), len(unique_y), len(unique_z)

            # Create the canonical grid that matches the flattened order
            zz, yy, xx = torch.meshgrid(
                torch.tensor(unique_z, dtype=torch.float32),
                torch.tensor(unique_y, dtype=torch.float32),
                torch.tensor(unique_x, dtype=torch.float32),
                indexing='ij'
            )
            self.grid_xyz = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

            for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
                npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
                if not os.path.exists(npz_file): continue

                with np.load(npz_file) as data_single:

                    atf_mag_algn = data_single['atf_mag_algn']  # (1331, 64)
                    np_of_mics, np_of_freqs = atf_mag_algn.shape
                    source_pos = data_single['posSrc']  # (3,)

                    # Reorder rows into the canonical layout using the permutation
                    atf_perm = torch.tensor(atf_mag_algn[perm], dtype=torch.float32)  # [1331, 64]

                    # chatgpt version: cube = atf_perm.T.view(64, nz, ny, nx)
                    # Reshape the ordered data into the 3D cube
                    full_cube = atf_perm.T.contiguous().view(np_of_freqs, self.nz, self.ny, self.nx)  # [64, 11, 11, 11]
                    cube = full_cube[:self.freq_up_to, :, :, :]

                    all_cubes.append(cube)
                    all_source_coords.append(torch.tensor(source_pos, dtype=torch.float32))
                    all_sample_info.append(torch.tensor([src_id], dtype=torch.int32))

            self.cubes = torch.stack(all_cubes)
            self.source_coords = torch.stack(all_source_coords)
            self.sample_info = torch.stack(all_sample_info)

            if self.normalize and self.mode == 'train':
                self.mean = self.cubes.mean()
                self.std = self.cubes.std()
                self.cubes = (self.cubes - self.mean) / (self.std + 1e-8)


            # --- Colleague's Fix 2: Save the grid and its dimensions with the cache ---
            save_data = {
                'cubes': self.cubes,
                'source_coords': self.source_coords,
                'sample_info': self.sample_info,
                'grid_xyz': self.grid_xyz,
                'nxnyz': (self.nx, self.ny, self.nz),
            }

            if self.normalize:
                save_data.update({'mean': self.mean, 'std': self.std})
            torch.save(save_data, processed_file)
            print(f"Saved processed ATF-3D {self.mode} data to {processed_file}")

        self.dummy = torch.nn.Buffer(torch.zeros(1))
        print(f"Loaded {len(self.cubes)} ATF-3D cubes for {self.mode} set.")
        print(f"Cube tensor shape: {self.cubes.shape}")

    def __len__(self):
        return len(self.cubes)

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        indices = torch.randperm(len(self.cubes))[:num_samples]
        z_full_batch = self.cubes[indices]
        src_xyz_batch = self.source_coords[indices]
        return z_full_batch.to(self.dummy.device), src_xyz_batch.to(self.dummy.device)


class SetEncoder(nn.Module):
    """Encodes a sparse set of observations into a sequence of tokens. and a pooled context vector."""

    def __init__(self, num_freqs=64, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        # MLP to tokenize each observation: [rel_coord(3), values(64)] -> d_model
        self.tokenizer_mlp = nn.Sequential(
            nn.Linear(3 + num_freqs, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder to mix observation tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.y_null_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, obs_coords_rel, obs_values, obs_mask):
        """
        Args:
            obs_coords_rel (Tensor): Relative mic coordinates [B, M_max, 3]
            obs_values (Tensor): ATF magnitudes at those mics [B, M_max, 64]
            obs_mask (Tensor): Boolean mask indicating valid observations [B, M_max]
        Returns:
            tokens (Tensor): Encoded observation tokens [B, M_max, d_model]
            pooled_context (Tensor): A single context vector per batch item [B, d_model]

        """
        # 1. Concatenate coordinates and values for each observation
        token_features = torch.cat([obs_coords_rel, obs_values], dim=-1)  # [B, M_max, 67]

        # 2. Project each observation to a token of dimension d_model
        tokens = self.tokenizer_mlp(token_features)  # [B, M_max, d_model]

        # 3. Use transformer to let observations communicate with each other
        # The transformer expects a padding mask where True means "ignore"
        padding_mask = ~obs_mask
        tokens = self.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        # --- NEW: Create the pooled context vector ---
        # To correctly average, we mask the padded tokens before summing.
        masked_tokens = tokens.masked_fill(~obs_mask.unsqueeze(-1), 0.0)
        # Sum valid tokens and divide by the number of valid tokens. Add epsilon for stability.
        num_valid_tokens = obs_mask.sum(dim=1, keepdim=True)
        pooled_context = masked_tokens.sum(dim=1) / (num_valid_tokens + 1e-8)

        return tokens, pooled_context

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
    # Used in 2d UNET ATFSliceGenerator, and original MNIST
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

        combined_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

        # --- ADD THIS CHECK TO HANDLE OLD MODELS ---
        # The data part of the input state `x` has x.shape[1] - 1 channels.
        # If the model's output has more channels than that, it's an old model.
        num_data_channels = x.shape[1] - 1
        if combined_field.shape[1] > num_data_channels:
            # Slice off the extra, meaningless channel(s) to match the data.
            return combined_field[:, :num_data_channels]

        return combined_field


class CFGVectorFieldODE_3D(ODE):
    """
    An ODE wrapper for the 3D U-Net and SetEncoder for the ATF_3D.
    """

    def __init__(self, unet, set_encoder, guidance_scale=1.0):
        self.unet = unet
        self.set_encoder = set_encoder
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # The simulator will pass 'y_tokens' and 'obs_mask' through kwargs
        y_tokens = kwargs['y_tokens']
        obs_mask = kwargs['obs_mask']

        # 1. Get the guided prediction
        guided_vector_field = self.unet(xt, t.squeeze(), context=y_tokens, context_mask=obs_mask)

        # 2. Get the unguided prediction
        null_tokens = self.set_encoder.y_null_token.expand(xt.shape[0], y_tokens.shape[1], -1)
        unguided_vector_field = self.unet(xt, t.squeeze(), context=null_tokens, context_mask=obs_mask)

        # 3. Combine using the CFG formula and return a single DRIFT tensor
        combined_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

        return combined_field

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
        super().__init__(models={'unet': model}, **kwargs)
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
            # x_t = (1 - t) * z_masked + t * z + noise #
            x_t = (1 - (1-self.sigma)*t) * z_masked + t * z #
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

        # if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            # ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        # elif self.model_mode == 'freq_cond':
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
            # x_t = (1 - t) * z_masked + t * z + noise  #
            x_t = (1 - (1-self.sigma)*t) * z_masked + t * z + noise  #
            # The target vector field is the difference vector
            ut_ref = z - z_masked  # The target velocity is the difference vector

        model_input = torch.cat([x_t, mask], dim=1)

        ut_theta = self.model(model_input, t, y)  # Use the true label for validation
        # error = torch.mean(torch.square(ut_theta[:, :-1, :-1, :-1] - ut_ref[:, :, :-1, :-1]))

        # if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            # ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        # elif self.model_mode == 'freq_cond':
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


class ATF3DTrainer(Trainer):
    def __init__(self, path, model, set_encoder, eta, M_range, sigma, grid_xyz, **kwargs):
        super().__init__(models={'unet': model, 'set_encoder': set_encoder})
        self.path = path
        self.set_encoder = set_encoder
        self.eta = eta
        self.M_range = (int(M_range[0]), int(M_range[1]))
        self.sigma = sigma
        self.grid_xyz = grid_xyz.to(next(model.parameters()).device)  # (1331, 3)

        # A learnable embedding for the unconditional (null) case
        d_model = set_encoder.d_model

    def make_observation_set(self, z_full, src_xyz):
        B, C, D, H, W = z_full.shape
        dev = z_full.device

        grid_xyz = self.grid_xyz.to(dev)  # ensure same device
        src_xyz = src_xyz.to(dev)

        N = self.grid_xyz.shape[0]  # Total number of mics (1331)
        M_max = self.M_range[1]
        obs_coords_rel_list, obs_values_list, obs_mask_list = [], [], []

        # Loop over each sample in the batch to handle variable M
        for i in range(B):
            # 1. Randomly pick M for this sample
            M = torch.randint(self.M_range[0], self.M_range[1] + 1, (1,)).item()

            # 2. Randomly choose M mic indices
            obs_indices = torch.randperm(N, device=dev)[:M]

            # 3. Gather coordinates and values
            obs_xyz = self.grid_xyz[obs_indices]  # [M, 3]
            obs_coords_rel = obs_xyz - src_xyz[i].unsqueeze(0)  # [M, 3]

            # Flatten spatial dimensions of the cube to easily gather values
            z_flat = z_full[i].view(C, -1)  # [64, 1331]
            obs_values = z_flat[:, obs_indices].transpose(0, 1)  # [M, 64]

            # 4. Pad tensors to max length for batching, gemini commented:
            pad_len = M_max - M
            obs_coords_rel_padded = nn.functional.pad(obs_coords_rel, (0, 0, 0, pad_len))
            obs_values_padded = nn.functional.pad(obs_values, (0, 0, 0, pad_len))
            # target sizes -- chatgpt version:
            # M_max = self.M_range[1]
            #
            # # coords: [M, 3] -> [M_max, 3]
            # obs_coords_rel_padded = torch.zeros(M_max, 3, device=z_full.device, dtype=obs_coords_rel.dtype)
            # obs_coords_rel_padded[:M] = obs_coords_rel
            #
            # # values: [M, 64] -> [M_max, 64]
            # obs_values_padded = torch.zeros(M_max, obs_values.shape[1], device=z_full.device, dtype=obs_values.dtype)
            # obs_values_padded[:M] = obs_values

            # Create a mask: True for valid observations, False for padding
            mask = torch.zeros(M_max, dtype=torch.bool, device=dev)
            mask[:M] = True

            obs_coords_rel_list.append(obs_coords_rel_padded)
            obs_values_list.append(obs_values_padded)
            obs_mask_list.append(mask)

        return torch.stack(obs_coords_rel_list), torch.stack(obs_values_list), torch.stack(obs_mask_list)

    def get_train_loss(self, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get('batch_size')
        # 1. Sample a batch of complete, clean 3D ATF cubes and their source coordinates
        z_full, src_xyz = self.path.p_data.sample(batch_size)

        dev = next(self.model.parameters()).device
        z_full = z_full.to(dev)
        src_xyz = src_xyz.to(dev)

        x1 = z_full

        # 2. Create the sparse observation set on the fly
        obs_coords_rel, obs_values, obs_mask = self.make_observation_set(z_full, src_xyz)

        # 3. Encode the observations into conditioning tokens
        y_tokens, _ = self.set_encoder(obs_coords_rel, obs_values, obs_mask)  # [B, M_max, d_model]

        # 4. Define the Flow Matching path from noise to data
        t = torch.rand(batch_size, device=x1.device).view(-1, 1, 1, 1, 1)
        x0 = torch.randn_like(x1)

        # xt = (1 - t) * x0 + t * x1
        xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
        # ut_ref = x1 - x0
        ut_ref = x1 - (1 - self.sigma) * x0

        # 5. Apply Classifier-Free Guidance during training
        # With probability eta, replace conditioning tokens with the null token
        is_conditional_mask = (torch.rand(batch_size, device=x1.device) > self.eta)

        # Broadcast y_null_token and select based on the mask
        null_tokens = self.set_encoder.y_null_token.expand(batch_size, y_tokens.shape[1], -1)
        final_tokens = torch.where(is_conditional_mask.view(-1, 1, 1), y_tokens, null_tokens)

        # The mask for the transformer (to ignore padding) is the same for both cases
        final_obs_mask = obs_mask

        # 6. Get the model's prediction for the velocity field
        # The 3D U-Net's forward pass must accept `context` and `context_mask`
        ut_theta = self.model(xt, t, context=final_tokens, context_mask=final_obs_mask)

        # 7. Compute the loss
        loss = torch.mean(torch.square(ut_theta - ut_ref))

        return loss

    @torch.no_grad()
    def get_valid_loss(self, valid_sampler: Sampleable, **kwargs) -> torch.Tensor:
        batch_size = kwargs.get('batch_size')
        z_full, src_xyz = valid_sampler.sample(batch_size)

        dev = next(self.model.parameters()).device
        z_full = z_full.to(dev)
        src_xyz = src_xyz.to(dev)

        x1 = z_full

        obs_coords_rel, obs_values, obs_mask = self.make_observation_set(z_full, src_xyz)
        y_tokens, _ = self.set_encoder(obs_coords_rel, obs_values, obs_mask)

        t = torch.rand(batch_size, device=x1.device).view(-1, 1, 1, 1, 1)
        x0 = torch.randn_like(x1)
        # xt = (1 - t) * x0 + t * x1
        xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
        # ut_ref = x1 - x0
        ut_ref = x1 - (1 - self.sigma) * x0

        # For validation, we are always conditional
        ut_theta = self.model(xt, t.squeeze(), context=y_tokens, context_mask=obs_mask)

        loss = torch.mean(torch.square(ut_theta - ut_ref))

        return loss

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


class CrossAttentionBlock3D(nn.Module):
    def __init__(self, in_channels, d_model, nhead=4):
        super().__init__()
        # The attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            kdim=d_model,  # Key dimension from context
            vdim=d_model,  # Value dimension from context
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x, context, context_mask):
        """
        Args:
            x (Tensor): The spatial feature map from the U-Net [B, C, D, H, W]
            context (Tensor): The conditioning tokens from SetEncoder [B, M, d_model]
            context_mask (Tensor): The padding mask for the context [B, M]
        """
        B, C, D, H, W = x.shape

        # 1. Reshape spatial features to a sequence for attention
        # Query: The pixels/voxels of our image
        query = x.view(B, C, -1).permute(0, 2, 1)  # [B, D*H*W, C]

        # 2. Perform cross-attention
        # The query (our image pixels) attends to the key/value (our observation tokens)
        attn_output, _ = self.attention(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~context_mask  # Invert mask: True means "ignore"
        )

        # 3. Add and normalize (residual connection)
        x_flat = query + attn_output
        x_flat = self.norm(x_flat)

        # 4. Reshape back to the original 3D spatial format
        return x_flat.permute(0, 2, 1).view(B, C, D, H, W)


# A more stable 3D convolutional block using GroupNorm
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        # Ensure num_groups is valid
        if out_channels < groups:
            groups = 1 if out_channels == 1 else 2  # A simple fallback

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU()  # Using SiLU (Swish) as a modern alternative to ReLU
        )

    def forward(self, x):
        return self.block(x)


# The full U-Net architecture with Pad-and-Crop and GroupNorm

# First version with non-dynamic fixed 3 channel unet
# class CrossAttentionUNet3D(nn.Module):
#     def __init__(self, in_channels=64, out_channels=64, channels=[32, 64, 128], d_model=256, nhead=4):
#         super().__init__()
#         self.channels = channels
#
#         # Sanity check: embed_dim must be divisible by nhead for each attn block
#         assert all(c % nhead == 0 for c in channels), "Channel dimensions must be divisible by nhead"
#
#         # Pad to (D,H,W)=(12,12,12) and crop back to (11,11,11)
#         self.pad = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.0)
#
#         # --- Time Embedding ---
#         self.time_embedder = FourierEncoder(d_model)
#         self.time_mlp = nn.Linear(d_model, channels[-1])
#
#         # --- Initial Convolution ---
#         self.init_conv = ConvBlock3D(in_channels, channels[0])
#
#         # --- Encoder ---
#         self.down1 = nn.Sequential(ConvBlock3D(channels[0], channels[1]), nn.MaxPool3d(2))  # 12->6
#         self.attn1 = CrossAttentionBlock3D(in_channels=channels[1], d_model=d_model, nhead=nhead)
#
#         self.down2 = nn.Sequential(ConvBlock3D(channels[1], channels[2]), nn.MaxPool3d(2))  # 6->3
#         self.attn2 = CrossAttentionBlock3D(in_channels=channels[2], d_model=d_model, nhead=nhead)
#
#         # --- Bottleneck ---
#         self.bottleneck = ConvBlock3D(channels[2], channels[2])
#         self.attn_mid = CrossAttentionBlock3D(in_channels=channels[2], d_model=d_model, nhead=nhead)
#
#         # --- Decoder ---
#         self.up1_trans = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)  # 3->6
#         self.up1_conv = ConvBlock3D(channels[1] * 2, channels[1])
#
#         self.up2_trans = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)  # 6->12
#         self.up2_conv = ConvBlock3D(channels[0] * 2, channels[0])
#
#         # --- Final ---
#         self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)
#
#     def forward(self, x, t, context, context_mask):
#         B = x.size(0)
#
#         # 0. Pad to 12x12x12
#         x = self.pad(x)
#
#         # 1. Initial Convolution
#         x1 = self.init_conv(x)
#
#         # 2. Encoder Path
#         x2 = self.down1(x1)
#         x2 = self.attn1(x2, context, context_mask)
#
#         x3 = self.down2(x2)
#         x3 = self.attn2(x3, context, context_mask)
#
#         # 3. Bottleneck
#         bn = self.bottleneck(x3)
#
#         # Add time embedding
#         t_emb = self.time_mlp(self.time_embedder(t.unsqueeze(-1)))
#         bn = bn + t_emb.view(B, self.channels[-1], 1, 1, 1)
#
#         # Add cross-attention in the bottleneck
#         bn = self.attn_mid(bn, context, context_mask)
#
#         # 4. Decoder Path
#         d1 = self.up1_trans(bn)
#         d1 = torch.cat([d1, x2], dim=1)  # Skip connection
#         d1 = self.up1_conv(d1)
#
#         d2 = self.up2_trans(d1)
#         d2 = torch.cat([d2, x1], dim=1)  # Skip connection
#         d2 = self.up2_conv(d2)
#
#         # 5. Final output and crop back to 11x11x11
#         out = self.final_conv(d2)
#         return out[..., :11, :11, :11]

# Second version with dynamic parametric channel unet

class CrossAttentionUNet3D(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, channels=[32, 64, 128], d_model=256, nhead=4, input_size=11):
        super().__init__()

        # Ensure channel dimensions are divisible by the number of attention heads
        assert all(c % nhead == 0 for c in channels), "Channel dimensions must be divisible by nhead"

        self.pad = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0.0)
        self.time_embedder = FourierEncoder(d_model)

        num_levels = len(channels) - 1
        divisor = 2 ** num_levels

        # Calculate the smallest target size divisible by the divisor
        self.target_size = math.ceil(input_size / divisor) * divisor
        total_pad = self.target_size - input_size

        # Distribute padding (e.g., for 5 total, pad with 2 on left, 3 on right)
        pad_front = total_pad // 2
        pad_back = total_pad - pad_front
        self.padding_tuple = (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back)

        # Store the crop indices
        self.crop_start = pad_front
        self.crop_end = pad_front + input_size


        # --- DYNAMICALLY BUILD THE U-NET ---

        # Initial convolution
        self.init_conv = ConvBlock3D(in_channels, channels[0])

        # --- Encoder Path ---
        self.encoders = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(nn.Sequential(ConvBlock3D(channels[i], channels[i + 1]), nn.MaxPool3d(2)))
            self.encoder_attns.append(CrossAttentionBlock3D(in_channels=channels[i + 1], d_model=d_model, nhead=nhead))

        # --- Bottleneck ---
        bottleneck_channels = channels[-1]
        self.bottleneck = ConvBlock3D(bottleneck_channels, bottleneck_channels)
        self.time_mlp = nn.Linear(d_model, bottleneck_channels)  # Projects time to the deepest channel dimension
        self.attn_mid = CrossAttentionBlock3D(in_channels=bottleneck_channels, d_model=d_model, nhead=nhead)

        # --- Decoder Path ---
        self.decoders = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            # Upsampling transpose convolution
            up_conv = nn.ConvTranspose3d(reversed_channels[i], reversed_channels[i + 1], kernel_size=2, stride=2)
            # Convolutional block after concatenating with skip connection
            conv = ConvBlock3D(reversed_channels[i + 1] * 2, reversed_channels[i + 1])
            self.decoders.append(nn.ModuleDict({'up_conv': up_conv, 'conv': conv}))

        # --- Final Convolution ---
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t, context, context_mask):
        # x: [B, C, 11, 11, 11]
        B = x.size(0)
        x = F.pad(x, self.padding_tuple, mode='reflect')

        # Initial conv
        x = self.init_conv(x)

        # --- Encoder with Skip Connections ---
        skip_connections = [x]
        for encoder, attn in zip(self.encoders, self.encoder_attns):
            x = encoder(x)
            x = attn(x, context, context_mask)
            skip_connections.append(x)

        # --- Bottleneck ---
        bn = self.bottleneck(x)
        t_emb = self.time_mlp(self.time_embedder(t.unsqueeze(-1)))
        bn = bn + t_emb.view(B, -1, 1, 1, 1)  # Use -1 to be fully dynamic
        bn = self.attn_mid(bn, context, context_mask)

        # --- Decoder ---
        # We iterate through decoders and the *reversed* skip connections
        x = bn
        for i, decoder_module in enumerate(self.decoders):
            skip = skip_connections[-(i + 2)]  # Get corresponding skip connection
            x = decoder_module['up_conv'](x)
            x = torch.cat([x, skip], dim=1)
            x = decoder_module['conv'](x)

        # --- Final Output ---
        out = self.final_conv(x)
        s = self.crop_start
        e = self.crop_end
        # print(s, e, out.shape)
        return out[..., s:e, s:e, s:e]  # Crop back to original size