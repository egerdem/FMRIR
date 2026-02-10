from torch.optim.lr_scheduler import _LRScheduler

import math

class ThreePhaseScheduler(_LRScheduler):
    """
    A custom LR scheduler that implements a three-phase schedule:
    1. Linear warm-up from a start_factor to 1.0.
    2. Cosine annealing decay from the peak LR down to a minimum LR.
    3. Constant "coast" phase at the minimum LR.
    """

    def __init__(self, optimizer, total_iterations, warmup_iterations, decay_iterations,
                 peak_lr, min_lr, start_factor=0.01, last_epoch=-1):
        self.total_iters = total_iterations
        self.warmup_iters = warmup_iterations
        self.decay_iters = decay_iterations
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Phase 1: Linear Warm-up
            start_lr = self.peak_lr * self.start_factor
            progress = self.last_epoch / self.warmup_iters
            return [start_lr + (self.peak_lr - start_lr) * progress]

        elif self.last_epoch < self.decay_iters:
            # Phase 2: Cosine Decay
            progress = (self.last_epoch - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            cos_out = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (self.peak_lr - self.min_lr) * cos_out]

        else:
            # Phase 3: Coast at min_lr
            return [self.min_lr]


# verify_scheduler.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Simulation Parameters (match your training run) ---
NUM_ITERATIONS = 700000
WARMUP_ITERATIONS = 5000
DECAY_ITERATIONS = 700000
PEAK_LR = 1e-4
MIN_LR = 1e-5

# 1. Create a dummy model and optimizer
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR)

# 2. Instantiate our new scheduler
scheduler = ThreePhaseScheduler(
    optimizer=optimizer,
    total_iterations=NUM_ITERATIONS,
    warmup_iterations=WARMUP_ITERATIONS,
    decay_iterations=DECAY_ITERATIONS,
    peak_lr=PEAK_LR,
    min_lr=MIN_LR
)

# 3. Simulate the training loop
lrs = []
print("Simulating scheduler steps...")
for i in range(NUM_ITERATIONS):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()
print("Simulation complete.")

# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(lrs)
plt.title("Learning Rate Schedule Verification")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.axvline(x=WARMUP_ITERATIONS, color='g', linestyle='--', label='End of Warm-up')
plt.axvline(x=DECAY_ITERATIONS, color='r', linestyle='--', label='End of Decay / Start of Coast')
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig("lr_schedule_plot.png")
print("\nPlot saved to lr_schedule_plot.png")
print(f"LR at start: {lrs[0]:.2e}")
print(f"LR at end of warm-up ({WARMUP_ITERATIONS}): {lrs[WARMUP_ITERATIONS]:.2e}")
print(f"LR at start of coast ({DECAY_ITERATIONS}): {lrs[DECAY_ITERATIONS]:.2e}")
print(f"LR at end: {lrs[-1]:.2e}")