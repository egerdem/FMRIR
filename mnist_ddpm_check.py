import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

# --- Basic Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 1. The Standard DDPM Noise Scheduler ---
class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_t = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps].view(-1, 1,
                                                                                                                1, 1)
        return sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise


# --- 2. Standard Sinusoidal Timestep Embedding ---
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t is a tensor of integer timesteps
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# --- 3. A Simplified U-Net for this Test ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        time_out = self.relu(self.time_mlp(t_emb))
        h = h + time_out.unsqueeze(-1).unsqueeze(-1)
        h = self.relu(self.conv2(h))
        return h


class SimpleUNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        self.time_mlp = SinusoidalEmbedding(time_emb_dim)
        self.down1 = Block(1, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bot1 = Block(128, 256, time_emb_dim)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up1 = Block(256 + 128, 128, time_emb_dim)
        self.up2 = Block(128 + 64, 64, time_emb_dim)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):  # Note: This Unet is unconditional
        t_emb = self.time_mlp(t)

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)

        x_bot = self.bot1(self.pool(x2), t_emb)

        x_up = self.up(x_bot)
        x_up = self.up1(torch.cat([x_up, x2], dim=1), t_emb)
        x_up = self.up(x_up)
        x_up = self.up2(torch.cat([x_up, x1], dim=1), t_emb)

        return self.out(x_up)


# --- 4. The DDIM Sampler ---
class DDIMSampler2D:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.num_timesteps = scheduler.num_timesteps

    @torch.no_grad()
    def sample(self, shape, num_inference_steps=50):
        xt = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)

        for t in tqdm(timesteps, desc="DDIM Sampling"):
            predicted_noise = self.model(xt, t.unsqueeze(0).expand(xt.shape[0]))

            alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
            alpha_bar_t_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(device)

            pred_x0 = (xt - sqrt_one_minus_alpha_bar_t * predicted_noise) / torch.sqrt(alpha_bar_t)
            pred_dir_xt = torch.sqrt(1. - alpha_bar_t_prev) * predicted_noise
            xt = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt

        return xt


# --- Main Training and Inference Logic ---
if __name__ == '__main__':
    # Hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 128
    LR = 1e-3

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model and Scheduler
    model = SimpleUNet().to(device)
    scheduler = DDPMScheduler()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images = images.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, scheduler.num_timesteps, (images.shape[0],), device=device).long()

            noisy_images = scheduler.add_noise(images, noise, timesteps)
            predicted_noise = model(noisy_images, timesteps)

            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Inference
    model.eval()
    sampler = DDIMSampler2D(model=model, scheduler=scheduler)
    generated_images = sampler.sample(shape=(100, 1, 28, 28), num_inference_steps=50)

    # Plot results
    grid = make_grid(generated_images, nrow=10, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    plt.axis("off")
    plt.show()