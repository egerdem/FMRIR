import torch, time

device = "cuda"
B = 4
C = 20
D = 11
N = D**3
M_range = (5, 50)
M_max = M_range[1]

z_full = torch.randn(B, C, D, D, D, device=device)
src_xyz = torch.randn(B, 3, device=device)
grid_xyz = torch.randn(N, 3, device=device)

def old_style():
    for i in range(B):
        M = torch.randint(M_range[0], M_range[1] + 1, (1,), device=device).item()
        idx = torch.randperm(N, device=device)[:M]
        obs_xyz = grid_xyz[idx]
        rel = obs_xyz - src_xyz[i].unsqueeze(0)
        z_flat = z_full[i].view(C, -1)
        vals = z_flat[:, idx].transpose(0, 1)

def new_style():
    M = torch.randint(M_range[0], M_range[1] + 1, (B,), device=device)
    scores = torch.rand(B, N, device=device)
    idx = torch.topk(scores, k=M_max, dim=1).indices
    obs_xyz = grid_xyz[idx]
    rel = obs_xyz - src_xyz.unsqueeze(1)
    z_flat = z_full.view(B, C, -1)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1)
    vals = torch.gather(z_flat, 2, idx_expand).transpose(1, 2)

# warmup
for _ in range(100):
    old_style()
    new_style()

torch.cuda.synchronize()

# timing
iters = 2000

start = time.time()
for _ in range(iters):
    old_style()
torch.cuda.synchronize()
t_old = time.time() - start

start = time.time()
for _ in range(iters):
    new_style()
torch.cuda.synchronize()
t_new = time.time() - start

print(f"Old avg per call: {t_old/iters*1000:.4f} ms")
print(f"New avg per call: {t_new/iters*1000:.4f} ms")
