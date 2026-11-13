import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

full_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
train_ds, val_ds = random_split(full_train, [55000, 5000],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, d))
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 784))

    def encode(self, x):
        return self.enc(x), None          # (код, None)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z, None       # (логиты, μ, logvar)


class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)
        self.dec = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def loss_fn(logits, x, mu, logvar):
    n = x.size(0)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n
    if logvar is None:                    # AE: только reconstruction
        return rec, rec, torch.tensor(0.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
    return rec + kl, rec, kl              # −ELBO


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, _ in loader:
            x = imgs.view(imgs.size(0), -1).to(device)
            logits, mu, logvar = model(x)
            loss, _, _ = loss_fn(logits, x, mu, logvar)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
    return total / count


def fit(model, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train": [], "val": []}
    for _ in range(epochs):
        hist["train"].append(run_epoch(model, train_loader, opt))
        hist["val"].append(run_epoch(model, val_loader))
    return hist


# Обучение: AE и VAE для d = 2 и d = 16
results = {}
for d in [2, 16]:
    for name, cls in [("AE", AE), ("VAE", VAE)]:
        torch.manual_seed(0)
        model = cls(d).to(device)
        hist = fit(model, epochs=15)
        results[(name, d)] = (model, hist)
        print(f"{name}, d={d}: финальный val loss = {hist['val'][-1]:.2f}")


@torch.no_grad()
def collect(model, loader, sample=False):
    Zs, Ys = [], []
    for imgs, y in loader:
        x = imgs.view(imgs.size(0), -1).to(device)
        if isinstance(model, VAE):
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar) if sample else mu
        else:
            z, _ = model.encode(x)
        Zs.append(z.cpu())
        Ys.append(y)
    return torch.cat(Zs).double(), torch.cat(Ys)


def spectrum(Z):
    Zc = Z - Z.mean(0)
    cov = Zc.T @ Zc / (Z.shape[0] - 1)
    lam = torch.flip(torch.linalg.eigvalsh(cov).clamp_min(0), [0])
    cum = (lam / lam.sum()).cumsum(0)
    k95 = int((cum < 0.95).sum()) + 1
    pr = (lam.sum() ** 2 / (lam ** 2).sum()).item()
    rank = torch.linalg.matrix_rank(Zc).item()
    return lam, cov, k95, pr, rank


def gaussian_check(Z):
    d = Z.shape[1]
    mean = Z.mean(0)
    cov = torch.cov(Z.T)
    return {
        "|mean|": mean.norm().item(),
        "||cov − I||_F": torch.linalg.norm(cov - torch.eye(d, dtype=Z.dtype)).item(),
        "std по измерениям": Z.std(0).numpy().round(2),
    }


def plot_latent(Z, y, title, ax):
    Zn = Z.numpy()
    if Zn.shape[1] == 2:
        emb = Zn
    else:
        idx = np.random.RandomState(0).choice(len(Zn), 3000, replace=False)
        Zn = Zn[idx]; y = y[idx]
        emb = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(Zn)
    ax.scatter(emb[:, 0], emb[:, 1], c=y.numpy() if torch.is_tensor(y) else y,
               cmap="tab10", s=3)
    ax.set_title(title)


# Сравнение: метрики распределения кодов
for d in [2, 16]:
    for name in ["AE", "VAE"]:
        model, _ = results[(name, d)]
        model.eval()
        Z, _ = collect(model, test_loader, sample=(name == "VAE"))
        lam, _, k95, pr, rank = spectrum(Z)
        g = gaussian_check(Z)
        print(f"\n{name}, d={d}")
        print(f"  k95={k95}, participation ratio={pr:.2f}, rank={rank}")
        print(f"  |mean|={g['|mean|']:.3f}, ||cov−I||_F={g['||cov − I||_F']:.3f}")


imgs_test, y_test = next(iter(test_loader))
x_test = imgs_test.view(imgs_test.size(0), -1).to(device)
n = 8

fig, axes = plt.subplots(5, n, figsize=(n * 1.2, 6.8))
rows = []
for d in [2, 16]:
    for name in ["AE", "VAE"]:
        model, _ = results[(name, d)]
        model.eval()
        with torch.no_grad():
            recon = model.decode(model.encode(x_test)[0]) if name == "AE" \
                else model.decode(model.encode(x_test)[0])
        rows.append((f"{name} d={d}", recon.cpu().view(-1, 28, 28)))

for j in range(n):
    axes[0, j].imshow(imgs_test[j, 0], cmap="gray"); axes[0, j].axis("off")
axes[0, 0].set_title("Оригинал", loc="left")
for r, (label, recon) in enumerate(rows, start=1):
    for j in range(n):
        axes[r, j].imshow(recon[j], cmap="gray"); axes[r, j].axis("off")
    axes[r, 0].set_title(label, loc="left")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for i, name in enumerate(["AE", "VAE"]):
    for j, d in enumerate([2, 16]):
        model, _ = results[(name, d)]
        model.eval()
        Z, y = collect(model, test_loader, sample=(name == "VAE"))
        plot_latent(Z, y, f"{name}, d={d}", axes[i, j])
plt.tight_layout()
plt.show()


vae2, _ = results[("VAE", 2)]
vae2.eval()
with torch.no_grad():
    grid = torch.linspace(-2.5, 2.5, 12)
    zs = torch.cartesian_prod(grid, grid).to(device)  # равномерная сетка в латентном пространстве
    imgs = vae2.decode(zs).cpu().view(-1, 28, 28)

plt.figure(figsize=(8, 8))
side = 12
canvas = imgs.view(side, side, 28, 28).permute(0, 2, 1, 3).reshape(side * 28, side * 28)
plt.imshow(canvas, cmap="gray"); plt.axis("off")
plt.title("VAE, d=2: декодирование сетки в латентном пространстве")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name in ["AE", "VAE"]:
    for d in [16]:
        model, _ = results[(name, d)]
        model.eval()
        Z, _ = collect(model, test_loader, sample=(name == "VAE"))
        lam, _, _, _, _ = spectrum(Z)
        cum = (lam / lam.sum()).cumsum(0)
        ax[0].semilogy(range(1, d + 1), lam.numpy(), "o-", label=name)
        ax[1].plot(range(1, d + 1), cum.numpy(), "o-", label=name)
ax[0].set_title("Спектр ковариации, d=16"); ax[0].set_xlabel("Компонента"); ax[0].legend(); ax[0].grid(True)
ax[1].set_title("Накопленная доля дисперсии, d=16"); ax[1].set_xlabel("Число компонент")
ax[1].axhline(0.95, color="r", linestyle="--"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()
