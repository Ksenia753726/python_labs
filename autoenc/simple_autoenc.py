import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),  # bottleneck без активации
        )
        # Декодер: восстанавливает вход из латентного вектора
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # выход в диапазоне [0, 1], как у пикселей
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


transform = transforms.ToTensor()
train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

model = Autoencoder(input_dim=784, latent_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for images, _ in loader:
        x = images.view(images.size(0), -1).to(device)  # (B, 1, 28, 28) -> (B, 784)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    for images, _ in loader:
        x = images.view(images.size(0), -1).to(device)
        total_loss += criterion(model(x), x).item() * x.size(0)
    return total_loss / len(loader.dataset)


history = {"train": [], "val": []}
epochs = 5
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(train_loader)
    val_loss = evaluate(test_loader)
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    print(f"Epoch {epoch:2d} | train MSE: {train_loss:.5f} | val MSE: {val_loss:.5f}")

# Визуализация: оригиналы и их реконструкции
model.eval()
all_codes = []
with torch.no_grad():
    for images, _ in train_loader:  # или test_loader, в зависимости от задачи
        x = images.view(images.size(0), -1).to(device)
        z = model.encoder(x)          # только энкодер, размер [B, 32]
        all_codes.append(z.cpu())

Z = torch.cat(all_codes, dim=0).double()
print("Размер матрицы кодов:", Z.shape)

mean_z = Z.mean(dim=0)  # [32]
cov_z = torch.cov(Z.T)
N, D = Z.shape

print("Среднее:", mean_z.shape)
print("Ковариация:", cov_z.shape)

with torch.no_grad():
    test_images, _ = next(iter(test_loader))
    x_test = test_images.view(test_images.size(0), -1).to(device)
    recon = model(x_test).cpu().view(-1, 1, 28, 28)

n = 8
fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
for i in range(n):
    axes[0, i].imshow(test_images[i, 0], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i, 0], cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_title("Оригинал", loc="left")
axes[1, 0].set_title("Реконструкция", loc="left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), history["train"], label="train")
plt.plot(range(1, epochs + 1), history["val"], label="val")
plt.xlabel("Эпоха")
plt.ylabel("Reconstruction loss (MSE)")
plt.title("Кривые обучения автоэнкодера")
plt.legend()
plt.grid(True)
plt.show()

# Использование энкодера отдельно: получить сжатое представление
with torch.no_grad():
    codes = model.encoder(x[:5])
    print("Размерность латентного кода:", codes.shape)  # torch.Size([5, 32])


# Собственные значения (по возрастанию), переводим в убывающий порядок
eigvals = torch.linalg.eigvalsh(cov_z)
eigvals = torch.flip(eigvals.clamp_min(0), dims=[0])  # clamp убирает шум вида -1e-17

# Доля объяснённой дисперсии и накопленная доля
ratio = eigvals / eigvals.sum()
cum = ratio.cumsum(dim=0)


# Число компонент, нужных для заданной доли дисперсии
k90 = int((cum < 0.90).sum().item()) + 1
k95 = int((cum < 0.95).sum().item()) + 1
k99 = int((cum < 0.99).sum().item()) + 1

# Participation ratio: «эффективное число» измерений
eff_dim = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
Zc = Z - Z.mean(dim=0)  # центрирование
rank = torch.linalg.matrix_rank(Zc)
print("Ранг матрицы кодов:", rank.item())

print(f"Компонент для 90% дисперсии: {k90}")
print(f"Компонент для 95% дисперсии: {k95}")
print(f"Компонент для 99% дисперсии: {k99}")
print(f"Эффективная размерность (participation ratio): {eff_dim:.2f} из {D}")

# Графики: спектр и накопленная дисперсия
idx = range(1, D + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(idx, eigvals.numpy(), "o-")
ax[0].set_xlabel("Номер компоненты")
ax[0].set_ylabel("Собственное значение (лог. шкала)")
ax[0].set_title("Спектр ковариации латентного пространства")
ax[0].grid(True)

ax[1].plot(idx, cum.numpy(), "o-")
ax[1].axhline(0.95, color="r", linestyle="--", label="95%")
ax[1].set_xlabel("Число компонент")
ax[1].set_ylabel("Накопленная доля дисперсии")
ax[1].set_title("Объяснённая дисперсия")
ax[1].legend()
ax[1].grid(True)
plt.tight_layout()
plt.show()
