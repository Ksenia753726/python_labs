import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Настройка устройства (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# 2. Загрузка датасета FashionMNIST
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# 3. Определение архитектуры автоэнкодера
class Autoencoder(nn.Module):

  def __init__(self, latent_dim=32):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, latent_dim),
    )

    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 28 * 28),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28)),
    )

  def forward(self, x):
    z = self.encoder(x)
    reconstructed = self.decoder(z)
    return reconstructed


# 4. Инициализация и обучение модели
latent_dim = 32
model = Autoencoder(latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
print("Начинаем обучение автоэнкодера...")
for epoch in range(epochs):
  model.train()
  total_loss = 0.0
  for images, _ in train_loader:
    images = images.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, images)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * images.size(0)

  avg_loss = total_loss / len(train_dataset)
  print(f"Эпоха [{epoch+1}/{epochs}], Потери (MSE): {avg_loss:.4f}")

# 5. Извлечение латентных кодов всего датасета
print("\nИзвлекаем латентные коды всего датасета...")
model.eval()
all_latent_codes = []

with torch.no_grad():
  for images, _ in train_loader:
    images = images.to(device)
    z = model.encoder(images)
    all_latent_codes.append(z)

# Объединяем все батчи в единую матрицу Z на GPU размерностью (60000, 32)
Z = torch.cat(all_latent_codes, dim=0)
print(f"Форма матрицы всех латентных кодов Z: {Z.shape}")

# 6. Расчет среднего вектора и ковариационной матрицы на GPU
mean_vector = torch.mean(Z, dim=0)
cov_matrix = torch.cov(Z.T)

print(f"Форма среднего вектора: {mean_vector.shape}")
print(f"Форма ковариационной матрицы: {cov_matrix.shape}")

# 7. Расчет спектра собственных значений (PCA) и эффективных измерений
print("\nВычисляем спектр собственных значений и эффективную размерность...")
eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Сортируем по убыванию (от самой важной компоненты к самой слабой)
eigenvalues = torch.flip(eigenvalues, dims=[0])

# Защита от микроскопических отрицательных чисел из-за погрешности float32
eigenvalues = torch.clamp(eigenvalues, min=0.0)

# Доля объясненной дисперсии и накопленная сумма
total_variance = torch.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)

# Пороги 95% и 99% информации
dim_95 = torch.searchsorted(cumulative_variance, 0.95).item() + 1
dim_99 = torch.searchsorted(cumulative_variance, 0.99).item() + 1

# Расчет эффективной размерности (Participation Ratio)
# Формула: (Сумма всех СЗ)^2 / Сумма квадратов всех СЗ
participation_ratio = (total_variance**2) / torch.sum(eigenvalues**2)
pr_val = participation_ratio.item()

# Нормированный PR (в процентах от общей емкости латентного слоя)
pr_utilization = (pr_val / latent_dim) * 100

print("-" * 50)
print(f"Всего выделено измерений (latent_dim): {latent_dim}")
print(f"Эффективная размерность (Participation Ratio): {pr_val:.2f}")
print(f"Эффективность использования пространства: {pr_utilization:.1f}%")
print(f"Измерений для сохранения 95% информации: {dim_95}")
print(f"Измерений для сохранения 99% информации: {dim_99}")
print("-" * 50)

# --- ДОПОЛНЕНИЕ В БЛОК 7: Расчет ранга матрицы Z по сингулярным числам ---

# 1. Обязательно центрируем матрицу латентных кодов перед SVD
Z_centered = Z - mean_vector

# 2. Вычисляем сингулярные числа на GPU (svdvals быстрее полного svd, так как не считает векторы)
singular_values = torch.linalg.svdvals(Z_centered)

# 3. Классический матричный ранг по сингулярным числам
# (считает количество сингулярных чисел выше стандартного численного порога)
matrix_rank = torch.linalg.matrix_rank(Z_centered).item()

# 4. Ранг с пользовательским порогом отсечения шума (например, 1% от максимального сингулярного числа)
tol = 0.01 * singular_values[0]
rank_filtered = torch.sum(singular_values > tol).item()

# 5. Энтропийный ранг (Effective Rank по Roy & Vetterli) через сингулярные числа
# Нормируем сингулярные числа в распределение вероятностей
p = singular_values / torch.sum(singular_values)
# Убираем нули для безопасного логарифмирования
p = p[p > 0]
shannon_entropy = -torch.sum(p * torch.log(p))
effective_rank_svd = torch.exp(shannon_entropy).item()

print("-" * 50)
print(f"Алгебраический ранг матрицы Z (SVD): {matrix_rank} / {latent_dim}")
print(f"Ранг без учета мелкого шума (>1% от σ_max): {rank_filtered} / {latent_dim}")
print(f"Энтропийный эффективный ранг (Effective Rank): {effective_rank_svd:.2f}")
print("-" * 50)


# 8. Визуализация спектра и эффективной размерности
ev_ratio = explained_variance_ratio.cpu().numpy() * 100
cum_ratio = cumulative_variance.cpu().numpy() * 100
components = np.arange(1, latent_dim + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
plt.subplots_adjust(hspace=0.25)

# ==============================================================================
# ГРАФИК 1: Накопленная дисперсия (Зоны полезного сигнала и избыточности)
# ==============================================================================
ax1.axvspan(
    1,
    dim_95,
    color="mediumseagreen",
    alpha=0.15,
    label=f"Активная зона: 95% инф. ({dim_95} изм.)",
)
ax1.axvspan(
    dim_95,
    latent_dim,
    color="lightcoral",
    alpha=0.15,
    label=f"Зона шума / избыточности ({latent_dim - dim_95} изм.)",
)

ax1.plot(
    components,
    cum_ratio,
    color="darkblue",
    linewidth=2.5,
    marker="o",
    markersize=5,
)

# Пороги 95% и 99%
ax1.axhline(95, color="green", linestyle="--", linewidth=1.2)
ax1.axvline(dim_95, color="green", linestyle="--", linewidth=1.2)
ax1.scatter([dim_95], [cum_ratio[dim_95 - 1]], color="green", s=80, zorder=5)
ax1.text(
    dim_95 + 0.4,
    88,
    f"95% ({dim_95} комп.)",
    color="darkgreen",
    fontweight="bold",
)

ax1.axhline(99, color="darkorange", linestyle=":", linewidth=1.2)
ax1.axvline(dim_99, color="darkorange", linestyle=":", linewidth=1.2)
ax1.scatter(
    [dim_99], [cum_ratio[dim_99 - 1]], color="darkorange", s=80, zorder=5
)
ax1.text(
    dim_99 + 0.4,
    92,
    f"99% ({dim_99} комп.)",
    color="darkorange",
    fontweight="bold",
)

# Карточка с итогом Participation Ratio
ax1.text(
    0.98,
    0.15,
    f"Эффективная размерность (PR): {pr_val:.2f} / {latent_dim}\n"
    f"Задействовано емкости: {pr_utilization:.1f}%",
    transform=ax1.transAxes,
    ha="right",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="royalblue",
        linewidth=1.5,
        alpha=0.95,
    ),
)

ax1.set_ylabel("Накопленная информация (%)", fontsize=11)
ax1.set_title(
    "Спектр собственных значений ковариации (PCA латентного пространства)",
    fontsize=13,
    fontweight="bold",
)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", frameon=True)

# ==============================================================================
# ГРАФИК 2: Вклад каждой компоненты и отметка Participation Ratio
# ==============================================================================
colors = [
    "mediumseagreen" if i < dim_95 else "slategray" for i in range(latent_dim)
]
ax2.bar(components, ev_ratio, color=colors, edgecolor="black", alpha=0.8)
ax2.set_yscale("log")

# Подписываем точные проценты над первыми 5 столбцами
for i in range(min(5, latent_dim)):
  height = ev_ratio[i]
  ax2.text(
      components[i],
      height * 1.15,
      f"{height:.1f}%",
      ha="center",
      va="bottom",
      fontsize=9,
      fontweight="bold",
  )

# Добавляем вертикальную линию эффективной размерности (PR)
ax2.axvline(
    pr_val,
    color="royalblue",
    linestyle="-.",
    linewidth=2,
    label=f"Эффективная размерность PR = {pr_val:.2f}",
)
ax2.axvline(
    dim_95 + 0.5,
    color="green",
    linestyle="--",
    linewidth=1.2,
    label=f"Порог 95% ({dim_95} комп.)",
)

ax2.set_xlabel(
    "Номер главной компоненты (отсортированы по убыванию)", fontsize=11
)
ax2.set_ylabel("Вклад компоненты, % (лог. шкала)", fontsize=11)
ax2.set_xticks(components)
ax2.grid(True, which="both", linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", frameon=True)
ax2.set_ylim(bottom=max(ev_ratio.min() * 0.5, 1e-4), top=ev_ratio.max() * 2.5)

plt.tight_layout()
plt.show()
