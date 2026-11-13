import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as tfs_v2
from torch.utils.data import DataLoader


class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(
                64, self.hidden_dim, kernel_size=7, stride=1, padding=0
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dim, 64, kernel_size=7, stride=1, padding=0
            ),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h


def train_model(hidden_dim, device, epochs=3):
    print(f"\nОбучение модели с d = {hidden_dim}")
    print("-" * 30)

    model = ConvolutionalAutoEncoder(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    transforms = tfs_v2.Compose(
        [tfs_v2.ToImage(), tfs_v2.ToDtype(dtype=torch.float32, scale=True)]
    )

    d_train = torchvision.datasets.FashionMNIST(
        root="./data", download=True, train=True, transform=transforms
    )
    train_loader = DataLoader(d_train, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        loss_mean = 0
        lm_count = 0
        for X, _ in train_loader:
            X = X.to(device)
            pred, _ = model(X)
            loss = loss_fn(pred, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = (
                1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            )
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {loss_mean:.4f}")

    return model


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Используется устройство: {device}")

# Размеры латентного пространства для эксперимента
latent_dimensions = [2, 8, 32, 64]
trained_models = {}

# Обучаем по очереди каждую модель (по 3 эпохи для экономии времени)
for d in latent_dimensions:
    trained_models[d] = train_model(hidden_dim=d, device=device, epochs=3)

# Подготовка тестовых данных для визуализации
transforms = tfs_v2.Compose(
    [tfs_v2.ToImage(), tfs_v2.ToDtype(dtype=torch.float32, scale=True)]
)
d_test = torchvision.datasets.FashionMNIST(
    root="./data", download=True, train=False, transform=transforms
)

num_images = 5
plt.figure(figsize=(12, 2 * (len(latent_dimensions) + 1)))

# Отрисовка оригиналов (верхняя строчка)
for i in range(num_images):
    img, _ = d_test[i]
    img_np = img.squeeze(0).numpy()

    ax = plt.subplot(len(latent_dimensions) + 1, num_images, i + 1)
    plt.imshow(img_np, cmap="gray")
    if i == 0:
        ax.set_ylabel("Original", fontsize=12, rotation=0, labelpad=40)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Отрисовка реконструкций для каждого размера d
for row_idx, d in enumerate(latent_dimensions):
    model = trained_models[d]
    model.eval()

    for i in range(num_images):
        img, _ = d_test[i]
        with torch.no_grad():
            predict, _ = model(img.unsqueeze(0).to(device))
        dec_img_np = predict.squeeze(0).squeeze(0).detach().cpu().numpy()

        position = (row_idx + 1) * num_images + i + 1
        ax = plt.subplot(len(latent_dimensions) + 1, num_images, position)
        plt.imshow(dec_img_np, cmap="gray")
        if i == 0:
            ax.set_ylabel(f"d = {d}", fontsize=12, rotation=0, labelpad=40)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
