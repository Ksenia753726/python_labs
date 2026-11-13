import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as tfs_v2
from torch.utils.data import DataLoader
from tqdm import tqdm


class ConvolutionalAutoEncoderMNIST(nn.Module):

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Вход: [batch, 1, 28, 28] -> Выход: [batch, hidden_dim, 1, 1]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 7, 7]
            nn.ELU(inplace=True),
            nn.Conv2d(64, self.hidden_dim, kernel_size=7, stride=1, padding=0),  # [batch, hidden_dim, 1, 1]
        )

        # Вход: [batch, hidden_dim, 1, 1] -> Выход: [batch, 1, 28, 28]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=7, stride=1, padding=0),  # [batch, 64, 7, 7]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 14, 14]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 28, 28]
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h


def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(dataloader, leave=True)
    for batch, (X, _) in enumerate(train_tqdm):
        X = X.to(device)

        pred, _ = model(X)
        loss = loss_fn(pred, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(
            f"Epoch [{epoch}/{epochs}], Train Loss={loss_mean:.4f}"
        )


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred, _ = model(X)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    print(f"Test Error: Avg validation loss: {test_loss:.4f} \n")


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Используется устройство: {device}")

model = ConvolutionalAutoEncoderMNIST(hidden_dim=128).to(device)

transforms = tfs_v2.Compose(
    [
        tfs_v2.ToImage(),
        tfs_v2.ToDtype(dtype=torch.float32, scale=True),
    ]
)

d_train = torchvision.datasets.FashionMNIST(
    root="./data", download=True, train=True, transform=transforms
)
d_test = torchvision.datasets.FashionMNIST(
    root="./data", download=True, train=False, transform=transforms
)

batch_size = 32
train_dataloader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(d_test, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 5

for t in range(epochs):
    print(f"epoch: {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device, t + 1, epochs)
    test_loop(test_dataloader, model, loss_fn, device)
print("done")

torch.save(model.state_dict(), "model_conv_autoencoder_mnist.tar")

n = 10
model.eval()

plt.figure(figsize=(2 * n, 2 * 2))
for i in range(n):
    img, _ = d_train[i]

    predict, _ = model(img.unsqueeze(0).to(device))
    predict = predict.squeeze(0)

    img_np = img.squeeze(0).detach().cpu().numpy()
    dec_img_np = predict.squeeze(0).detach().cpu().numpy()

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img_np, cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax2 = plt.subplot(2, n, i + n + 1)
    plt.imshow(dec_img_np, cmap="gray")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

plt.show()
