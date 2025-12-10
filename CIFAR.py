
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.srcnn import SRCNN
from models.ressrcnn import ResSRCNN
from utils import create_lr, save_examples


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print("Using device:", device)


def train_cifar(
    data_root="data",
    out_dir="outputs",
    epochs=100,
    batch_size=64,
    lr=1e-4,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ResSRCNN(num_res_blocks=8).to(device)
    criterion = nn.L1Loss()  # L1 produces sharper images
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for hr, _ in train_loader:
            hr = hr.to(device)

            lr_up = create_lr(hr, factor=0.5)
            sr = model(lr_up)

            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} — Train Loss: {total_loss / len(train_loader):.6f}")

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for hr, _ in test_loader:
                hr = hr.to(device)
                lr_up = create_lr(hr)
                sr = model(lr_up)
                test_loss += criterion(sr, hr).item()

        print(f"Epoch {epoch}/{epochs} — Test Loss: {test_loss / len(test_loader):.6f}")


        hr_batch, _ = next(iter(test_loader))
        hr_batch = hr_batch.to(device)
        lr_up = create_lr(hr_batch)
        sr_batch = model(lr_up)

        save_examples(lr_up, sr_batch, hr_batch, out_dir / f"cifar_epoch_{epoch:03d}.png")

    torch.save(model.state_dict(), out_dir / "cifar_res_srcnn_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train_cifar()
