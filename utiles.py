import torch
from torchvision.utils import save_image, make_grid


def create_lr(hr, factor=0.5):
    """
    Create low-resolution (LR) image:
    - Downsample by factor
    - Upsample back to original size
    """
    lr_small = torch.nn.functional.interpolate(
        hr, scale_factor=factor, mode="bicubic", align_corners=False
    )
    lr_up = torch.nn.functional.interpolate(
        lr_small, size=hr.shape[-2:], mode="bicubic", align_corners=False
    )
    return lr_up


def save_examples(lr, sr, hr, out_path, num=8):
    """Save grid of LR / SR / HR for visualization."""
    lr = lr[:num].cpu()
    sr = sr[:num].cpu()
    hr = hr[:num].cpu()

    grid = make_grid(torch.cat([lr, sr, hr], dim=0), nrow=num, padding=2)
    save_image(grid, out_path)
