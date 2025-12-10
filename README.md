 Deep Learning Super-Resolution on MNIST and CIFAR-10

This project implements **deep learning–based image super-resolution** using two benchmark datasets:

- MNIST (28×28 grayscale digits)
- CIFAR-10 (32×32 RGB natural images)

The goal is to train a neural network that can take a **low-resolution (LR) image and reconstruct a high-resolution (HR) version with improved sharpness and detail.

This repository includes:
- Classical SRCNN (Super-Resolution CNN)
- A deeper ResSRCNN (Residual SRCNN) with multiple residual blocks
- Full training pipelines for MNIST and CIFAR-10
- Visualization utilities
- Ready-to-run scripts for CPU, GPU, and Apple Silicon (MPS)


