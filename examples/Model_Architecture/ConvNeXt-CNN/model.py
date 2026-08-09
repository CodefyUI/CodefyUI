"""
ConvNeXt for CIFAR-10
=====================
ConvNeXt (Liu et al., "A ConvNet for the 2020s", Meta/FAIR 2022) modernizes the
standard ResNet architecture by incorporating design choices from Vision Transformers:

  - Patchify stem (4x4 non-overlapping convolution, stride 4)
  - Inverted bottleneck with depthwise separable convolutions
  - Larger 7x7 depthwise kernels for expanded receptive fields
  - LayerNorm instead of BatchNorm
  - GELU activation instead of ReLU
  - Fewer activation and normalization layers per block

The result is a pure convolutional model that matches or exceeds Swin Transformer
accuracy while retaining the simplicity and efficiency of standard ConvNets.

Architecture: Stem -> 4 stages [3,3,9,3] blocks at dims [96,192,384,768] -> Head
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block: DWConv 7x7 -> LN -> 1x1 expand -> GELU -> 1x1 project."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)                     # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)              # (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)              # back to (B, C, H, W)
        return x + residual


class DownsampleLayer(nn.Module):
    """Spatial downsampling between stages: LN -> 2x2 conv stride 2."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ConvNeXt(nn.Module):
    """
    ConvNeXt model sized for CIFAR-10 (32x32 images).

    The original stem uses a 4x4 stride-4 patchify convolution designed for
    224x224 ImageNet images. For CIFAR-10 we keep the same stem but note that
    the spatial resolution drops to 8x8 after the stem, which is still workable
    for the four subsequent stages.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
    ):
        super().__init__()
        # Patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0]),  # applied after permute in forward
        )

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        # Downsample layers between stages (none after the last stage)
        self.downsamples = nn.ModuleList()
        for i in range(3):
            self.downsamples.append(DownsampleLayer(dims[i], dims[i + 1]))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: conv then channel-last LayerNorm
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return self.head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # -- Data ------------------------------------------------------------------
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=2)

    # -- Model -----------------------------------------------------------------
    model = ConvNeXt(num_classes=10).to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        # -- Train -------------------------------------------------------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # -- Evaluate ----------------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = 100.0 * correct / total

        print(f"Epoch [{epoch:>3}/{num_epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
