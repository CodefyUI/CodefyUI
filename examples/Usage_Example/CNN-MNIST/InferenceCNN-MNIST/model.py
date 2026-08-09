"""
Inference CNN on MNIST — Load Trained Weights & Predict
========================================================
Corresponds to the graph.json that uses:
  - SequentialModel  →  same architecture as TrainCNN-MNIST
  - ModelLoader      →  load saved weights (state_dict)
  - ImageReader      →  load a test image
  - Inference        →  forward pass with eval() + no_grad()
  - Softmax + Print  →  show prediction probabilities

Prerequisites:
  1. Train the model first using the TrainCNN-MNIST example, or run:
       python ../TrainCNN-MNIST/model.py
  2. This script saves/loads weights from ./mnist_cnn.pt

Run: python model.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Model: identical architecture to TrainCNN-MNIST ──────────────

class SimpleCNN(nn.Module):
    """
    Must match the training architecture exactly for state_dict loading:
      Conv2d(1→32, k=3, p=1) → ReLU → MaxPool(2)
      Conv2d(32→64, k=3, p=1) → ReLU → MaxPool(2)
      Flatten → FC(3136→128) → ReLU → FC(128→10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Device selection ─────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Main ─────────────────────────────────────────────────────────

WEIGHTS_PATH = "mnist_cnn.pt"


def train_and_save():
    """Train the model and save weights (run once)."""
    import torch.optim as optim

    device = select_device()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("Training for 5 epochs...")
    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch}/5 — Loss: {total_loss / len(train_loader):.4f}")

    # Save weights (state_dict) — same as ModelSaver node
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\nWeights saved to {WEIGHTS_PATH}")


@torch.no_grad()
def run_inference():
    """Load weights and run inference — mirrors the graph.json pipeline."""
    device = select_device()
    print(f"Device: {device}")

    # 1. Build model architecture (= SequentialModel node)
    model = SimpleCNN()

    # 2. Load trained weights (= ModelLoader node, state_dict mode)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 3. Switch to eval mode + no_grad (= Inference node)
    model.eval()

    # 4. Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 5. Run inference on full test set
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)                   # raw output (= Inference node output)
        probs = torch.softmax(logits, dim=-1)    # (= Softmax node)
        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Test Accuracy: {correct}/{total} ({correct / total:.1%})")

    # 6. Show a few individual predictions
    sample_images, sample_labels = next(iter(DataLoader(test_dataset, batch_size=10, shuffle=True)))
    sample_images = sample_images.to(device)
    logits = model(sample_images)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=1)

    print("\nSample predictions:")
    print(f"  {'True':>5}  {'Pred':>5}  {'Confidence':>10}")
    print(f"  {'-'*25}")
    for i in range(10):
        print(f"  {sample_labels[i].item():>5d}  {preds[i].item():>5d}  {probs[i, preds[i]].item():>9.1%}")


if __name__ == "__main__":
    from pathlib import Path

    if not Path(WEIGHTS_PATH).exists():
        print(f"No weights found at {WEIGHTS_PATH}, training first...\n")
        train_and_save()
        print()

    run_inference()
