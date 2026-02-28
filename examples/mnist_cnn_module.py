"""
ScryNeuro Example: CNN MNIST Training Module
=============================================

This module contains all Python-side logic for the CNN MNIST pipeline.
Prolog imports this module and calls its functions — no inline Python strings needed.

Benefits of this pattern:
  - Python code gets full IDE support (autocomplete, type checking, linting)
  - Easier to debug, test, and modify the Python side independently
  - Prolog stays clean: orchestration logic only, no embedded strings

Usage from Prolog:
    ?- py_import("mnist_cnn_module", M).
    ?- py_call(M, "create_pipeline", Pipeline).
    ?- py_call(Pipeline, "load_data", InfoDict).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class MnistCNN(nn.Module):
    """Simple CNN for MNIST digit classification.

    Architecture:
        Conv2d(1,16,3,pad=1) -> ReLU -> MaxPool(2)
        Conv2d(16,32,3,pad=1) -> ReLU -> MaxPool(2)
        Flatten -> Linear(32*7*7, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Pipeline Class — encapsulates all training state
# ---------------------------------------------------------------------------


class MnistPipeline:
    """End-to-end MNIST CNN training pipeline.

    Each method corresponds to a pipeline step that Prolog can call.
    All shared state (model, device, loaders) lives here — Prolog only
    holds a single handle to this object.
    """

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.model: Optional[MnistCNN] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.train_dataset: Optional[datasets.MNIST] = None
        self.test_dataset: Optional[datasets.MNIST] = None

    # -- Step 1+2: Load data ------------------------------------------------

    def load_data(self, data_dir: str = "./data") -> dict:
        """Download MNIST and create data loaders.

        Returns:
            dict with keys 'train_size' and 'test_size'.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.train_dataset = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
        )

        return {
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
        }

    # -- Step 3: Setup model + optimizer ------------------------------------

    def setup(self) -> str:
        """Instantiate model, optimizer, and loss function.

        Automatically detects CUDA/MPS/CPU.

        Returns:
            Device name string (e.g. "cuda", "cpu").
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = MnistCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        return str(self.device)

    # -- Step 4: Train one epoch --------------------------------------------

    def train_one_epoch(self) -> dict:
        """Train for one epoch over the full training set.

        Returns:
            dict with keys 'loss' (float) and 'accuracy' (float, percentage).
        """
        assert self.model is not None, "Call setup() first"
        assert self.train_loader is not None, "Call load_data() first"

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(batch_x)
            loss = self.criterion(out, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (out.argmax(1) == batch_y).sum().item()
            total += batch_x.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": 100.0 * correct / total,
        }

    # -- Step 5: Evaluate on test set ---------------------------------------

    def evaluate(self) -> float:
        """Evaluate model accuracy on the test set.

        Returns:
            Test accuracy as a percentage (float).
        """
        assert self.model is not None, "Call setup() first"
        assert self.test_loader is not None, "Call load_data() first"

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = self.model(batch_x)
                correct += (out.argmax(1) == batch_y).sum().item()
                total += batch_x.size(0)

        return 100.0 * correct / total

    # -- Step 6: Single-image inference (neural predicate) ------------------

    def predict_digit(self, index: int) -> int:
        """Predict the digit class for a single test image.

        Args:
            index: Index into the test dataset.

        Returns:
            Predicted class (0-9).
        """
        assert self.model is not None, "Call setup() first"
        assert self.test_dataset is not None, "Call load_data() first"

        self.model.eval()
        with torch.no_grad():
            image = self.test_dataset[index][0].unsqueeze(0).to(self.device)
            output = self.model(image)
            return int(output.argmax(1).item())

    def true_label(self, index: int) -> int:
        """Get the ground-truth label for a test image.

        Args:
            index: Index into the test dataset.

        Returns:
            True label (0-9).
        """
        assert self.test_dataset is not None, "Call load_data() first"
        return int(self.test_dataset[index][1])

    # -- Step 7: Save model -------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save the trained model's state_dict to a file.

        Args:
            path: File path for the saved model (e.g. "mnist_cnn.pt").
        """
        assert self.model is not None, "Call setup() first"
        torch.save(self.model.state_dict(), path)


# ---------------------------------------------------------------------------
# Module-level factory — called from Prolog via py_call(Module, "create_pipeline", P)
# ---------------------------------------------------------------------------


def create_pipeline() -> MnistPipeline:
    """Create a new MnistPipeline instance.

    This is the entry point called from Prolog:
        py_import("mnist_cnn_module", M),
        py_call(M, "create_pipeline", Pipeline).
    """
    return MnistPipeline()
