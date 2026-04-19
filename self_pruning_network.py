#!/usr/bin/env python3
"""
Self-Pruning Neural Network on CIFAR-10
========================================
A neural network that learns to prune itself during training by using
learnable gate parameters on each weight. An L1 sparsity penalty on the
sigmoid-activated gates drives unimportant connections to zero.

Author : Swapnil
Project: Tredence Analytics — AI Engineering Intern Case Study
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOT_DIR = PROJECT_ROOT / "plots"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("self_pruning_network")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPARSITY_THRESHOLD: float = 5e-2  # Gate values below this are considered pruned


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — PrunableLinear Layer
# ═══════════════════════════════════════════════════════════════════════════
class PrunableLinear(nn.Module):
    """A fully-connected layer with learnable per-weight gate parameters.

    Each weight ``w_ij`` is multiplied element-wise by
    ``sigmoid(gate_score_ij)`` before the linear transformation.  During
    training, an L1 penalty on the gate values drives unimportant gates
    towards 0, effectively pruning the corresponding weights.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight & bias — Kaiming uniform init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Gate scores — same shape as weight; initialised to 3.0 so that
        # sigmoid(3.0) ≈ 0.953 (gates start near fully-open).  Starting
        # near 1 means the CE gradient can keep important connections at
        # high values while L1 drives unimportant ones to exactly 0,
        # producing the expected bimodal gate distribution.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialise weights (Kaiming), bias (zero), gate scores (3.0 → sigmoid ≈ 0.953)."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)
        # sigmoid(3.0) ≈ 0.953 — gates start near fully-open.  Important
        # connections are kept there by the CE gradient; unimportant ones
        # are pushed to 0 by L1, creating the bimodal distribution.
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_features)``.
        """
        gates = torch.sigmoid(self.gate_scores)               # ∈ (0, 1)
        pruned_weights = self.weight * gates                   # element-wise
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Self-Pruning Network
# ═══════════════════════════════════════════════════════════════════════════
class SelfPruningNet(nn.Module):
    """Feed-forward classifier for CIFAR-10 using ``PrunableLinear`` layers.

    Architecture::

        Flatten(3×32×32=3072)
        → PrunableLinear(3072, 512) → ReLU → Dropout(0.3)
        → PrunableLinear(512,  256) → ReLU → Dropout(0.3)
        → PrunableLinear(256,  128) → ReLU
        → PrunableLinear(128,   10)          [logits]
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Batch of CIFAR-10 images, shape ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, 10)``.
        """
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # ---- gate inspection helpers ----------------------------------------

    def get_all_gates(self) -> torch.Tensor:
        """Concatenate sigmoid-activated gate values from every ``PrunableLinear`` layer.

        Returns
        -------
        torch.Tensor
            1-D tensor containing all gate values in the network.
        """
        gates: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(torch.sigmoid(module.gate_scores).flatten())
        return torch.cat(gates)

    def sparsity_level(self, threshold: float = SPARSITY_THRESHOLD) -> float:
        """Percentage of gate values below *threshold*.

        A higher percentage means more weights are effectively pruned.

        Parameters
        ----------
        threshold : float
            Gate values below this are considered "pruned".

        Returns
        -------
        float
            Sparsity percentage in ``[0, 100]``.
        """
        gates = self.get_all_gates()
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return 100.0 * pruned / total


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Loss Computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """L1 norm of all gate values across every PrunableLinear layer.

    Since gates = sigmoid(gate_scores) are always positive, L1 = sum.

    Parameters
    ----------
    model : SelfPruningNet
        The network whose gate values are penalised.

    Returns
    -------
    torch.Tensor
        Scalar sparsity loss.
    """
    return model.get_all_gates().sum()


def compute_total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: SelfPruningNet,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total loss = CrossEntropy + λ × SparsityLoss.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output (B, 10).
    targets : torch.Tensor
        Ground-truth class indices (B,).
    model : SelfPruningNet
        Network (for gate access).
    lam : float
        Sparsity coefficient λ.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_loss, ce_loss, sparsity_loss)``
    """
    ce_loss = F.cross_entropy(logits, targets)
    sp_loss = compute_sparsity_loss(model)
    total = ce_loss + lam * sp_loss
    return total, ce_loss, sp_loss


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Data Loaders
# ═══════════════════════════════════════════════════════════════════════════
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 train / validation / test data loaders.

    The original 50 000-image training set is split 45 000 / 5 000 for
    train / val.  Standard normalisation is applied.

    Parameters
    ----------
    batch_size : int
        Mini-batch size.
    data_dir : str
        Where to download / cache CIFAR-10.
    num_workers : int
        DataLoader workers.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True,
                                   download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False,
                                 download=True, transform=transform_test)

    # Split train into train + val
    train_size = 45_000
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lam: float,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns
    -------
    tuple[float, float]
        ``(avg_total_loss, avg_ce_loss)``
    """
    model.train()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    n_batches = 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        total_loss, ce_loss, _ = compute_total_loss(logits, targets, model, lam)
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        ce_loss_sum += ce_loss.item()
        n_batches += 1

    return total_loss_sum / n_batches, ce_loss_sum / n_batches


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    lam: float,
) -> tuple[float, float, float]:
    """Evaluate on a dataset.

    Returns
    -------
    tuple[float, float, float]
        ``(avg_loss, accuracy_pct, sparsity_pct)``
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        logits = model(images)
        loss, _, _ = compute_total_loss(logits, targets, model, lam)
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    n_batches = len(loader)
    accuracy = 100.0 * correct / total
    sparsity = model.sparsity_level()
    return loss_sum / n_batches, accuracy, sparsity


def train_model(
    lam: float,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> dict:
    """Full training run for a given λ value.

    Parameters
    ----------
    lam : float
        Sparsity coefficient.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        Summary with keys: lambda, test_accuracy, sparsity_level,
        best_val_accuracy, epoch_logs, checkpoint_path.
    """
    logger.info("══════════════════════════════════════════════")
    logger.info("Starting training — λ = %s", lam)
    logger.info("══════════════════════════════════════════════")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size)
    model = SelfPruningNet().to(DEVICE)

    # Gate scores use a 3× higher LR than weights/biases.  At init=3.0
    # the sigmoid derivative g(1-g) ≈ 0.046 is ~5× smaller than at 0.5,
    # so the higher LR ensures gates converge within 30 epochs.
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    optimizer = torch.optim.Adam([
        {"params": other_params},
        {"params": gate_params, "lr": lr * 3},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    checkpoint_path = CHECKPOINT_DIR / f"model_lambda_{lam}.pt"
    epoch_logs: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, lam)
        val_loss, val_acc, sparsity = evaluate(model, val_loader, lam)
        scheduler.step()
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "sparsity_level": round(sparsity, 2),
        }
        epoch_logs.append(log_entry)

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.2f%% | sparsity=%.2f%% | %.1fs",
            epoch, epochs, train_loss, val_loss, val_acc, sparsity, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ Saved new best checkpoint (val_acc=%.2f%%)", val_acc)

    # Test evaluation with best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    _, test_acc, test_sparsity = evaluate(model, test_loader, lam)
    logger.info("λ=%s | Test Accuracy: %.2f%% | Sparsity: %.2f%%",
                lam, test_acc, test_sparsity)

    result = {
        "lambda": lam,
        "test_accuracy": round(test_acc, 2),
        "sparsity_level": round(test_sparsity, 2),
        "best_val_accuracy": round(best_val_acc, 2),
        "epochs": epochs,
        "epoch_logs": epoch_logs,
        "checkpoint_path": str(checkpoint_path),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_gate_distribution(model: SelfPruningNet, lam: float) -> str:
    """Histogram of gate values for the given model.

    Parameters
    ----------
    model : SelfPruningNet
        Trained model.
    lam : float
        Lambda value (for title/filename).

    Returns
    -------
    str
        Path to the saved PNG.
    """
    gates = model.get_all_gates().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=100, edgecolor="black", alpha=0.75, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xlabel("Gate Value (sigmoid output)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Gate Value Distribution — λ = {lam}")
    ax.axvline(x=SPARSITY_THRESHOLD, color="red", linestyle="--",
               label=f"Pruning threshold ({SPARSITY_THRESHOLD})")
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / f"gate_distribution_lambda_{lam}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved gate distribution plot → %s", path)
    return str(path)


def plot_accuracy_vs_lambda(results: list[dict]) -> str:
    """Bar chart: test accuracy for each λ.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``train_model``.

    Returns
    -------
    str
        Path to saved PNG.
    """
    lambdas = [str(r["lambda"]) for r in results]
    accs = [r["test_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(lambdas, accs, color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black")
    ax.set_xlabel("Lambda (λ)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs. Sparsity Coefficient λ")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()

    path = PLOT_DIR / "accuracy_vs_lambda.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy vs λ plot → %s", path)
    return str(path)


def plot_sparsity_vs_lambda(results: list[dict]) -> str:
    """Bar chart: sparsity level for each λ.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``train_model``.

    Returns
    -------
    str
        Path to saved PNG.
    """
    lambdas = [str(r["lambda"]) for r in results]
    sparsity = [r["sparsity_level"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(lambdas, sparsity, color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black")
    ax.set_xlabel("Lambda (λ)")
    ax.set_ylabel("Sparsity Level (%)")
    ax.set_title("Sparsity Level vs. Sparsity Coefficient λ")
    ax.set_ylim(0, 100)
    for bar, sp in zip(bars, sparsity):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{sp:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()

    path = PLOT_DIR / "sparsity_vs_lambda.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved sparsity vs λ plot → %s", path)
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — run all experiments
# ═══════════════════════════════════════════════════════════════════════════

LAMBDA_VALUES: list[float] = [1e-6, 1e-5, 1e-4]


def main() -> None:
    """Run training for all λ values, evaluate, plot, and print summary."""
    logger.info("Device: %s", DEVICE)
    logger.info("Lambda values to evaluate: %s", LAMBDA_VALUES)

    all_results: list[dict] = []
    for lam in LAMBDA_VALUES:
        result = train_model(lam=lam, epochs=30)
        all_results.append(result)

    # ── Print results table ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)':<18}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:<18.2f} {r['sparsity_level']:<18.2f}")
    print("=" * 60)

    # ── Gate distribution plot for best model ────────────────────────────
    best = max(all_results, key=lambda r: r["test_accuracy"])
    best_model = SelfPruningNet().to(DEVICE)
    best_model.load_state_dict(
        torch.load(best["checkpoint_path"], map_location=DEVICE, weights_only=True)
    )
    plot_gate_distribution(best_model, best["lambda"])

    # ── Comparison plots ─────────────────────────────────────────────────
    plot_accuracy_vs_lambda(all_results)
    plot_sparsity_vs_lambda(all_results)

    logger.info("All experiments complete. Checkpoints → %s, Plots → %s",
                CHECKPOINT_DIR, PLOT_DIR)


if __name__ == "__main__":
    main()
