#!/usr/bin/env python3
"""
Self-Pruning Neural Network on CIFAR-10
========================================
A feed-forward neural network that learns to prune itself during training.

Each weight in every fully-connected layer is paired with a learnable scalar
gate parameter.  Passing the gate through a sigmoid confines it to (0, 1); an
L1 sparsity penalty on all gate values drives unimportant connections toward
exactly zero, effectively removing them from the network without any
post-training pruning step.

Pipeline followed  in this implementation
--------
1. PrunableLinear  — custom linear layer with per-weight sigmoid gates.
2. SelfPruningNet  — VGG-style CNN backbone + prunable FC classification head.
3. Loss            — Cross-Entropy + λ × L1-gate penalty.
4. Data            — CIFAR-10 with augmentation; clean train/val split.
5. Training        — Adam + warmup + cosine annealing; best-checkpoint saving.
6. Plotting        — Gate-value distribution histogram for the optimal model.

Usage
-----
    python self_pruning_network.py
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # use non-interactive backend (server / Colab safe)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Paths & logging

PROJECT_ROOT   = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOT_DIR       = PROJECT_ROOT / "plots"
CHECKPOINT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("self_pruning_network")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# A gate whose sigmoid value falls below this threshold is considered pruned.
# Value 1e-2 is taken here as threshold
SPARSITY_THRESHOLD: float = 1e-2

# Per-channel mean and std of CIFAR-10 (pre-computed over the training set).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Device: GPU > CPU, chosen automatically at runtime.
DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available()          else
    "cpu"
)

# Three λ values that demonstrate the full sparsity-accuracy trade-off:
#   1e-6  → mild penalty    → low sparsity,  near-peak accuracy
#   1e-5  → moderate penalty → high sparsity, minimal accuracy drop (sweet spot)
#   1e-4  → strong penalty  → very high sparsity, small accuracy cost
LAMBDA_VALUES: list[float] = [1e-6, 1e-5, 1e-4]


# PART 1 — Prunable Linear Layer

class PrunableLinear(nn.Module):
    """Fully-connected layer with a learnable per-weight gate parameter.

    For every weight w_ij the layer maintains a gate score s_ij (same shape
    as the weight matrix).  During the forward pass:

        gate_ij        = sigmoid(s_ij)           ∈ (0, 1)
        pruned_weight  = w_ij × gate_ij          element-wise masking
        output         = F.linear(x, pruned_weight, bias)

    Gate scores are initialised to 3.0 so that sigmoid(3.0) ≈ 0.953: all
    gates start nearly open.  The L1 sparsity loss provides a constant
    downward gradient on each gate, while the cross-entropy loss provides
    upward gradient only for connections that genuinely help classification.
    Connections that are unhelpful receive no CE defence, so their gates
    collapse to zero — effectively pruning those weights.

    Gradients flow through both `weight` and `gate_scores` automatically via
    PyTorch autograd; no custom backward pass is required.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features (neurons).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Learnable weight and bias — standard for any linear layer.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # Gate scores — same shape as weight so every weight has its own gate.
        # Registered as a Parameter so the optimizer updates them alongside
        # the weights during each backward pass.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming-uniform weights, zero bias, constant gate scores = 3.0."""
        # Kaiming uniform: the standard PyTorch default for linear layers.
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)
        # sigmoid(3.0) ≈ 0.953: gates open at ~95% so the network begins
        # training with almost full capacity before sparsity pressure builds.
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gated linear transformation.

        Steps
        -----
        1. Compute gates = sigmoid(gate_scores)  — squashes scores to (0, 1).
        2. Multiply weights element-wise by gates — gates near 0 silence the
           corresponding weight entirely.
        3. Apply the standard affine transformation via F.linear.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch_size, in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch_size, out_features)``.
        """
        gates          = torch.sigmoid(self.gate_scores)  # ∈ (0, 1)
        pruned_weights = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weights, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"



# PART 2 — Self-Pruning Network

class SelfPruningNet(nn.Module):
    """VGG-style CNN backbone with a prunable fully-connected classification head.

    The CNN backbone uses standard Conv2d + BatchNorm2d layers to extract
    spatial features from 32×32 CIFAR-10 images.  It is not pruned.

    The FC head consists of three PrunableLinear layers whose gate parameters
    are jointly optimised with the classification loss.  Sparsity is driven
    exclusively in these layers.

    Architecture
    ------------
    Backbone (not prunable):
        Conv(3→64)→BN→ReLU   → Conv(64→64)→BN→ReLU   → MaxPool  [32→16]
        Conv(64→128)→BN→ReLU → Conv(128→128)→BN→ReLU → MaxPool  [16→8]
        Conv(128→256)→BN→ReLU → Conv(256→256)→BN→ReLU → AvgPool [8→1]
        Flatten → 256-dim feature vector

    FC Head (prunable):
        PrunableLinear(256→512) → BatchNorm1d → ReLU → Dropout(p)
        PrunableLinear(512→256) → BatchNorm1d → ReLU → Dropout(p)
        PrunableLinear(256→10)  → logits

    Parameters
    ----------
    dropout_p : float
        Dropout probability applied after each hidden prunable layer.
        Default 0.3 provides moderate regularisation without over-dropping.
    """

    def __init__(self, dropout_p: float = 0.3) -> None:
        super().__init__()

        def _conv_block(in_ch: int, out_ch: int) -> list:
            """Return [Conv2d, BatchNorm2d, ReLU] as a list for nn.Sequential unpacking."""
            return [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),   # bias=False because BN has its own learnable shift
                nn.ReLU(inplace=True),
            ]

        # --- CNN Backbone ---------------------------------------------------
        self.backbone = nn.Sequential(
            # Block 1: 32×32 → 16×16
            *_conv_block(3,   64),  *_conv_block(64,  64),  nn.MaxPool2d(2, 2),
            # Block 2: 16×16 → 8×8
            *_conv_block(64, 128),  *_conv_block(128, 128), nn.MaxPool2d(2, 2),
            # Block 3: 8×8 → 1×1  (AdaptiveAvgPool removes dependency on input size)
            *_conv_block(128, 256), *_conv_block(256, 256), nn.AdaptiveAvgPool2d((1, 1)),
        )

        # --- Prunable FC Head -----------------------------------------------
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            PrunableLinear(256, 512),   # hidden layer 1
            PrunableLinear(512, 256),   # hidden layer 2
            PrunableLinear(256,  10),   # output layer (10 CIFAR-10 classes)
        ])
        # BatchNorm1d after each hidden PrunableLinear stabilises activations.
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(256),
        ])
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run input through backbone then prunable FC head.

        Hidden layers receive BatchNorm → ReLU → Dropout.
        The output layer returns raw logits (no activation).

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(B, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, 10)``.
        """
        x = self.backbone(x)   # (B, 256, 1, 1)
        x = self.flatten(x)    # (B, 256)

        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)                    # gated linear transform
            if i < last_idx:               # hidden layers only
                x = self.bn_layers[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        return x                           # raw logits

    # -----------------------------------------------------------------------
    # Gate inspection helpers
    # -----------------------------------------------------------------------

    def get_all_gates(self) -> torch.Tensor:
        """Return a flat 1-D tensor of every sigmoid-activated gate value.

        Concatenates gates from all three PrunableLinear layers.  Used both
        for the sparsity loss and for diagnostic plotting.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(total_gate_count,)`` with values in (0, 1).
        """
        return torch.cat([
            torch.sigmoid(layer.gate_scores).flatten()
            for layer in self.layers
        ])

    def sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity penalty — the sum of all gate values across the FC head.

        Since sigmoid outputs are strictly positive, the L1 norm equals the
        plain sum.  This scalar is multiplied by λ and added to the CE loss
        to form the total training objective.

        Returns
        -------
        torch.Tensor
            Scalar sparsity loss term.
        """
        return self.get_all_gates().sum()

    def sparsity_level(self, threshold: float = SPARSITY_THRESHOLD) -> float:
        """Compute the fraction of gates that have been effectively pruned.

        A gate is considered pruned when its sigmoid value falls below
        `threshold` (default 1e-2 per the assignment specification).

        Parameters
        ----------
        threshold : float
            Gates below this value count as pruned.

        Returns
        -------
        float
            Sparsity percentage in [0, 100].
        """
        gates  = self.get_all_gates()
        pruned = (gates < threshold).sum().item()
        return 100.0 * pruned / gates.numel()



# PART 3 — Loss Function


def compute_total_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    model:   SelfPruningNet,
    lam:     float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the combined classification + sparsity loss.

    Total loss formula:
        L_total = CrossEntropy(logits, targets) + λ × Σ sigmoid(gate_scores)

    The cross-entropy term drives accurate classification; the λ-scaled L1
    gate term drives gates toward zero.  A higher λ produces a sparser network
    at the potential cost of accuracy.

    Label smoothing (ε = 0.1) is applied to the cross-entropy term to prevent
    the model from becoming over-confident, which typically improves
    generalisation by ~0.5–1% on CIFAR-10.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model output, shape ``(B, num_classes)``.
    targets : torch.Tensor
        Ground-truth class indices, shape ``(B,)``.
    model : SelfPruningNet
        The network (used to read the current gate values).
    lam : float
        Sparsity coefficient λ.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_loss, ce_loss, sparsity_loss)`` — all scalar tensors.
    """
    ce_loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
    sp_loss = model.sparsity_loss()
    total   = ce_loss + lam * sp_loss
    return total, ce_loss, sp_loss


# PART 4 — Data Loaders

def get_dataloaders(
    batch_size:  int = 128,
    data_dir:    str = "./data",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build CIFAR-10 train, validation, and test data loaders.

    The 50 000-image training set is split deterministically (seed = 42) into
    45 000 training images and 5 000 validation images.

    Important: two separate dataset objects are created — one with training
    augmentations and one without.  The augmented object is used for the
    training split; the plain object is used for the validation split.  This
    prevents augmented images from appearing in validation, which would make
    val_acc an unreliable metric for early stopping.

    Training augmentations applied:
        • RandomHorizontalFlip  — horizontal symmetry invariance
        • RandomCrop(32, pad=4) — translation invariance
        • ColorJitter           — robustness to lighting variation
        • Normalise             — zero-mean, unit-variance per channel

    Parameters
    ----------
    batch_size : int
        Number of samples per mini-batch.
    data_dir : str
        Directory where CIFAR-10 is downloaded / cached.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load the training data twice: once augmented (for training subsets)
    # and once plain (for the validation subset).
    full_train_aug   = datasets.CIFAR10(data_dir, train=True,  download=True,
                                         transform=transform_train)
    full_train_plain = datasets.CIFAR10(data_dir, train=True,  download=True,
                                         transform=transform_test)
    test_set         = datasets.CIFAR10(data_dir, train=False, download=True,
                                         transform=transform_test)

    # Deterministic shuffle then split: first 45 000 → train, rest → val.
    train_size  = 45_000
    all_indices = torch.randperm(
        len(full_train_aug),
        generator=torch.Generator().manual_seed(42),
    ).tolist()
    train_set = torch.utils.data.Subset(full_train_aug,   all_indices[:train_size])
    val_set   = torch.utils.data.Subset(full_train_plain, all_indices[train_size:])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


 
# PART 5 — Training & Evaluation

def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    lam:       float,
) -> tuple[float, float]:
    """Run one full pass over the training set and update all parameters.

    Gradient clipping (max-norm 1.0) is applied before each optimiser step
    to prevent occasional gradient spikes that can corrupt BatchNorm running
    statistics in deep networks with gated weights.

    Parameters
    ----------
    model : SelfPruningNet
        Network in training mode.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimiser instance (Adam with separate param groups for gates).
    lam : float
        Sparsity coefficient λ.

    Returns
    -------
    tuple[float, float]
        ``(avg_total_loss, avg_ce_loss)`` averaged over all mini-batches.
    """
    model.train()
    total_loss_sum = ce_loss_sum = 0.0
    n_batches = 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        total_loss, ce_loss, _ = compute_total_loss(logits, targets, model, lam)
        total_loss.backward()

        # Clip the global gradient norm to 1.0 before the weight update.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += total_loss.item()
        ce_loss_sum    += ce_loss.item()
        n_batches      += 1

    return total_loss_sum / n_batches, ce_loss_sum / n_batches


@torch.no_grad()
def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    lam:    float,
) -> tuple[float, float, float]:
    """Evaluate the model on a dataset without updating parameters.

    Parameters
    ----------
    model : SelfPruningNet
        Network (set to eval mode internally).
    loader : DataLoader
        Validation or test data loader.
    lam : float
        Sparsity coefficient λ (used to compute the total loss for logging).

    Returns
    -------
    tuple[float, float, float]
        ``(avg_loss, accuracy_pct, sparsity_pct)``
    """
    model.eval()
    loss_sum = correct = total = 0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        logits = model(images)
        loss, _, _ = compute_total_loss(logits, targets, model, lam)

        loss_sum += loss.item()
        preds     = logits.argmax(dim=1)
        correct  += (preds == targets).sum().item()
        total    += targets.size(0)

    accuracy = 100.0 * correct / total
    sparsity = model.sparsity_level()
    return loss_sum / len(loader), accuracy, sparsity


def train_model(
    lam:        float,
    epochs:     int   = 60,
    lr:         float = 1e-3,
    batch_size: int   = 128,
) -> dict:
    """Run a full training experiment for a single λ value.

    Optimizer strategy
    ------------------
    Two parameter groups are created:
    • Backbone / FC weights & biases — Adam with lr=1e-3, weight_decay=5e-4.
      L2 weight decay regularises the backbone weights without interfering
      with the gate L1 mechanism.
    • Gate scores — Adam with lr=3e-3 (3× higher).
      At initialisation, sigmoid(3.0) ≈ 0.953 places gates in the saturated
      region where the sigmoid's Jacobian g(1-g) ≈ 0.046 is ~5× smaller than
      at g=0.5.  The higher gate LR compensates for this reduced gradient
      magnitude so gates move meaningfully within the 60-epoch budget.

    Scheduler strategy
    ------------------
    5-epoch linear warmup (lr: 0.1× → 1.0×) followed by cosine annealing to
    eta_min=1e-5.  Warmup stabilises BatchNorm running statistics before the
    full learning rate is applied.

    Parameters
    ----------
    lam : float
        Sparsity coefficient λ.
    epochs : int
        Total number of training epochs.
    lr : float
        Base learning rate for Adam.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        Keys: lambda, test_accuracy, sparsity_level, best_val_accuracy,
        epochs, epoch_logs, checkpoint_path.
    """
    logger.info("══════════════════════════════════════════════")
    logger.info("Starting training  |  λ = %s  |  device = %s", lam, DEVICE)
    logger.info("══════════════════════════════════════════════")

    data_dir = str(PROJECT_ROOT / "data")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size, data_dir)
    model = SelfPruningNet().to(DEVICE)

    # Separate parameter groups: gates get a higher LR to overcome the
    # small sigmoid Jacobian at initialisation (sigmoid is saturated at 3.0).
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": 5e-4},  # L2 on weights
        {"params": gate_params,  "lr": lr * 3},           # higher LR for gates
    ], lr=lr)

    # 5-epoch warmup then cosine decay to 1e-5.
    _WARMUP = 5
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=_WARMUP,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - _WARMUP, eta_min=1e-5,
            ),
        ],
        milestones=[_WARMUP],
    )

    best_val_acc    = 0.0
    checkpoint_path = CHECKPOINT_DIR / f"model_lambda_{lam}.pt"
    epoch_logs: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, lam)
        val_loss, val_acc, sparsity = evaluate(model, val_loader, lam)
        scheduler.step()

        epoch_logs.append({
            "epoch":          epoch,
            "train_loss":     round(train_loss, 4),
            "val_loss":       round(val_loss,   4),
            "val_accuracy":   round(val_acc,    2),
            "sparsity_level": round(sparsity,   2),
        })

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.2f%% | sparsity=%.2f%% | %.1fs",
            epoch, epochs, train_loss, val_loss, val_acc, sparsity,
            time.time() - t0,
        )

        # Persist the model whenever validation accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ↳ New best checkpoint saved (val_acc=%.2f%%)", val_acc)

    # Reload best checkpoint and measure final test-set performance.
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    )
    _, test_acc, test_sparsity = evaluate(model, test_loader, lam)
    logger.info("λ=%s | Test Accuracy: %.2f%% | Sparsity: %.2f%%",
                lam, test_acc, test_sparsity)

    return {
        "lambda":            lam,
        "test_accuracy":     round(test_acc,      2),
        "sparsity_level":    round(test_sparsity, 2),
        "best_val_accuracy": round(best_val_acc,  2),
        "epochs":            epochs,
        "epoch_logs":        epoch_logs,
        "checkpoint_path":   str(checkpoint_path),
    }


# PART 6 — Plotting

def plot_gate_distribution(model: SelfPruningNet, lam: float) -> str:
    """To plot a histogram of all sigmoid gate values for the given model.

    The y-axis is log-scaled because the pruned population (gates near 0)
    and the surviving population (gates near 1) differ by several orders of
    magnitude.  A linear scale would render the surviving cluster invisible.

    A successful self-pruning result shows a clear bimodal distribution:
        • A large spike at 0  — pruned connections (gate < SPARSITY_THRESHOLD)
        • A cluster near 1   — surviving, task-relevant connections

    The red dashed vertical line marks SPARSITY_THRESHOLD (1e-2), the
    boundary used to count pruned vs. active gates.

    Parameters
    ----------
    model : SelfPruningNet
        Trained model whose gate distribution will be visualised.
    lam : float
        The λ value used during training (for the plot title and filename).

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    gates = model.get_all_gates().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=100, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("Gate Value  (sigmoid output)")
    ax.set_ylabel("Count  (log scale)")
    ax.set_title(f"Gate Value Distribution  —  λ = {lam}")
    ax.axvline(
        x=SPARSITY_THRESHOLD, color="red", linestyle="--",
        label=f"Pruning threshold  ({SPARSITY_THRESHOLD})",
    )
    ax.legend()
    fig.tight_layout()

    path = PLOT_DIR / f"gate_distribution_lambda_{lam}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved gate distribution plot → %s", path)
    return str(path)


# MAIN — Orchestrate all experiments

def _print_results_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout after all experiments finish."""
    header = f"{'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)':<18}"
    print("\n" + "=" * 60)
    print(header)
    print("-" * 60)
    for r in results:
        print(f"{r['lambda']:<12} {r['test_accuracy']:<18.2f} {r['sparsity_level']:<18.2f}")
    print("=" * 60)


def main() -> None:
    """Train for all λ values, evaluate on the test set, and plot results.

    Optimal model selection
    -----------------------
    After all runs, the "optimal" model is defined as the one with the
    highest sparsity level among all models whose test accuracy is within
    1% of the best observed accuracy.  This selects the most compressed
    model that retains near-peak predictive performance — the practical
    goal of dynamic pruning.
    """
    logger.info("Device         : %s", DEVICE)
    logger.info("Lambda values  : %s", LAMBDA_VALUES)

    # --- Run one experiment per λ value ------------------------------------
    all_results: list[dict] = []
    for lam in LAMBDA_VALUES:
        all_results.append(train_model(lam=lam))

    # --- Print summary table -----------------------------------------------
    _print_results_table(all_results)

    # --- Select optimal model and plot its gate distribution ---------------
    # Optimal = most sparse model whose accuracy is within 1% of peak.
    peak_acc   = max(r["test_accuracy"] for r in all_results)
    candidates = [r for r in all_results if peak_acc - r["test_accuracy"] <= 1.0]
    optimal    = max(candidates, key=lambda r: r["sparsity_level"])

    logger.info(
        "Optimal model  : λ=%s | test_acc=%.2f%% | sparsity=%.2f%%",
        optimal["lambda"], optimal["test_accuracy"], optimal["sparsity_level"],
    )

    best_model = SelfPruningNet().to(DEVICE)
    best_model.load_state_dict(
        torch.load(optimal["checkpoint_path"], map_location=DEVICE, weights_only=True)
    )
    plot_gate_distribution(best_model, optimal["lambda"])

    logger.info(
        "All experiments complete.  Checkpoints → %s  |  Plots → %s",
        CHECKPOINT_DIR, PLOT_DIR,
    )


if __name__ == "__main__":
    main()
