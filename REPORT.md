# Report on Self-Pruning Neural Network 

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity?

### The Role of Sigmoid

Each weight $w_{ij}$ in a `PrunableLinear` layer is associated with a learnable gate score $s_{ij}$. The gate is computed as:

$$g_{ij} = \sigma(s_{ij}) = \frac{1}{1 + e^{-s_{ij}}}$$

This constrains every gate to the range $(0, 1)$. When $g_{ij} \to 0$, the corresponding weight is effectively removed from the network; when $g_{ij} \to 1$, it is fully active.

### Why L1 (Not L2) Drives Values to Exactly Zero?

The total loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i,j} g_{ij}$$

Since all gates are positive after the sigmoid transform, the L1 norm simplifies to the sum.

The key difference between L1 and L2 regularisation lies in their **gradient behaviour near zero**:

| Norm | Penalty | Gradient w.r.t. $g$ (for $g > 0$) |
| ---- | ------- | --------------------------------- |
| L1   | $\|g\|$ | $+1$ (constant)                   |
| L2   | $g^2$   | $2g$ (vanishes as $g \to 0$)      |

The table above compares penalty gradients with respect to $g$ directly. With L2, the gradient $2g$ shrinks toward zero as $g \to 0$, providing **diminishing pressure** on small gates. With L1, the gradient with respect to $g$ is a constant $+1$ — every active gate receives **equal penalty regardless of its magnitude**.

However, the optimiser does not update $g$ directly; it updates the underlying gate score $s$. The actual gradient that reaches $s$ passes through the sigmoid derivative and vanishes as $g \to 0$ (see **Gradient Flow** below). The reason L1 still drives gates to zero is because L1 penalises all active gates equally rather than easing off on small ones the way L2 does. Combined with the sigmoid squashing, once a gate begins drifting toward 0, the effective weight $w_{ij} \cdot g_{ij} \to 0$, so the cross-entropy loss stops defending that connection. With no sufficient CE gradient to resist the L1 pressure, the gate collapses fully thereby producing genuinely **sparse** solutions.

### Gradient Flow Through the Gate

The gradient of $g_{ij}$ with respect to the gate score $s_{ij}$ is:

$$\frac{\partial g_{ij}}{\partial s_{ij}} = g_{ij}(1 - g_{ij})$$

Applying the chain rule through both the L1 penalty and the sigmoid gives the full derivation:

$$\frac{\partial \mathcal{L}_{\text{sparsity}}}{\partial s_{ij}} = \lambda \cdot \frac{\partial \sum_{i,j} g_{ij}}{\partial g_{ij}} \cdot \frac{\partial g_{ij}}{\partial s_{ij}} = \lambda \cdot \underbrace{1}_{\text{L1 gradient}} \cdot \underbrace{g_{ij}(1 - g_{ij})}_{\text{sigmoid Jacobian}}$$

The $\cdot 1 \cdot$ term is the key: it is the constant L1 gradient with respect to $g$, which distinguishes L1 from L2 (where the corresponding term would be $2g_{ij}$, vanishing as $g \to 0$). The full gradient that actually reaches $s_{ij}$ is therefore:

$$\frac{\partial \mathcal{L}_{\text{sparsity}}}{\partial s_{ij}} = \lambda \cdot g_{ij}(1 - g_{ij})$$

When a gate is near 0, this gradient is small: already-pruned gates stay pruned because the sparsity loss no longer pushes them further. 
When a gate is near 0.5, the gradient is maximal: the optimiser must decide whether the CE benefit of keeping the connection outweighs the L1 penalty. Gates whose connections are redundant receive little CE support and are gradually driven to zero.

---

## 2. Results Summary

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 1e-06  | **91.85**         | 38.80              |
| 1e-05  | 91.71             | **87.92**          |
| 0.0001 | 91.44             | 96.53              |

Best model (optimal trade-off): **λ = 1e-05** — 87.92% of FC weights pruned while retaining 91.71% test accuracy.

---

## 3. Analysis of the λ Tradeoff

- **λ = 1e-06 (low):** The sparsity penalty is mild relative to the cross-entropy loss. Only 38.80% of gates are pruned; most connections are retained and the network behaves close to an unregularised model. Test accuracy peaks at **91.85%**.

- **λ = 1e-05 (medium — optimal):** This is the sweet spot. The network prunes **87.92%** of its FC-layer connections while maintaining **91.71% test accuracy** — a drop of only 0.14% vs. the low-λ baseline. The L1 penalty is strong enough to drive aggressive sparsity without meaningfully harming task performance. The gate distribution for this model shows a clear bimodal shape: a large spike near 0 and a distinct surviving cluster near 1.

- **λ = 0.0001 (high):** The sparsity penalty dominates. Over **96.53%** of gates are driven below the threshold, leaving an almost entirely sparse FC head. Despite this extreme compression, test accuracy remains at **91.44%** — only 0.41% below the unpruned baseline — confirming that the CNN backbone carries the representational capacity and the vast majority of FC weights are genuinely redundant.

**Key insight:** Accuracy is remarkably stable across the full sparsity range (91.85% → 91.44%, a spread of just 0.41%) even as sparsity jumps from 38.80% to 96.53%. This demonstrates that the VGG-style CNN backbone learns robust spatial features, and the prunable FC head can be compressed aggressively with negligible accuracy cost.

**Best balance:** λ = 1e-05 — 87.92% sparsity with 91.71% accuracy.

---

## 4. Gate Value Distribution

The gate distribution histogram for the optimal model (λ = 1e-05) is plotted on a **log y-scale**, which is necessary because the pruned population (gates near 0) and the surviving population (gates near 1) differ by several orders of magnitude — a linear scale would make the surviving cluster invisible.

Key observations:

- **Large spike at 0:** Approximately **87.92%** of gates fall below the pruning threshold (1e-02). These connections have been effectively zeroed out by the L1 penalty — the network has learned they are redundant for CIFAR-10 classification.
- **A distinct cluster near 0.9–1.0:** The surviving connections that the cross-entropy gradient defended against L1 pressure. These are the most informative FC connections for the classification task.
- **Near-empty middle region (0.02–0.8):** Very few gates occupy intermediate values. This bimodal “on/off” distribution is the hallmark of successful L1-driven sparsification, gates commit decisively to either being pruned or surviving.

The plot demonstrates **a large spike at 0 and a separate cluster of values away from 0**, with the red dashed pruning threshold (1e-02) cleanly separating the two populations.

---

## 5. Network Architecture

```
Input Image (3×32×32)
        │
┌───────────────────────────────────────────────┐
│  CNN Backbone (standard, non-prunable)             │
│                                                    │
│  Conv(3→64)→BN→ReLU → Conv(64→64)→BN→ReLU         │
│  MaxPool(2×2)                       [32→16]       │
│  Conv(64→128)→BN→ReLU → Conv(128→128)→BN→ReLU    │
│  MaxPool(2×2)                       [16→8]        │
│  Conv(128→256)→BN→ReLU → Conv(256→256)→BN→ReLU   │
│  AdaptiveAvgPool(1×1)               [8→1]         │
│  Flatten                            → 256           │
└───────────────────────────────────────────────┘
        │
┌───────────────────────────────────────────────┐
│  Self-Pruning FC Head (prunable)                   │
│                                                    │
│  PrunableLinear(256→512)                           │
│  └─ weight × sigmoid(gate_scores) → F.linear       │
│  BatchNorm1d(512) → ReLU → Dropout(0.3)            │
│                                                    │
│  PrunableLinear(512→256)                           │
│  └─ weight × sigmoid(gate_scores) → F.linear       │
│  BatchNorm1d(256) → ReLU → Dropout(0.3)            │
│                                                    │
│  PrunableLinear(256→10)  [output logits]           │
│  └─ weight × sigmoid(gate_scores) → F.linear       │
└───────────────────────────────────────────────┘
        │
   10 class logits (CIFAR-10)

Gate Mechanism (per PrunableLinear layer):
  gate_scores  ───→  sigmoid()  ───→  gates ∈ (0, 1)
  pruned_weights = weight × gates          (element-wise)
  output         = F.linear(x, pruned_weights, bias)

Loss:
  L_total = CrossEntropy(logits, y) + λ × Σ gates
```

The CNN backbone uses standard `nn.Conv2d` + `nn.BatchNorm2d` layers (not prunable). All pruning occurs exclusively in the three `PrunableLinear` layers of the FC head, which together contain the gate parameters that the L1 penalty acts on.

---

## 6. Plots

| Plot | File |
| ---- | ---- |
| Gate distribution for optimal model (λ = 1e-05) | `plots/gate_distribution_lambda_1e-05.png` |

