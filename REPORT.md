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

However, the optimiser does not update $g$ directly; it updates the underlying gate score $s$. The actual gradient that reaches $s$ passes through the sigmoid derivative and vanishes as $g \to 0$ (see **Gradient Flow** below). The reason L1 still drives gates to zero is more subtle: L1 penalises all active gates equally rather than easing off on small ones the way L2 does. Combined with the sigmoid squashing, once a gate begins drifting toward 0, the effective weight $w_{ij} \cdot g_{ij} \to 0$, so the cross-entropy loss stops defending that connection. With no sufficient CE gradient to resist the L1 pressure, the gate collapses fully — producing genuinely **sparse** solutions.

### Gradient Flow Through the Gate

The gradient of $g_{ij}$ with respect to the gate score $s_{ij}$ is:

$$\frac{\partial g_{ij}}{\partial s_{ij}} = g_{ij}(1 - g_{ij})$$

Combined with the L1 penalty gradient:

$$\frac{\partial \mathcal{L}_{\text{sparsity}}}{\partial s_{ij}} = \lambda \cdot g_{ij}(1 - g_{ij})$$

When a gate is near 0, this gradient is small — already-pruned gates stay pruned because the sparsity loss no longer pushes them further. When a gate is near 0.5, the gradient is maximal: the optimiser must decide whether the CE benefit of keeping the connection outweighs the L1 penalty. Gates whose connections are redundant receive little CE support and are gradually driven to zero.

---

## 2. Results Summary

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 1e-06  | 51.16             | 15.63              |
| 1e-05  | 53.73             | 81.06              |
| 0.0001 | 51.09             | 99.14              |

---

## 3. Analysis of the λ Tradeoff

- **λ = 1e-06 (low):** The sparsity penalty is weak. Only 15.63% of gates are pruned; the network retains most connections and behaves close to a standard feed-forward network. Accuracy is 51.16%.

- **λ = 1e-05 (medium):** Shows well-balanced tradeoff. The network prunes **81.06%** of its connections while maintaining the highest test accuracy of **53.73%**. The L1 penalty is strong enough to drive aggressive sparsity without significantly harming task performance. This is the **best balance**.

- **λ = 0.0001 (high):** The sparsity penalty dominates. Over **99%** of gates are driven below the threshold, producing an almost entirely sparse network. Despite this extreme compression, test accuracy remains at 51.09%, remarkably close to the low-λ result confirming that the vast majority of weights are genuinely redundant for CIFAR-10.

**Best balance:** λ = 1e-05 offers the most practical tradeoff — it achieves 81.06% sparsity (dramatic compression) while achieving the highest test accuracy of 53.73%.

---

## 4. Gate Value Distribution

The gate distribution histogram for the best model (λ = 1e-05) is plotted on a **log y-scale**, which is necessary because the pruned population (~1,000,000 gates near 0) and the surviving population (~1,000–2,000 gates near 0.9–1.0) differ by three orders of magnitude — a linear scale would make the second cluster invisible.

Key observations:

- **Large spike at 0**, approximately 81% of gates are below the pruning threshold (5e-2). These connections have been effectively zeroed out by the L1 penalty.
- **A second cluster near 0.8–0.95** visible as a rising plateau on the right side of the log-scale plot. These are the connections the network preserved; they retain high gate values because their CE gradient outweighed the L1 pressure.
- **Decaying middle region (0.05–0.7)** gates that were partially penalised but not yet fully saturated in either direction within 30 epochs.

The distribution shows the two populations , a large group near 0 (pruned) and a smaller but distinct group away from 0 (surviving). The sparsity threshold is set at 5e-02.

---

## 5. Network Architecture

```
Input Image (3×32×32)
        │
   ┌────▼────┐
   │ Flatten  │  → 3072
   └────┬────┘
        │
   ┌────▼──────────────┐
   │ PrunableLinear     │  3072 → 512
   │ (weight × gates)  │
   └────┬──────────────┘
        │ ReLU → Dropout(0.3)
   ┌────▼──────────────┐
   │ PrunableLinear     │  512 → 256
   │ (weight × gates)  │
   └────┬──────────────┘
        │ ReLU → Dropout(0.3)
   ┌────▼──────────────┐
   │ PrunableLinear     │  256 → 128
   │ (weight × gates)  │
   └────┬──────────────┘
        │ ReLU
   ┌────▼──────────────┐
   │ PrunableLinear     │  128 → 10
   │ (weight × gates)  │
   └────┬──────────────┘
        │
   ┌────▼────┐
   │ Logits  │  → 10 classes (CIFAR-10)
   └─────────┘

Gate Mechanism (per PrunableLinear layer):
  gate_scores ──→ sigmoid() ──→ gates ∈ (0,1)
  output = F.linear(x, weight * gates, bias)

Loss:
  L_total = CrossEntropy(logits, y) + λ × Σ gates
```

---

## 6. Plots

| Plot                       | File                                   |
| -------------------------- | -------------------------------------- |
| Gate distribution (best λ) | `plots/gate_distribution_lambda_1e-05.png` |
| Accuracy vs λ              | `plots/accuracy_vs_lambda.png`         |
| Sparsity vs λ              | `plots/sparsity_vs_lambda.png`         |
