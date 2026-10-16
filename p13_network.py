"""
p13_network.py - Multi-Task Learning Neural Network on Synthetic Problem P13(b)
==============================================================================
NumPy-only multi-task network:
- Synthetic dataset: n=200 samples, d=5 input dimensions
- Shared representation body: 1 hidden layer (8 units, tanh activation)
- Task 1 head: Binary classification (sigmoid activation, Binary Cross-Entropy loss)
- Task 2 head: Regression (linear activation, Mean Squared Error loss)
- Hand-coded analytical forward and backward passes
- Rigorous finite-difference gradient check (max relative error < 1e-5)
- Training comparison: Equal Weights vs MGDA vs GradNorm
- Imbalanced gradient scenario: Task 2 gradients scaled 100x
- Outputs: results/data/task_d5_network.csv, results/figures/task_d5_network.png
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any


# ==============================================================================
# 1. SYNTHETIC DATA GENERATION
# ==============================================================================

def generate_synthetic_data(n: int = 200, d: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic multi-task data with shared underlying latent factors."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    
    # Ground truth weights
    w_shared = rng.standard_normal((d, 3))
    latent = np.tanh(X @ w_shared)
    
    # Task 1: Classification target
    w_c = rng.standard_normal((3, 1))
    logits = latent @ w_c + rng.normal(0, 0.1, size=(n, 1))
    prob = 1.0 / (1.0 + np.exp(-logits))
    y1 = (prob > 0.5).astype(float)
    
    # Task 2: Regression target
    w_r = rng.standard_normal((3, 1))
    y2 = latent @ w_r + rng.normal(0, 0.1, size=(n, 1))
    
    return X, y1, y2


# ==============================================================================
# 2. MULTI-TASK NEURAL NETWORK
# ==============================================================================

class MultiTaskNet:
    """Multi-task neural network with shared body and task-specific heads."""
    def __init__(self, d_in: int = 5, d_hidden: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Shared layer: (d_in, d_hidden)
        self.W_shared = rng.standard_normal((d_in, d_hidden)) * np.sqrt(2.0 / (d_in + d_hidden))
        self.b_shared = np.zeros((1, d_hidden))
        
        # Head 1 (Classification): (d_hidden, 1)
        self.W1 = rng.standard_normal((d_hidden, 1)) * np.sqrt(2.0 / (d_hidden + 1))
        self.b1 = np.zeros((1, 1))
        
        # Head 2 (Regression): (d_hidden, 1)
        self.W2 = rng.standard_normal((d_hidden, 1)) * np.sqrt(2.0 / (d_hidden + 1))
        self.b2 = np.zeros((1, 1))

    def get_params_dict(self) -> Dict[str, np.ndarray]:
        return {
            'W_shared': self.W_shared, 'b_shared': self.b_shared,
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2
        }

    def set_params_dict(self, params: Dict[str, np.ndarray]):
        self.W_shared = params['W_shared'].copy()
        self.b_shared = params['b_shared'].copy()
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass computing shared representation and head outputs."""
        # Shared body with tanh
        u = X @ self.W_shared + self.b_shared
        h = np.tanh(u)
        
        # Head 1: Binary cross-entropy with sigmoid
        z1 = h @ self.W1 + self.b1
        y1_pred = 1.0 / (1.0 + np.exp(-np.clip(z1, -30.0, 30.0)))
        
        # Head 2: Regression with linear activation
        y2_pred = h @ self.W2 + self.b2
        
        return h, y1_pred, y2_pred

    def compute_losses(self, X: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       task2_scale: float = 1.0) -> Tuple[float, float]:
        """Computes individual task losses."""
        _, y1_pred, y2_pred = self.forward(X)
        eps = 1e-12
        loss1 = -float(np.mean(y1 * np.log(y1_pred + eps) + (1.0 - y1) * np.log(1.0 - y1_pred + eps)))
        loss2 = float(np.mean((y2_pred - y2) ** 2)) * task2_scale
        return loss1, loss2

    def backward(self, X: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                 task2_scale: float = 1.0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Hand-coded analytical backpropagation returning per-task gradients."""
        n = X.shape[0]
        h, y1_pred, y2_pred = self.forward(X)
        
        # --- Task 1 Gradients (Classification) ---
        dz1 = (y1_pred - y1) / n  # (n, 1)
        dW1 = h.T @ dz1          # (d_hidden, 1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        dh1 = dz1 @ self.W1.T    # (n, d_hidden)
        du1 = dh1 * (1.0 - h ** 2)
        dW_shared_1 = X.T @ du1
        db_shared_1 = np.sum(du1, axis=0, keepdims=True)
        
        grads1 = {
            'W_shared': dW_shared_1, 'b_shared': db_shared_1,
            'W1': dW1, 'b1': db1,
            'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2)
        }
        
        # --- Task 2 Gradients (Regression) ---
        dy2 = (2.0 / n) * (y2_pred - y2) * task2_scale  # (n, 1)
        dW2 = h.T @ dy2
        db2 = np.sum(dy2, axis=0, keepdims=True)
        
        dh2 = dy2 @ self.W2.T
        du2 = dh2 * (1.0 - h ** 2)
        dW_shared_2 = X.T @ du2
        db_shared_2 = np.sum(du2, axis=0, keepdims=True)
        
        grads2 = {
            'W_shared': dW_shared_2, 'b_shared': db_shared_2,
            'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
            'W2': dW2, 'b2': db2
        }
        
        return grads1, grads2


# ==============================================================================
# 3. FINITE-DIFFERENCE GRADIENT CHECK
# ==============================================================================

def check_gradients(net: MultiTaskNet, X: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                    eps: float = 1e-6) -> float:
    """Performs finite-difference central-difference gradient check.
    Returns maximum relative error across all parameters and both tasks."""
    grads1, grads2 = net.backward(X, y1, y2)
    max_rel_error = 0.0

    for task_idx, (loss_func, task_grads) in enumerate([
        (lambda: net.compute_losses(X, y1, y2)[0], grads1),
        (lambda: net.compute_losses(X, y1, y2)[1], grads2)
    ]):
        for param_name, grad_analytic in task_grads.items():
            param_arr = getattr(net, param_name)
            it = np.nditer(param_arr, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                orig_val = param_arr[idx]
                
                param_arr[idx] = orig_val + eps
                l_plus = loss_func()
                
                param_arr[idx] = orig_val - eps
                l_minus = loss_func()
                
                param_arr[idx] = orig_val
                
                grad_num = (l_plus - l_minus) / (2.0 * eps)
                grad_ana = grad_analytic[idx]
                
                denom = max(abs(grad_num), abs(grad_ana), 1e-7)
                rel_err = abs(grad_num - grad_ana) / denom
                if rel_err > max_rel_error:
                    max_rel_error = rel_err
                
                it.iternext()

    return max_rel_error


# ==============================================================================
# 4. MULTI-TASK BALANCING ALGORITHMS (Equal Weights, MGDA, GradNorm)
# ==============================================================================

def solve_mgda_2d(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float]:
    """Finds min-norm convex combination min ||alpha*g1 + (1-alpha)*g2||^2."""
    diff = g1 - g2
    denom = float(diff @ diff)
    if denom < 1e-12:
        return 0.5, 0.5
    alpha = float((g2 - g1) @ g2) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha, 1.0 - alpha


def train_network(mode: str, X: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                  n_epochs: int = 150, lr: float = 0.05,
                  task2_scale: float = 1.0, seed: int = 42) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """Trains network using Equal Weights, MGDA, or GradNorm."""
    net = MultiTaskNet(d_in=5, d_hidden=8, seed=seed)
    loss1_hist, loss2_hist = [], []
    weight_hist = []
    
    # Initial losses for GradNorm
    l1_0, l2_0 = net.compute_losses(X, y1, y2, task2_scale=task2_scale)
    weights = np.array([1.0, 1.0]) # w1, w2
    alpha_gradnorm = 1.5
    lr_weights = 0.025

    for epoch in range(n_epochs):
        l1, l2 = net.compute_losses(X, y1, y2, task2_scale=task2_scale)
        loss1_hist.append(l1)
        loss2_hist.append(l2)
        
        g1, g2 = net.backward(X, y1, y2, task2_scale=task2_scale)
        
        # Flatten shared gradients for multi-objective combination
        g1_shared = np.concatenate([g1['W_shared'].flatten(), g1['b_shared'].flatten()])
        g2_shared = np.concatenate([g2['W_shared'].flatten(), g2['b_shared'].flatten()])
        
        if mode == 'equal':
            w1, w2 = 0.5, 0.5
            comb_shared_W = w1 * g1['W_shared'] + w2 * g2['W_shared']
            comb_shared_b = w1 * g1['b_shared'] + w2 * g2['b_shared']
            weight_hist.append((w1, w2))
            
        elif mode == 'mgda':
            w1, w2 = solve_mgda_2d(g1_shared, g2_shared)
            comb_shared_W = w1 * g1['W_shared'] + w2 * g2['W_shared']
            comb_shared_b = w1 * g1['b_shared'] + w2 * g2['b_shared']
            weight_hist.append((w1, w2))
            
        elif mode == 'gradnorm':
            # GradNorm computation
            G1 = np.linalg.norm(g1['W_shared'])
            G2 = np.linalg.norm(g2['W_shared'])
            G_avg = 0.5 * (G1 + G2)
            
            # Inverse training rates
            r1 = (l1 / max(l1_0, 1e-8))
            r2 = (l2 / max(l2_0, 1e-8))
            r_avg = 0.5 * (r1 + r2)
            r1_rel = (r1 / max(r_avg, 1e-8)) ** alpha_gradnorm
            r2_rel = (r2 / max(r_avg, 1e-8)) ** alpha_gradnorm
            
            target1 = G_avg * r1_rel
            target2 = G_avg * r2_rel
            
            # Gradient of GradNorm loss w.r.t weights
            dw1 = np.sign(G1 - target1)
            dw2 = np.sign(G2 - target2)
            
            weights[0] = max(0.01, weights[0] - lr_weights * dw1)
            weights[1] = max(0.01, weights[1] - lr_weights * dw2)
            weights = 2.0 * weights / np.sum(weights) # Renormalize sum to T=2
            
            w1, w2 = float(weights[0]), float(weights[1])
            comb_shared_W = w1 * g1['W_shared'] + w2 * g2['W_shared']
            comb_shared_b = w1 * g1['b_shared'] + w2 * g2['b_shared']
            weight_hist.append((w1, w2))
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Numerical stability: gradient clipping
        gnorm = np.sqrt(np.sum(comb_shared_W ** 2) + np.sum(comb_shared_b ** 2))
        if gnorm > 15.0:
            scale_factor = 15.0 / gnorm
            comb_shared_W *= scale_factor
            comb_shared_b *= scale_factor

        # Update shared parameters
        net.W_shared -= lr * comb_shared_W
        net.b_shared -= lr * comb_shared_b
        
        # Update head parameters with task-specific gradients
        net.W1 -= lr * g1['W1']
        net.b1 -= lr * g1['b1']
        net.W2 -= lr * g2['W2']
        net.b2 -= lr * g2['b2']

    return loss1_hist, loss2_hist, weight_hist


# ==============================================================================
# 5. MAIN BENCHMARK & PLOTTING EXECUTION
# ==============================================================================

def main():
    print("=" * 78)
    print("P13 PART (b): Multi-Task Neural Network Training & Gradient Balance Study")
    print("=" * 78)
    
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    X, y1, y2 = generate_synthetic_data(n=200, d=5, seed=42)
    print(f"Generated synthetic dataset: X={X.shape}, y1={y1.shape}, y2={y2.shape}")
    
    # 1. Finite-Difference Gradient Check Unit Test
    print("\nRunning finite-difference gradient check unit test...")
    net_test = MultiTaskNet(d_in=5, d_hidden=8, seed=42)
    max_err = check_gradients(net_test, X, y1, y2, eps=1e-6)
    print(f"Maximum relative gradient error across all parameters: {max_err:.2e}")
    assert max_err < 1e-5, f"Gradient check failed! Max relative error {max_err:.2e} >= 1e-5"
    print("GRADIENT CHECK PASSED! (max_rel_error < 1e-5)\n")
    
    # 2. Run Experiments: Standard vs 100x Gradient Imbalance
    scenarios = [
        ("standard", 1.0, 0.03),
        ("imbalanced_100x", 100.0, 0.003)
    ]
    
    results = {}
    csv_rows = []
    
    for scen_name, t2_scale, scen_lr in scenarios:
        print(f"--- Running Scenario: {scen_name} (Task 2 Gradient Scale = {t2_scale}x, lr = {scen_lr}) ---")
        for mode in ['equal', 'mgda', 'gradnorm']:
            l1_h, l2_h, w_h = train_network(mode, X, y1, y2, n_epochs=150, lr=scen_lr,
                                            task2_scale=t2_scale, seed=42)
            results[(scen_name, mode)] = (l1_h, l2_h, w_h)
            final_l1, final_l2 = l1_h[-1], l2_h[-1]
            final_w1, final_w2 = w_h[-1]
            print(f"  [{mode.upper():8s}] Final Loss 1 (BCE): {final_l1:.4f} | Loss 2 (MSE): {final_l2:.4f} | Final Weights: ({final_w1:.3f}, {final_w2:.3f})")
            
            csv_rows.append({
                'Scenario': scen_name,
                'Task2_Scale': t2_scale,
                'Algorithm': mode.upper(),
                'Final_Loss1_BCE': round(final_l1, 5),
                'Final_Loss2_MSE': round(final_l2, 5),
                'Final_Weight_Task1': round(final_w1, 4),
                'Final_Weight_Task2': round(final_w2, 4)
            })

    # Save results to CSV
    csv_path = 'results/data/task_d5_network.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Scenario', 'Task2_Scale', 'Algorithm', 'Final_Loss1_BCE', 'Final_Loss2_MSE',
            'Final_Weight_Task1', 'Final_Weight_Task2'
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved benchmark results to: {csv_path}")

    # 3. Generate Comparative Figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # (a) Standard: Loss 1 (Classification)
    for mode, color, ls in [('equal', 'black', '--'), ('mgda', 'crimson', '-'), ('gradnorm', 'navy', '-.')]:
        axes[0, 0].plot(results[('standard', mode)][0], color=color, linestyle=ls, label=mode.upper(), lw=1.8)
    axes[0, 0].set_title("Standard: Task 1 Loss (BCE)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].legend()

    # (b) Standard: Loss 2 (Regression)
    for mode, color, ls in [('equal', 'black', '--'), ('mgda', 'crimson', '-'), ('gradnorm', 'navy', '-.')]:
        axes[0, 1].plot(results[('standard', mode)][1], color=color, linestyle=ls, label=mode.upper(), lw=1.8)
    axes[0, 1].set_title("Standard: Task 2 Loss (MSE)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].legend()

    # (c) 100x Imbalanced: Loss 1 (Classification under Dominance)
    for mode, color, ls in [('equal', 'black', '--'), ('mgda', 'crimson', '-'), ('gradnorm', 'navy', '-.')]:
        axes[1, 0].plot(results[('imbalanced_100x', mode)][0], color=color, linestyle=ls, label=mode.upper(), lw=1.8)
    axes[1, 0].set_title("100x Imbalance: Task 1 Loss (BCE)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()

    # (d) 100x Imbalanced: Dynamic Weight Trajectories for Task 1
    for mode, color, ls in [('equal', 'black', '--'), ('mgda', 'crimson', '-'), ('gradnorm', 'navy', '-.')]:
        w1_vals = [w[0] for w in results[('imbalanced_100x', mode)][2]]
        axes[1, 1].plot(w1_vals, color=color, linestyle=ls, label=f"{mode.upper()} w1", lw=1.8)
    axes[1, 1].set_title("100x Imbalance: Dynamic Weight $w_1$ for Task 1", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Weight $w_1$")
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = 'results/figures/task_d5_network.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved comparative figure to: {fig_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
