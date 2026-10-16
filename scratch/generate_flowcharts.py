"""
Generate matplotlib-rendered flowcharts for Task C deep dive algorithms
to replace mermaid diagrams as required by MASTER_PLAN.md.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart_alg05(output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    steps = [
        ("Elicit DM Preferences\nRisk aversion $\\lambda$, Target vector $z^*$", (5, 6.0), "#E1F5FE", "#0288D1"),
        ("Formulate Scalar Utility Function $U(f(x))$\nMarkowitz: $U(w) = \\mu^T w - \\frac{\\lambda}{2}w^T\\Sigma w$\nTarget: $U(f) = -\\sum w_m ((f_m - z_m^*)/\\sigma_m)^2$", (5, 4.6), "#E8F5E9", "#388E3C"),
        ("Formulate Single-Objective NLP\n$\\min_x \\Phi(x) = -U(f(x)) + \\rho_{\\rm pen} \\cdot CV(x)$", (5, 3.2), "#FFF9C4", "#FBC02D"),
        ("Solve via Non-Linear Optimizer\nSequential Least Squares Programming (SLSQP)\nKKT Gradient Descent with Bounds", (5, 1.8), "#F3E5F5", "#8E24AA"),
        ("Output Optimal Compromise\nOptimal Decision $x^*$, Objective Values $f(x^*)$, Utility $U^*$", (5, 0.5), "#E0F2F1", "#00897B")
    ]

    box_w, box_h = 7.4, 0.95
    for text, (cx, cy), bg_col, edge_col in steps:
        rect = patches.FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=2, edgecolor=edge_col, facecolor=bg_col)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=9.5, fontweight='semibold', color='#212121')

    # Draw arrows between steps
    arrow_props = dict(facecolor='#424242', edgecolor='#424242', width=1.5, headwidth=8, headlength=8)
    for i in range(len(steps) - 1):
        y_start = steps[i][1][1] - box_h/2
        y_end = steps[i+1][1][1] + box_h/2
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=arrow_props)

    plt.title("Algorithm 05 — Value / Utility Function Optimization Architecture", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def draw_flowchart_alg17(output_path):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')

    steps = [
        ("Discretize Objective Space into $K^M$ Hyperboxes\nNormalize bounds across archive: $[f_m^{\\min}, f_m^{\\max}]$", (5, 8.2), "#E1F5FE", "#0288D1"),
        ("Map Archive Members into Grid Hyperboxes\nCompute Squeeze Factor ${\\rm sq}(B) = |\\{x \\in {\\rm Archive} : B(x) = B\\}|$", (5, 6.8), "#E8F5E9", "#388E3C"),
        ("Region-Based Binary Tournament Selection\nSample 2 non-empty hyperboxes $B_1, B_2$; lower squeeze factor ${\\rm sq}(B)$ wins", (5, 5.4), "#FFF9C4", "#FBC02D"),
        ("Sample Parents from Winning Hyperbox\nSample $P_1 \\sim {\\rm Uniform}(B_1^*)$ and $P_2 \\sim {\\rm Uniform}(B_2^*)$", (5, 4.0), "#F3E5F5", "#8E24AA"),
        ("Recombination & Genetic Variation\nApply SBX Crossover ($\\eta_c=20$) and Polynomial Mutation ($\\eta_m=20$)", (5, 2.6), "#FBE9E7", "#D84315"),
        ("Environmental Archive Update & Hyperbox Pruning\nInsert non-dominated offspring; if archive exceeds capacity, prune most crowded box", (5, 1.1), "#E0F2F1", "#00897B")
    ]

    box_w, box_h = 7.8, 0.95
    for text, (cx, cy), bg_col, edge_col in steps:
        rect = patches.FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      linewidth=2, edgecolor=edge_col, facecolor=bg_col)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=9.5, fontweight='semibold', color='#212121')

    arrow_props = dict(facecolor='#424242', edgecolor='#424242', width=1.5, headwidth=8, headlength=8)
    for i in range(len(steps) - 1):
        y_start = steps[i][1][1] - box_h/2
        y_end = steps[i+1][1][1] + box_h/2
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=arrow_props)

    plt.title("Algorithm 17 — PESA-II Hyperbox Selection & Archive Architecture", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def draw_flowchart_alg29(output_path):
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Step boxes
    b1 = ("Initialize Swarm & External Archive\nPositions $x_i$, Velocities $v_i = 0$, Loudness $A_i \\in [1, 2]$, Pulse Rates $r_i \\in [0, 1]$", (5, 9.8), "#E1F5FE", "#0288D1", 8.0, 0.8)
    b2 = ("Select Guide $x^*$ from External Archive via Crowding Selection", (5, 8.5), "#E8F5E9", "#388E3C", 8.0, 0.7)
    b3 = ("Frequency & Velocity Update\n$f_i = f_{\\min} + (f_{\\max}-f_{\\min})\\beta, \\quad v_i = v_i + (x_i - x^*) f_i, \\quad x_{cand} = x_i + v_i$", (5, 7.1), "#FFF9C4", "#FBC02D", 8.0, 0.85)
    b4 = ("Decision: rand > $r_i$ ?", (5, 5.6), "#FFE082", "#FFA000", 4.0, 0.7)
    b5_yes = ("Local Random Walk\n$x_{cand} = x^* + \\sigma \\cdot \\mathcal{N}(0, I)$\n$\\sigma = 0.01 \\cdot A_{mean} \\cdot (u - l)$", (2.5, 4.2), "#F3E5F5", "#8E24AA", 4.2, 0.95)
    b5_no = ("Retain Global Position\n$x_{cand} = x_{cand}$ from velocity", (7.5, 4.2), "#E0E0E0", "#757575", 4.2, 0.95)
    b6 = ("Evaluate Objectives & Constraint Violation $CV(x_{cand})$", (5, 2.7), "#E0F7FA", "#00ACC1", 8.0, 0.65)
    b7 = ("Acceptance & Parameter Adaptation\nIf $x_{cand} \\prec x_i$ or rand < $A_i$: Accept $x_i = x_{cand}$, $A_i = \\alpha A_i$, $r_i = r_0 [1 - \\exp(-\\gamma t)]$", (5, 1.4), "#FBE9E7", "#D84315", 8.2, 0.85)
    b8 = ("Update External Archive & Apply Crowding Distance Pruning", (5, 0.3), "#E0F2F1", "#00897B", 8.0, 0.6)

    boxes = [b1, b2, b3, b4, b5_yes, b5_no, b6, b7, b8]
    for text, (cx, cy), bg_col, edge_col, w, h in boxes:
        rect = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      linewidth=2, edgecolor=edge_col, facecolor=bg_col)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=9.0, fontweight='semibold', color='#212121')

    arrow_props = dict(facecolor='#424242', edgecolor='#424242', width=1.5, headwidth=7, headlength=7)
    # b1 -> b2
    ax.annotate('', xy=(5, 8.85), xytext=(5, 9.4), arrowprops=arrow_props)
    # b2 -> b3
    ax.annotate('', xy=(5, 7.52), xytext=(5, 8.15), arrowprops=arrow_props)
    # b3 -> b4
    ax.annotate('', xy=(5, 5.95), xytext=(5, 6.67), arrowprops=arrow_props)
    # b4 -> b5_yes (Yes)
    ax.annotate('', xy=(2.5, 4.67), xytext=(3.5, 5.6), arrowprops=arrow_props)
    ax.text(2.6, 5.3, "Yes\n(Exploitation)", fontsize=8.5, fontweight='bold', color='#2E7D32')
    # b4 -> b5_no (No)
    ax.annotate('', xy=(7.5, 4.67), xytext=(6.5, 5.6), arrowprops=arrow_props)
    ax.text(6.8, 5.3, "No\n(Exploration)", fontsize=8.5, fontweight='bold', color='#C62828')
    # b5_yes -> b6
    ax.annotate('', xy=(4.0, 3.02), xytext=(2.5, 3.72), arrowprops=arrow_props)
    # b5_no -> b6
    ax.annotate('', xy=(6.0, 3.02), xytext=(7.5, 3.72), arrowprops=arrow_props)
    # b6 -> b7
    ax.annotate('', xy=(5, 1.82), xytext=(5, 2.37), arrowprops=arrow_props)
    # b7 -> b8
    ax.annotate('', xy=(5, 0.6), xytext=(5, 0.97), arrowprops=arrow_props)

    plt.title("Algorithm 29 — Multi-Objective Bat Algorithm (MOBA) Echolocation Dynamics", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)
    draw_flowchart_alg05("results/figures/flowchart_alg05_value_function.png")
    draw_flowchart_alg17("results/figures/flowchart_alg17_pesa2.png")
    draw_flowchart_alg29("results/figures/flowchart_alg29_moba.png")
