"""
generate_report.py
------------------
Automatically generates the final "Manufacturing & ML Design Justification" Report.
Outputs:
1. Complete text report containing the neural network architecture, optimization weights,
   and the top 3 ranked design candidates physically validated (ready to print).
2. Correlation Heatmap plotting the linear relationship between design variables and performance.
3. Sensitivity plots showing how tweaking one variable impacts the design safety factor and torque.

Usage:
  python scripts/generate_report.py
"""

import sys
import datetime
import json
import logging
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mech390.config import load_config
from mech390.ml import features as F
from mech390.ml.infer import SurrogatePredictor
import summarize_results as _report

logger = logging.getLogger("generate_report")

# Output paths
REPORT_DIR = PROJECT_ROOT / "reports" / "optimization"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TXT_REPORT_PATH = REPORT_DIR / f"Opt_Report_{timestamp}.txt"
HEATMAP_PATH     = REPORT_DIR / "correlation_heatmap.png"
SENSITIVITY_PATH = REPORT_DIR / "sensitivity_analysis.png"
FEASIBILITY_PATH = REPORT_DIR / "feasibility_map.png"
ENGINEERING_PATH = REPORT_DIR / "engineering_analysis.png"

class DualWriter:
    """Redirects stdout to both the console and a text file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def generate_correlation_plot():
    """12-input × 6-output rectangular Pearson-r heatmap from real passing designs."""
    print(f"\n[1/7] Generating Correlation Heatmap -> {HEATMAP_PATH}")
    csv_path = PROJECT_ROOT / "data" / "preview" / "passed_configs.csv"
    if not csv_path.exists():
        print(f"Warning: File {csv_path} not found. Skipping correlation plot.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Warning: passed_configs.csv is empty.")
        return

    df['slenderness_r'] = df['r'] / df['thickness_r']
    df['slenderness_l'] = df['l'] / df['thickness_l']
    df['min_n_fatigue'] = df[['n_f_rod', 'n_f_crank', 'n_f_pin']].min(axis=1)

    input_cols  = ['r', 'l', 'e', 'width_r', 'thickness_r', 'width_l', 'thickness_l',
                   'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C', 'slenderness_r', 'slenderness_l']
    output_cols = ['total_mass', 'tau_A_max', 'min_n_fatigue', 'n_shaft', 'n_buck', 'utilization']

    # Scale to mm for correlation (monotone transform, Pearson r unchanged)
    df_in = df[input_cols].copy()
    for c in ['r', 'l', 'e', 'width_r', 'thickness_r', 'width_l', 'thickness_l',
              'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C']:
        df_in[c] *= 1000

    df_out = df[output_cols].copy()
    df_out['total_mass'] *= 1000  # convert to grams

    corr_matrix = np.zeros((len(input_cols), len(output_cols)))
    for i, ic in enumerate(input_cols):
        for j, oc in enumerate(output_cols):
            corr_matrix[i, j] = df_in[ic].corr(df_out[oc])

    row_labels = [
        "r [mm]", "l [mm]", "e [mm]", "width_r [mm]", "t_r [mm]",
        "width_l [mm]", "t_l [mm]", "d_A [mm]", "d_B [mm]", "d_C [mm]",
        "λ_r = r/t_r", "λ_l = l/t_l"
    ]
    col_labels = ["Mass\n[g]", "τ_A\n[N·m]", "min n_fat\n[-]", "n_shaft\n[-]", "n_buck\n[-]", "Utiliz.\n[-]"]

    corr_df = pd.DataFrame(corr_matrix, index=row_labels, columns=col_labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 10})
    plt.title(f"Parameter–Output Correlation (Pearson r) — {len(df):,} Passing Designs", fontsize=13)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=300)
    plt.close()


def _plot_trend(ax, x, y, n_bins=25, color='darkorange', lw=2):
    """Overlay a binned-median trend line (no extra deps beyond scipy)."""
    from scipy.stats import binned_statistic
    stat, edges, _ = binned_statistic(x, y, statistic='median', bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = ~np.isnan(stat)
    ax.plot(centers[mask], stat[mask], color=color, lw=lw, zorder=5)


def generate_sensitivity_plots():
    """3x3 scatter grid from real ROM/QRR-valid designs. No surrogate extrapolation."""
    print(f"[5/7] Generating Sensitivity Plots -> {SENSITIVITY_PATH}")

    csv_path      = PROJECT_ROOT / "data" / "preview" / "passed_configs.csv"
    csv_fail_path = PROJECT_ROOT / "data" / "preview" / "failed_configs.csv"
    cand_path     = PROJECT_ROOT / "data" / "results" / "candidates.json"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping sensitivity plots.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Warning: passed_configs.csv is empty.")
        return

    # Derived columns
    df['slenderness_l']  = df['l'] / df['thickness_l']
    df['min_n_fatigue']  = df[['n_f_rod', 'n_f_crank', 'n_f_pin']].min(axis=1)
    df['n_buck_capped']  = df['n_buck'].clip(upper=500)
    df['n_f_pin_capped'] = df['n_f_pin'].clip(upper=20)
    df['total_mass_g']   = df['total_mass'] * 1000
    # mm columns for display
    for c in ['r', 'l', 'e', 'thickness_l', 'thickness_r', 'd_shaft_A', 'pin_diameter_B']:
        df[c + '_mm'] = df[c] * 1000

    # Best candidate by weighted score
    cand = None
    if cand_path.exists():
        with open(cand_path) as fh:
            cands = json.load(fh)
        if cands:
            cand = max(cands, key=lambda c: c.get('weighted_score', 0))

    sc_kw   = dict(alpha=0.025, s=1, color='steelblue', rasterized=True)
    star_kw = dict(marker='*', s=200, color='crimson', zorder=10, label='Optimal design')

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(
        f"Design Parameter Sensitivity — {len(df):,} ROM/QRR-Valid Passing Designs",
        fontsize=14
    )

    def _r_annot(ax, col_a, col_b):
        rv = df[[col_a, col_b]].corr().iloc[0, 1]
        ax.annotate(f'r = {rv:.2f}', xy=(0.97, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9, color='gray')

    # ── Row 0: Mass Drivers ──────────────────────────────────────────────────

    # (0,0) Rod thickness → Total mass
    ax = axes[0, 0]
    ax.scatter(df['thickness_l_mm'], df['total_mass_g'], **sc_kw)
    _plot_trend(ax, df['thickness_l_mm'].values, df['total_mass_g'].values)
    if cand:
        ax.scatter(cand['thickness_l'] * 1000, cand.get('pred_total_mass', 0) * 1000,
                   **star_kw)
        ax.legend(fontsize=8)
    _r_annot(ax, 'thickness_l', 'total_mass')
    ax.set_xlabel("Rod thickness $t_l$ [mm]")
    ax.set_ylabel("Total mass [g]")
    ax.set_title("Rod Thickness → Total Mass")
    ax.grid(True, alpha=0.3)

    # (0,1) Crank thickness → Total mass
    ax = axes[0, 1]
    ax.scatter(df['thickness_r_mm'], df['total_mass_g'], **sc_kw)
    _plot_trend(ax, df['thickness_r_mm'].values, df['total_mass_g'].values)
    if cand:
        ax.scatter(cand['thickness_r'] * 1000, cand.get('pred_total_mass', 0) * 1000,
                   marker='*', s=200, color='crimson', zorder=10)
    _r_annot(ax, 'thickness_r', 'total_mass')
    ax.set_xlabel("Crank thickness $t_r$ [mm]")
    ax.set_ylabel("Total mass [g]")
    ax.set_title("Crank Thickness → Total Mass")
    ax.grid(True, alpha=0.3)

    # (0,2) Rod slenderness → Buckling FoS (log scale)
    ax = axes[0, 2]
    ax.scatter(df['slenderness_l'], df['n_buck_capped'], **sc_kw)
    _plot_trend(ax, df['slenderness_l'].values, df['n_buck_capped'].values)
    ax.axhline(3.0, color='red', linestyle='--', lw=1.5, label='Design floor n=3')
    ax.set_yscale('log')
    if cand:
        sl = cand['l'] / cand['thickness_l']
        ax.axvline(sl, color='crimson', linestyle=':', lw=1.5, label='Optimal $λ_l$')
    _r_annot(ax, 'slenderness_l', 'n_buck')
    ax.set_xlabel("Rod slenderness $l/t_l$")
    ax.set_ylabel("Buckling FoS $n_{buck}$ (log, capped 500)")
    ax.set_title("Rod Slenderness → Buckling Safety")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ── Row 1: Safety Factor Drivers ─────────────────────────────────────────

    # (1,0) Shaft diameter → Shaft FoS
    ax = axes[1, 0]
    ax.scatter(df['d_shaft_A_mm'], df['n_shaft'], **sc_kw)
    _plot_trend(ax, df['d_shaft_A_mm'].values, df['n_shaft'].values)
    ax.axhline(2.0, color='red', linestyle='--', lw=1.5, label='Min FoS = 2')
    if cand:
        ax.scatter(cand['d_shaft_A'] * 1000, cand.get('pred_n_shaft', 0),
                   marker='*', s=200, color='crimson', zorder=10)
    rv = df[['d_shaft_A', 'n_shaft']].corr().iloc[0, 1]
    ax.annotate(f'r = {rv:.2f} (near-linear)', xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlabel("Shaft diameter $d_A$ [mm]")
    ax.set_ylabel("Shaft FoS $n_{shaft}$")
    ax.set_title("Shaft Diameter → Shaft Safety")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Pin B diameter → Pin fatigue FoS
    ax = axes[1, 1]
    ax.scatter(df['pin_diameter_B_mm'], df['n_f_pin_capped'], **sc_kw)
    _plot_trend(ax, df['pin_diameter_B_mm'].values, df['n_f_pin_capped'].values)
    ax.axhline(1.0, color='red', linestyle='--', lw=1.5, label='Goodman limit')
    if cand:
        ax.scatter(cand['pin_diameter_B'] * 1000, cand.get('pred_min_n_fatigue', 0),
                   marker='*', s=200, color='crimson', zorder=10)
    _r_annot(ax, 'pin_diameter_B', 'n_f_pin')
    ax.set_xlabel("Pin B diameter $d_B$ [mm]")
    ax.set_ylabel("Pin fatigue FoS (capped at 20)")
    ax.set_title("Pin B Diameter → Fatigue Safety")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Pin B diameter → Stress utilization
    ax = axes[1, 2]
    ax.scatter(df['pin_diameter_B_mm'], df['utilization'], **sc_kw)
    _plot_trend(ax, df['pin_diameter_B_mm'].values, df['utilization'].values)
    ax.axhline(1.0, color='red', linestyle='--', lw=1.5, label='Yield limit')
    if cand:
        ax.scatter(cand['pin_diameter_B'] * 1000, cand.get('pred_utilization', 0),
                   marker='*', s=200, color='crimson', zorder=10)
    _r_annot(ax, 'pin_diameter_B', 'utilization')
    ax.set_xlabel("Pin B diameter $d_B$ [mm]")
    ax.set_ylabel("Stress utilization $\\sigma/S_y$ [-]")
    ax.set_title("Pin B Diameter → Stress Utilization")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Kinematic Context ──────────────────────────────────────────────

    # (2,0) Rod length → Crank torque
    ax = axes[2, 0]
    ax.scatter(df['l_mm'], df['tau_A_max'], **sc_kw)
    _plot_trend(ax, df['l_mm'].values, df['tau_A_max'].values)
    if cand:
        ax.scatter(cand['l'] * 1000, cand.get('pred_tau_A_max', 0),
                   marker='*', s=200, color='crimson', zorder=10)
    _r_annot(ax, 'l', 'tau_A_max')
    ax.set_xlabel("Rod length $l$ [mm]")
    ax.set_ylabel("Peak crank torque $\\tau_A$ [N·m]")
    ax.set_title("Rod Length → Crank Torque")
    ax.grid(True, alpha=0.3)

    # (2,1) Offset → Crank torque
    ax = axes[2, 1]
    ax.scatter(df['e_mm'], df['tau_A_max'], **sc_kw)
    _plot_trend(ax, df['e_mm'].values, df['tau_A_max'].values)
    if cand:
        ax.scatter(cand['e'] * 1000, cand.get('pred_tau_A_max', 0),
                   marker='*', s=200, color='crimson', zorder=10)
    _r_annot(ax, 'e', 'tau_A_max')
    ax.set_xlabel("Offset $e$ [mm]")
    ax.set_ylabel("Peak crank torque $\\tau_A$ [N·m]")
    ax.set_title("Offset → Crank Torque")
    ax.grid(True, alpha=0.3)

    # (2,2) Mass vs Shaft FoS — Pareto hexbin
    ax = axes[2, 2]
    hb = ax.hexbin(df['total_mass_g'], df['n_shaft'], gridsize=40, cmap='Blues', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.axhline(2.0, color='red', linestyle='--', lw=1.5, label='Min FoS = 2')
    if cand:
        cand_mass   = cand.get('pred_total_mass', 0) * 1000
        cand_nshaft = cand.get('pred_n_shaft', 0)
        ax.axvline(cand_mass, color='crimson', linestyle=':', lw=1.5)
        ax.scatter(cand_mass, cand_nshaft, marker='*', s=200, color='crimson',
                   zorder=10, label='Optimal design')
        ax.axvspan(ax.get_xlim()[0], cand_mass, alpha=0.08, color='green',
                   label='Lighter + safe zone')
    ax.set_xlabel("Total mass [g]")
    ax.set_ylabel("Shaft FoS $n_{shaft}$")
    ax.set_title("Mass vs. Shaft Safety (Pareto)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.savefig(SENSITIVITY_PATH, dpi=300)
    plt.close()

    # Feasibility map saved separately
    _generate_feasibility_map(df, csv_fail_path, cand)


def _generate_feasibility_map(df_pass, csv_fail_path, cand):
    """Pass/fail scatter in (rod thickness, pin B diameter) space."""
    print(f"  -> Generating feasibility map -> {FEASIBILITY_PATH}")
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(8, 6))

    if csv_fail_path.exists():
        df_fail = pd.read_csv(csv_fail_path)
        idx_f = rng.choice(len(df_fail), size=min(3000, len(df_fail)), replace=False)
        df_f = df_fail.iloc[idx_f]
        ax.scatter(df_f['thickness_l'] * 1000, df_f['pin_diameter_B'] * 1000,
                   color='tomato', alpha=0.3, s=6, label=f"Fail (n={len(df_fail):,})")

    idx_p = rng.choice(len(df_pass), size=min(3000, len(df_pass)), replace=False)
    df_p = df_pass.iloc[idx_p]
    ax.scatter(df_p['thickness_l'] * 1000, df_p['pin_diameter_B'] * 1000,
               color='steelblue', alpha=0.3, s=6, label=f"Pass (n={len(df_pass):,})")

    if cand:
        cand_tl = cand['thickness_l'] * 1000
        cand_dB = cand['pin_diameter_B'] * 1000
        ax.scatter(cand_tl, cand_dB, color='gold', edgecolors='black',
                   marker='*', s=300, zorder=20, label='Optimal design')
        ax.annotate(
            f"Optimal: $t_l$={cand_tl:.1f} mm, $d_B$={cand_dB:.1f} mm",
            xy=(cand_tl, cand_dB),
            xytext=(cand_tl + 0.5, cand_dB + 0.2),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'),
        )

    ax.set_xlabel("Rod thickness $t_l$ [mm]")
    ax.set_ylabel("Pin B diameter $d_B$ [mm]")
    ax.set_title("Feasibility Map: Rod Thickness vs. Pin B Diameter")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FEASIBILITY_PATH, dpi=300)
    plt.close()





def generate_learning_curves():
    print("[2/7] Generating learning curves...")
    hist_path = PROJECT_ROOT / 'data' / 'models' / 'optuna_history.json'
    if not hist_path.exists():
        print("No optuna history found.")
        return
    with open(hist_path, 'r') as f:
        history = json.load(f)
        
    num_trials = len(history)
    best_f1_overall, best_trial_id = -1.0, -1
    for t in history:
        if t['best_val_f1'] > best_f1_overall:
            best_f1_overall = t['best_val_f1']
            best_trial_id = t['trial_id']
            
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    for t in history:
        epochs = [h['epoch'] for h in t['history']]
        train_loss = [h['train_loss'] for h in t['history']]
        val_loss = [h['val_loss'] for h in t['history']]
        alpha = 0.8 if t['trial_id'] == best_trial_id else 0.15
        color_t = 'tab:blue'
        color_v = 'tab:orange'
        axes[0].plot(epochs, train_loss, color=color_t, alpha=alpha)
        axes[0].plot(epochs, val_loss, color=color_v, alpha=alpha)
    
    axes[0].plot([], [], color='tab:blue', label='Train Loss')
    axes[0].plot([], [], color='tab:orange', label='Val Loss')
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Val F1
    for t in history:
        epochs = [h['epoch'] for h in t['history']]
        val_f1 = [h['val_f1'] for h in t['history']]
        alpha = 0.8 if t['trial_id'] == best_trial_id else 0.15
        color = 'tab:green' if t['trial_id'] == best_trial_id else 'tab:gray'
        axes[1].plot(epochs, val_f1, color=color, alpha=alpha)
        
    axes[1].plot([], [], color='tab:green', label='Best Trial')
    axes[1].plot([], [], color='tab:gray', label='Other Trials')
    axes[1].set_title("Validation F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "training" / f"Learning_Curves_{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_parity_plots():
    print("[3/7] Generating parity plots...")
    preds_path = PROJECT_ROOT / 'data' / 'models' / 'validation_preds.npz'
    if not preds_path.exists():
        print("No validation preds found.")
        return
        
    data = np.load(preds_path)
    preds_reg = data['preds_reg']
    labels_reg = data['labels_reg']
    
    cols = 3
    rows = int(np.ceil(preds_reg.shape[1] / cols))
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    for i, name in enumerate(F.REGRESSION_TARGETS):
        ax = fig.add_subplot(rows, cols, i+1)
        y_true = labels_reg[:, i]
        y_pred = preds_reg[:, i]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot > 1e-12:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = float('nan')
        
        ax.set_title(f"{name} (R²={r2:.3f})")
        ax.set_xlabel("True (Normalised)")
        ax.set_ylabel("Predicted (Normalised)")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = PROJECT_ROOT / "reports" / "training" / f"Parity_Plots_{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_convergence_plot():
    print("[4/7] Generating convergence history plot...")
    conv_path = PROJECT_ROOT / 'data' / 'results' / 'convergence_log.json'
    if not conv_path.exists():
        print("No convergence log found.")
        return
        
    with open(conv_path, 'r') as f:
        history = json.load(f)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='tab:purple')
    plt.title("Differential Evolution Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Score (Maximised)")
    plt.grid(True, alpha=0.3)
    
    out_path = PROJECT_ROOT / "reports" / "optimization" / f"Convergence_{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_text_report():
    print(f"[7/7] Generating final text report -> {TXT_REPORT_PATH}")
    writer = DualWriter(TXT_REPORT_PATH)
    original_stdout = sys.stdout
    sys.stdout = writer
    
    try:
        print("="*80)
        print("    MECH 390 — NEURAL NETWORK & DESIGN OPTIMIZATION REPORT    ")
        print("="*80)
        print(f"Generated on : {timestamp}")
        print(f"Pipeline Dir : {PROJECT_ROOT}\n")
        
        # 1. Neural Network Architecture Details
        print("--- MACHINE LEARNING MODEL ARCHITECTURE ---")
        ckpt_path = PROJECT_ROOT / "data" / "models" / "surrogate_best.pt"
        if ckpt_path.exists():
            # Use weights_only=True to prevent warning
            ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
            hparams = ckpt.get("hparams", {})
            print(f"Input features   : {hparams.get('input_dim', 'N/A')} dimensions")
            print(f"Reg targets      : {hparams.get('n_reg_targets', 'N/A')} targets")
            print(f"Hidden Layers    : {hparams.get('hidden_sizes', 'N/A')}")
            print(f"Dropout Rate     : {hparams.get('dropout_rate', 'N/A')}")
            # Handle hardcoded -1 epoch from train.py sweep
            total_epochs = ckpt.get('epoch', 'N/A')
            epoch_str = "Auto-Stopped (Best Trial)" if total_epochs == -1 else total_epochs

            print(f"Validation F1    : {ckpt.get('val_f1', 0.0):.4f}")
            print(f"Total Epochs     : {epoch_str}")
        else:
            print("Model checkpoint not found. Surrogate details disabled.\n")
            
        # 2. Optimization Weights / Constraints
        print("\n--- OPTIMIZATION PRIORITY WEIGHTS & CONSTRAINTS ---")
        opt_yaml = PROJECT_ROOT / "configs" / "optimize" / "search.yaml"
        if opt_yaml.exists():
            with open(opt_yaml, "r") as f:
                opt_cfg = yaml.safe_load(f)
                
            print("The following weights were used by the Differential Evolution Algorithm")
            print("to select the absolute best candidate out of millions of combinations:\n")
            
            for obj, spec in opt_cfg.get("objectives", {}).items():
                print(f"  {obj.ljust(20)} | Weight: {spec.get('weight', 0)*100:4.1f}% | Direction: {spec.get('direction', 'min')}")
            
            print("\nHard Constraints Evaluated:")
            for k, v in opt_cfg.get("constraints", {}).items():
                print(f"  {k.ljust(25)} = {v}")
        else:
            print("Search config not found.\n")
            
        print("\n" + "="*80)
        print("    TOP COMPILED OPTIMIZATION CANDIDATES (PHYSICS VALIDATED)    ")
        print("="*80 + "\n")
        
        # 3. Use summarize_results.py engine directly!
        gen_path = PROJECT_ROOT / "configs" / "generate" / "baseline.yaml"
        cand_path = PROJECT_ROOT / "data" / "results" / "candidates.json"
        
        if gen_path.exists() and cand_path.exists():
            _report.run(
                candidates_path=str(cand_path),
                config_path=str(gen_path),
                top_n=3
            )
        else:
            print("ERROR: Missing dependencies to print final candidates.")

    finally:
        sys.stdout = original_stdout
        writer.close()
        print(f"\n[✓] Text report successfully created: {TXT_REPORT_PATH}")


def generate_engineering_analysis():
    """2x3 mechanical engineering analysis: kinematics, load path, mass breakdown, multi-objective."""
    print(f"[6/7] Generating Engineering Analysis -> {ENGINEERING_PATH}")

    csv_path  = PROJECT_ROOT / "data" / "preview" / "passed_configs.csv"
    cand_path = PROJECT_ROOT / "data" / "results" / "candidates.json"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping engineering analysis.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Warning: passed_configs.csv is empty.")
        return

    # Derived columns
    df['r_mm']  = df['r']  * 1000
    df['tg']    = df['total_mass'] * 1000
    df['frac_rod']    = df['mass_rod']    / df['total_mass']
    df['frac_crank']  = df['mass_crank']  / df['total_mass']
    df['frac_slider'] = df['mass_slider'] / df['total_mass']
    df['power_W'] = df['E_rev'] * (df['omega'] / (2 * np.pi))  # P = E_rev * f [W]
    df['min_n_fatigue'] = df[['n_f_rod', 'n_f_crank', 'n_f_pin']].min(axis=1)

    # Best candidate
    cand = None
    if cand_path.exists():
        with open(cand_path) as fh:
            cands_raw = json.load(fh)
        if cands_raw:
            cand = max(cands_raw, key=lambda c: c.get('weighted_score', 0))

    sc_kw = dict(alpha=0.025, s=1, color='steelblue', rasterized=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"Mechanical Engineering Analysis — {len(df):,} Passing Designs",
        fontsize=14
    )

    # ── (0,0) r vs QRR — kinematic quick-return map ──────────────────────────
    ax = axes[0, 0]
    ax.scatter(df['r_mm'], df['QRR'], **sc_kw)
    _plot_trend(ax, df['r_mm'].values, df['QRR'].values)
    ax.axhspan(1.5, 2.5, alpha=0.08, color='green', label='QRR spec window')
    ax.axhline(1.5, color='green', linestyle='--', lw=1, alpha=0.6)
    ax.axhline(2.5, color='green', linestyle='--', lw=1, alpha=0.6)
    if cand:
        ax.axvline(cand['r'] * 1000, color='crimson', linestyle=':', lw=1.5,
                   label=f"Optimal r={cand['r']*1000:.1f} mm")
    rv = df[['r', 'QRR']].corr().iloc[0, 1]
    ax.annotate(f'r = {rv:.2f}', xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, color='gray')
    ax.set_xlabel("Crank radius $r$ [mm]")
    ax.set_ylabel("Quick return ratio QRR")
    ax.set_title("Crank Radius vs. Quick Return Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,1) F_B_max distribution — pin B bearing load ─────────────────────
    ax = axes[0, 1]
    ax.hist(df['F_B_max'], bins=60, color='steelblue', edgecolor='none', alpha=0.8)
    if cand:
        # Find closest row in dataset by geometry
        close = df.iloc[(df['r'] - cand['r']).abs().argsort()[:1]]
        fb_cand = close['F_B_max'].values[0]
        ax.axvline(fb_cand, color='crimson', linestyle='--', lw=2,
                   label=f'Nearest dataset design\nF_B = {fb_cand:.1f} N')
        ax.legend(fontsize=8)
    ax.set_xlabel("Peak crank-pin load $F_B$ [N]")
    ax.set_ylabel("Count")
    ax.set_title("Crank-Pin Bearing Load Distribution")
    median_fb = df['F_B_max'].median()
    ax.annotate(f'Median = {median_fb:.1f} N', xy=(0.97, 0.93), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, color='gray')
    ax.grid(True, alpha=0.3)

    # ── (0,2) tau_A_max vs volume_envelope colored by min_n_fatigue ──────────
    # Multi-objective Pareto: torque (motor cost) vs size vs fatigue life
    ax = axes[0, 2]
    n_fat_capped = df['min_n_fatigue'].clip(upper=20)
    sc = ax.scatter(df['tau_A_max'], df['volume_envelope'] * 1e6,  # cm3
                    c=n_fat_capped, cmap='RdYlGn', alpha=0.15, s=2, rasterized=True,
                    vmin=1, vmax=20)
    plt.colorbar(sc, ax=ax, label='min fatigue FoS (capped 20)')
    if cand:
        ax.scatter(cand.get('pred_tau_A_max', 0),
                   cand.get('pred_volume_envelope', 0) * 1e6,
                   marker='*', s=200, color='crimson', zorder=10, label='Optimal design')
        ax.legend(fontsize=8)
    ax.set_xlabel("Peak crank torque $\\tau_A$ [N·m]")
    ax.set_ylabel("Volume envelope [cm³]")
    ax.set_title("Multi-Objective Map: Torque vs. Size vs. Fatigue")
    ax.grid(True, alpha=0.3)

    # ── (1,0) Mass breakdown violin — which link dominates? ──────────────────
    ax = axes[1, 0]
    fracs = [df['frac_crank'] * 100, df['frac_rod'] * 100, df['frac_slider'] * 100]
    labels = ['Crank', 'Rod', 'Slider']
    # Use boxplot (no seaborn needed; consistent with rest of file)
    bp = ax.boxplot(fracs, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    if cand:
        # Compute actual fractions from candidates.json masses (not available)
        # Just annotate median values
        pass
    ax.set_ylabel("Mass fraction [%]")
    ax.set_title("Link Mass Breakdown (Rod dominates)")
    for i, (lbl, frac) in enumerate(zip(labels, fracs), start=1):
        ax.annotate(f'med {frac.median():.0f}%', xy=(i, frac.median()),
                    xytext=(i + 0.15, frac.median()), fontsize=8, va='center', color='black')
    ax.grid(True, alpha=0.3, axis='y')

    # ── (1,1) Motor power requirement distribution ────────────────────────────
    # P = E_rev * RPM/60  — negative means deceleration phase dominates
    ax = axes[1, 1]
    power_W = df['power_W']
    ax.hist(power_W[power_W < 0], bins=40, color='royalblue',
            alpha=0.8, edgecolor='none', label=f'Regenerative ({(power_W<0).sum():,} designs)')
    ax.hist(power_W[power_W >= 0], bins=40, color='darkorange',
            alpha=0.8, edgecolor='none', label=f'Motoring ({(power_W>=0).sum():,} designs)')
    ax.axvline(0, color='black', lw=1.5, linestyle='--')
    if cand:
        omega_cand = 30 * 2 * np.pi / 60
        p_cand = cand.get('pred_E_rev', 0) * (omega_cand / (2 * np.pi))
        ax.axvline(p_cand, color='crimson', linestyle=':', lw=2,
                   label=f'Optimal P={p_cand:.3f} W')
    ax.set_xlabel("Net motor power [W]  (negative = braking phase dominates)")
    ax.set_ylabel("Count")
    ax.set_title("Motor Power Requirement Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,2) Joint load comparison — F_A vs F_B vs F_C ─────────────────────
    ax = axes[1, 2]
    # Subsample to avoid overplotting
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(5000, len(df)), replace=False)
    ds = df.iloc[idx]
    ax.scatter(ds['F_A_max'], ds['F_B_max'],
               alpha=0.15, s=2, color='steelblue', rasterized=True, label='F_A vs F_B')
    ax.scatter(ds['F_A_max'], ds['F_C_max'],
               alpha=0.15, s=2, color='darkorange', rasterized=True, label='F_A vs F_C')
    # 45-degree reference line
    lim = max(df['F_B_max'].max(), df['F_C_max'].max(), df['F_A_max'].max())
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='F_A = F_x')
    if cand:
        close_row = df.iloc[(df['r'] - cand['r']).abs().argsort()[:1]]
        ax.scatter(close_row['F_A_max'].values[0], close_row['F_B_max'].values[0],
                   marker='*', s=200, color='crimson', zorder=10)
    ax.set_xlabel("Motor shaft load $F_A$ [N]")
    ax.set_ylabel("Pin load [N]")
    ax.set_title("Joint Load Comparison F_A vs F_B, F_C")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.savefig(ENGINEERING_PATH, dpi=300)
    plt.close()
    print(f"  -> Saved: {ENGINEERING_PATH}")


if __name__ == "__main__":
    generate_correlation_plot()
    generate_learning_curves()
    generate_parity_plots()
    generate_convergence_plot()
    generate_sensitivity_plots()
    generate_engineering_analysis()
    generate_text_report()
