import sys
import re

with open('scripts/generate_report.py', 'r') as f:
    text = f.read()

# 1. The three functions
funcs = r"""
def generate_learning_curves():
    print("[1.5/4] Generating learning curves...")
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
    print("[1.6/4] Generating parity plots...")
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
    print("[1.7/4] Generating convergence history plot...")
    conv_path = PROJECT_ROOT / 'data' / 'results' / 'convergence_log.json'
    if not conv_path.exists():
        print("No convergence log found.")
        return
        
    with open(conv_path, 'r') as f:
        history = json.load(f)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history, marker='o', linestyle='-', color='tab:purple')
    plt.title("Differential Evolution Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Score (Maximised)")
    plt.grid(True, alpha=0.3)
    
    out_path = PROJECT_ROOT / "reports" / "optimization" / f"Convergence_{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_text_report():"""

text = text.replace('def generate_text_report():', funcs)

# 2. Main block call injections
main_block = r"""if __name__ == "__main__":
    generate_correlation_plot()
    generate_learning_curves()
    generate_parity_plots()
    generate_convergence_plot()
    generate_sensitivity_plots()
    generate_text_report()"""

text = re.sub(
    r'if __name__ == "__main__":\s+generate_correlation_plot\(\)\s+generate_sensitivity_plots\(\)\s+generate_text_report\(\)',
    main_block,
    text
)

with open('scripts/generate_report.py', 'w') as f:
    f.write(text)
