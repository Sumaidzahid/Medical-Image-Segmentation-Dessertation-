import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

json_dir = r"C:\MMU\MSc Project\Models Results\Swin UNETR\metrics"
plots_dir = r"C:\MMU\MSc Project\Models Results\Swin UNETR\plots"

save_plots = True
os.makedirs(plots_dir, exist_ok=True)

# best epochs per fold, set them here:
best_epochs = {
    0: 44,
    1: 17,
    # add more folds
}

# ============================================================
# LOAD ALL FOLD JSON FILES
# ============================================================

fold_files = [f for f in os.listdir(json_dir) if f.startswith("metrics_fold")]
fold_data = {}
# Load each fold's metrics from JSON files
for file in fold_files:
    fold_id = int(file.replace("metrics_fold", "").replace(".json", "")) # Extract fold ID from filename
    with open(os.path.join(json_dir, file), "r") as f:
        fold_data[fold_id] = json.load(f) # Store the loaded metrics in the fold_data dictionary

# ============================================================
# FUNCTION TO SAVE OR SHOW PLOTS
# ============================================================

def save_or_show(fig, fold_id, name, combined_path=None):
    if save_plots:
        fold_dir = os.path.join(plots_dir, f"fold {fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        fig.savefig(os.path.join(fold_dir, name), dpi=300)
        print(f"[Fold {fold_id}] Saved plot: {name}")
        plt.close(fig)
    elif save_plots and combined_path is not None:
        fig.savefig(combined_path, dpi=300)
        print(f"Saved combined plot: {name}")
        plt.close(fig)
    else:
        plt.show()

# ============================================================
# PLOTTING
# ============================================================

def plot_metrics(epochs, metrics, metric_name, fold_id, best_epoch=None):
    fig = plt.figure(figsize=(12, 6))
    for metric_label, metric_values in metrics.items():
        plt.plot(epochs, metric_values, label=metric_label)
    if best_epoch is not None:
        plt.axvline(best_epoch, color='red', linestyle='--', label=f"Best Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Fold {fold_id} - {metric_name} Curves")
    plt.legend()
    plt.grid(True)
    save_or_show(fig, fold_id, f"{metric_name.lower().replace(' ', '_')}_curve.png")

metrics_to_plot = {
    "Dice": ["dice_wt", "dice_tc", "dice_et"],
    "Precision": ["precision_wt", "precision_tc", "precision_et"],
    "Sensitivity": ["sensitivity_wt", "sensitivity_tc", "sensitivity_et"],
    "IoU": ["iou_wt", "iou_tc", "iou_et"]
}
    # -------------------------------
    # WT / TC / ET:
    #    DICE CURVES
    #    Precision Curves
    #    Sensitivity Curves
    #    IoU Curves
    # -------------------------------
for fold_id, fd in sorted(fold_data.items()):
    # Extract metrics for the current fold
    epochs = fd["epochs"]
    best_epoch = best_epochs.get(fold_id, None)
    for metric_name, metric_keys in metrics_to_plot.items():
        metrics = {k: fd[k] for k in metric_keys if fd.get(k)}
        if len(metrics) == len(metric_keys):
            plot_metrics(epochs, metrics, metric_name, fold_id, best_epoch)

# ============================================================
# COMBINED MEAN DICE: ONE LINE PER FOLD
# ============================================================
fig = plt.figure(figsize=(12, 6))
for fold_id, fd in sorted(fold_data.items()):
    plt.plot(fd["epochs"], fd["mean_dice"], linewidth=2, label=f"Fold {fold_id}")

    best_epoch = best_epochs.get(fold_id)
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", linewidth=1, alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Mean Validation Dice Across All Folds")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

combined_path = os.path.join(plots_dir, "all_folds_mean_dice_curve.png")
save_or_show(fig, "all_folds", "all_folds_mean_dice_curve.png", combined_path=combined_path)

# ============================================================
# COMBINED MEAN DICE: AVERAGE ACROSS ALL FOLDS
# ============================================================
mean_dice_curves = [
    np.asarray(fd["mean_dice"], dtype=float)
    for _, fd in sorted(fold_data.items())
]
combined_mean_dice = np.mean(np.vstack(mean_dice_curves), axis=0)
first_epochs = sorted(fold_data.items())[0][1]["epochs"]

fig = plt.figure(figsize=(12, 6))
plt.plot(first_epochs, combined_mean_dice, color="black", linewidth=3,label="Combined Mean Dice")
plt.xlabel("Epoch")
plt.ylabel("Mean Dice Score")
plt.title("Combined Mean Dice Across All Folds")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

combined_mean_path = os.path.join(plots_dir, "combined_mean_dice_curve.png")
save_or_show(fig, "combined", "combined_mean_dice_curve.png", combined_path=combined_mean_path)

# ============================================================
# COMBINED DICE LOSS
# ============================================================
fig = plt.figure(figsize=(12, 6))
for fold_id, fd in sorted(fold_data.items()):
    plt.plot(fd["epochs"], fd["val_loss"], linewidth=2, label=f"Fold {fold_id}")
    plt.plot(fd["epochs"], fd["train_loss"], linewidth=1, linestyle='--', alpha=0.5, label=f"Fold {fold_id} Train Loss")

    best_epoch = best_epochs.get(fold_id)
    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", linewidth=1, alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Across All Folds")
plt.legend()
plt.grid(True) 
plt.tight_layout()

combined_loss_path = os.path.join(plots_dir, "all_folds_loss_curve.png")
save_or_show(fig, "all_folds", "all_folds_loss_curve.png", combined_path=combined_loss_path)

# ============================================================
# Combined Mean Dice: Average Across All Folds
# ============================================================

print(f"Combined mean Dice: {combined_mean_dice[-1]:.4f}")
print("\nAll plots generated successfully.")