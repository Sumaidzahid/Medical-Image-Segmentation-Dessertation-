import torch
import numpy as np
import os
import json

model_paths = [
    "C:\\MMU\\MSc Project\\Models Results\\Swin UNETR\\best_model_fold0.pth",
    "C:\\MMU\\MSc Project\\Models Results\\Swin UNETR\\best_model_fold1.pth",
    # add more folds...
]
fold_results = [] # create a list for all fold results

for path in model_paths:
    print(f"Loading model fold from: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    fold_results.append({
        "fold": data["fold"],
        "epoch": data["epoch"],
        "best_val_dice": float(data["best_val_dice"]),
        "val_loss": float(data["val_loss"]),
        "val_reg_dice": [float(x) for x in data["val_reg_dice"]],
        "best_model_path": path
    }) # add fold results 

print("\n" + "="*80)
print("K-Fold Summary:")

# display results
for f in fold_results:
    print(f"Fold {f['fold']}:")
    print(f"  Best Epoch       : {f['epoch']}")
    print(f"  Best Mean Dice   : {f['best_val_dice']:.4f}")
    print(f"  Val Loss         : {f['val_loss']:.4f}")
    print(f"  Regional Dice    : WT={f['val_reg_dice'][0]:.4f}, "
          f"TC={f['val_reg_dice'][1]:.4f}, ET={f['val_reg_dice'][2]:.4f}")
    print("-"*80)

mean_dice = np.mean([f["best_val_dice"] for f in fold_results])
std_dice = np.std([f["best_val_dice"] for f in fold_results])

print(f"Mean(mean_val_dice) = {mean_dice:.4f} ± {std_dice:.4f}")
# save .json summary
save_dir = "C:\\MMU\\MSc Project\\Models Results\\Swin UNETR" # change path to the model folds .pth folder

with open(os.path.join(save_dir, "kfold_summary.json"), "w") as fh:
    json.dump(fold_results, fh, indent=2)
    print("K-fold summary saved to:", os.path.join(save_dir, "kfold_summary.json"))
