import re
import json
import os

log_path = "C:\\MMU\\MSc Project\\Models Results\\Swin UNETR\\Swin UNETR.txt"  # training log file

json_base_dir = r"C:\MMU\MSc Project\Models Results\Swin UNETR\metrics"
os.makedirs(json_base_dir, exist_ok=True)

fold_data = {}
current_fold = None

with open(log_path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()

        # Detect fold
        m_fold = re.search(r"Current Fold\s+(\d+)", line)
        if m_fold:
            current_fold = int(m_fold.group(1))
            fold_data.setdefault(current_fold, {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "mean_dice": [],
                "dice_wt": [],
                "dice_tc": [],
                "dice_et": [],
                "precision_wt": [],
                "precision_tc": [],
                "precision_et": [],
                "sensitivity_wt": [],
                "sensitivity_tc": [],
                "sensitivity_et": [],
                "iou_wt": [],
                "iou_tc": [],
                "iou_et": [],
            })
            continue

        if current_fold is None:
            continue

        fd = fold_data[current_fold]

        # Epoch
        m_ep = re.search(r"Epoch\s+(\d+)", line)
        if m_ep:
            fd["epochs"].append(int(m_ep.group(1)))
            continue

        # Train Loss
        m_tl = re.search(r"Train Loss:\s*([0-9]+\.[0-9]+)", line)
        if m_tl:
            fd["train_loss"].append(float(m_tl.group(1)))
            continue

        # Val Loss
        m_vl = re.search(r"Val Loss:\s*([0-9]+\.[0-9]+)", line)
        if m_vl:
            fd["val_loss"].append(float(m_vl.group(1)))

        # Mean Val Dice
        m_md = re.search(r"Mean Val Dice:\s*([0-9]+\.[0-9]+)", line)
        if m_md:
            fd["mean_dice"].append(float(m_md.group(1)))
            continue

        # WT / TC / ET Dice
        m_d = re.search(r"Val Dice\s*\(WT:([0-9]+\.[0-9]+),\s*TC:([0-9]+\.[0-9]+),\s*ET:([0-9]+\.[0-9]+)\)",line,)
        if m_d:
            fd["dice_wt"].append(float(m_d.group(1)))
            fd["dice_tc"].append(float(m_d.group(2)))
            fd["dice_et"].append(float(m_d.group(3)))
            continue
        m_p = re.search(r"Precision\s*\(WT:([0-9]+\.[0-9]+),\s*TC:([0-9]+\.[0-9]+),\s*ET:([0-9]+\.[0-9]+)\)", line)
        if m_p:
            fd["precision_wt"].append(float(m_p.group(1)))
            fd["precision_tc"].append(float(m_p.group(2)))
            fd["precision_et"].append(float(m_p.group(3)))
            continue
        m_s = re.search(r"Sensitivity\s*\(WT:([0-9]+\.[0-9]+),\s*TC:([0-9]+\.[0-9]+),\s*ET:([0-9]+\.[0-9]+)\)", line)
        if m_s:
            fd["sensitivity_wt"].append(float(m_s.group(1)))
            fd["sensitivity_tc"].append(float(m_s.group(2)))
            fd["sensitivity_et"].append(float(m_s.group(3)))
            continue
        m_vi = re.search(r"Val IoU\s*\(WT:([0-9]+\.[0-9]+),\s*TC:([0-9]+\.[0-9]+),\s*ET:([0-9]+\.[0-9]+)\)", line)
        if m_vi:
            fd["iou_wt"].append(float(m_vi.group(1)))
            fd["iou_tc"].append(float(m_vi.group(2)))
            fd["iou_et"].append(float(m_vi.group(3)))
            continue


# Save JSON per fold
for fold_id, fd in fold_data.items():

    for name, values in fd.items():
        if name != "epochs" and len(values) != len(fd["epochs"]):
            raise ValueError(f"Fold {fold_id}: {name} has {len(values)} values, "
                            f"but epochs has {len(fd['epochs'])}.")
    
    json_path = os.path.join(json_base_dir, f"metrics_fold{fold_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fd, f, indent=2)
    print(f"Saved metrics JSON for fold {fold_id} → {json_path}")
