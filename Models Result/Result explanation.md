Results summary

UNet
- Best validation Dice: 0.753 for fold 0 and 0.763 for fold 1.
- The regional Dice values range from 0.621 to 0.851.
- The combined Dice plot rises during training and levels off at approximately 0.75.
- Across the loss plot, both training and validation loss decrease over the 50 epochs; the final training loss is lower than the final validation loss.
- In the fold 0 IoU and sensitivity plots, WT is the highest curve and ET is the lowest curve; the precision plot also shows ET below WT and TC.
- In the displayed prediction images, the main predicted region overlaps the ground-truth region, with a few small additional predicted areas.

Swin UNETR
- Best validation Dice: 0.786 for fold 0 (epoch 44) and 0.799 for fold 1 (epoch 17).
- The regional Dice values range from 0.726 to 0.868.
- The combined Dice plot rises and levels off at approximately 0.78.
- Across the loss plot, both training and validation loss decrease over the 50 epochs; the final training loss is lower than the final validation loss.
- In the fold 0 IoU and sensitivity plots, WT is the highest curve and ET is the lowest curve; the precision plot also shows ET below WT and TC.
- In the displayed prediction images, the main predicted region overlaps the ground-truth region, with some additional small regions outside it.

nnU-Net
- Cross-validation Dice: 0.899 for WT, 0.837 for TC, 0.697 for ET, and 0.811 for Foreground.
- The post-processed scores are the same as the cross-validation scores.
- The fold comparison plot shows fold 0 above fold 1 for WT, TC, and ET.
- The nnU-Net training plots show fluctuating pseudo-Dice values and a steadily rising moving-average curve for both folds.
- The displayed prediction image shows the predicted regions closely matching the ground-truth regions.

Sources
- UNet metrics: UNet/kfold_summary.json and UNet/metrics/metrics_fold0.json.
- UNet plots and images: UNet/plots/fold combined/combined_mean_dice_curve.png and UNet/Qualitative result(unet)_0.png.
- Additional UNet plots: UNet/plots/fold 0/iou_curve.png, UNet/plots/fold 0/precision_curve.png, and UNet/plots/fold 0/sensitivity_curve.png.
- Swin UNETR metrics: Swin UNETR/kfold_summary.json and Swin UNETR/metrics/metrics_fold0.json.
- Swin UNETR plots and images: Swin UNETR/plots/fold combined/combined_mean_dice_curve.png and Swin UNETR/Swin_UNETR_predicted_0.png.
- Additional Swin UNETR plots: Swin UNETR/plots/fold 0/iou_curve.png, Swin UNETR/plots/fold 0/precision_curve.png, and Swin UNETR/plots/fold 0/sensitivity_curve.png.
- nnU-Net metrics: nnunet_results/my_all_processed_results/nnunet_metrics.csv and summary.json.
- nnU-Net plots and image: nnunet_results/my_all_processed_results/nnunet_seg_pred_ensemble.png.
