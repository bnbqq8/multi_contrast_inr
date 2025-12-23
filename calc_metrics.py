from pathlib import Path

import lpips
import nibabel as nib
import torch
import yaml

from dataset import InferDataset, MultiModalDataset
from dataset_utils import get_image_coordinate_grid_nib
from utils import compute_metrics, dict2obj

GPU_DEVICE = 0
device = f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
lpips_loss = lpips.LPIPS(net="alex").to(device)
config = "config/config_brats_mlpv2_custom.yaml"
with open(config) as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)
config = dict2obj(config_dict)

dataset = MultiModalDataset(
    image_dir=config.SETTINGS.DIRECTORY,
    name=config.SETTINGS.PROJECT_NAME,
    subject_id=config.DATASET.SUBJECT_ID,
    contrast1_LR_str=config.DATASET.LR_CONTRAST1,
    contrast2_LR_str=config.DATASET.LR_CONTRAST2,
)

x_dim, y_dim, z_dim = dataset.get_dim()
gt = dataset.get_contrast1_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()
# pred = dataset.get_contrast1_pred().reshape((x_dim, y_dim, z_dim)).cpu().numpy()
pred1_path = "/home/czfy/multi_contrast_inr/runs/IXI dataset_images/IXI dataset_subid-IXI002-Guys-0828_ct1LR-T1_ct2LR-T2_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__Adam_0.0004__e49__ct1.nii.gz"
pred1 = (
    get_image_coordinate_grid_nib(nib.load(str(pred1_path)))["intensity_norm"]
    .reshape((x_dim, y_dim, z_dim))
    .cpu()
    .numpy()
)
compute_metrics1 = compute_metrics(
    gt, pred1, dataset.get_contrast1_gt_mask(), lpips_loss, device
)

print("Metrics for Pred 1:")
for key, value in compute_metrics1.items():
    print(f"{key}: {value:.4f}")

pred2_path = "/home/czfy/smore/result/T1/T1_smore4.nii.gz"
pred2 = (
    get_image_coordinate_grid_nib(nib.load(str(pred2_path)))["intensity_norm"]
    .reshape((x_dim, y_dim, z_dim))
    .cpu()
    .numpy()
)

compute_metrics2 = compute_metrics(
    gt, pred2, dataset.get_contrast1_gt_mask(), lpips_loss, device
)

print("Metrics for Pred 2:")
for key, value in compute_metrics2.items():
    print(f"{key}: {value:.4f}")
