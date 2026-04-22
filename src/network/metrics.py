import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TP : True Positive
        # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TN : True Negative
        # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
        # TP : True Positive
        # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC

def get_hd95(pred, gt):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    gt[gt >= 0.5] = 1
    gt[gt < 0.5] = 0

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if not np.any(pred) or not np.any(gt):
        return 0.0  # 如果预测或标签为空，HD95设为0（或根据需要设为最大值）

    # 获取边界点
    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)

    # 计算距离矩阵
    dt_pred = distance_transform_edt(~pred_border)
    dt_gt = distance_transform_edt(~gt_border)

    surface_distances_pred = dt_gt[pred_border]
    surface_distances_gt = dt_pred[gt_border]

    all_surface_distances = np.hstack((surface_distances_pred, surface_distances_gt))

    if len(all_surface_distances) == 0:
        return 0.0
    return np.percentile(all_surface_distances, 95)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    
    output_ = torch.tensor(output_)
    target_=torch.tensor(target_)
    SE = get_sensitivity(output_,target_,threshold=0.5)
    PC = get_precision(output_,target_,threshold=0.5)
    SP= get_specificity(output_,target_,threshold=0.5)
    ACC=get_accuracy(output_,target_,threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)
    hd95 = get_hd95(output, target)
    return iou, dice , SE, PC, F1,SP,ACC,hd95


def iou_score_roc(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)
    return iou, dice, SE, PC, F1, SP, ACC

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


# 整体SSIM结构分量计算函数
def ssim_structure(x: torch.Tensor, y: torch.Tensor,
                   window_size: int = 11, sigma: float = 1.5,
                   data_range: float = 1.0) -> torch.Tensor:
    """
    计算整体SSIM结构分量
    """
    # 确保输入在[0,1]范围内
    x = torch.clamp(x, 0, 1)
    y = torch.clamp(y, 0, 1)

    channel = x.size(1)

    def create_window(window_size, sigma, channel):
        coords = torch.arange(window_size, dtype=torch.float32, device=x.device)
        coords -= window_size // 2

        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        gauss_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        window = gauss_2d.expand(channel, 1, window_size, window_size)

        return window

    window = create_window(window_size, sigma, channel)

    def gaussian_filter(img):
        return torch.nn.functional.conv2d(img, window, padding=window_size // 2, groups=channel)

    mu_x = gaussian_filter(x)
    mu_y = gaussian_filter(y)

    sigma_x2 = gaussian_filter(x * x) - mu_x * mu_x
    sigma_y2 = gaussian_filter(y * y) - mu_y * mu_y
    sigma_xy = gaussian_filter(x * y) - mu_x * mu_y

    C3 = (0.03 * data_range) ** 2
    eps = 1e-8

    structure = (2 * sigma_xy + C3) / (sigma_x2 + sigma_y2 + C3 + eps)

    return structure.mean()


# 器官区域SSIM结构分量计算函数
def organ_ssim_structure(output, target):
    """
    计算器官区域的SSIM结构分量
    """
    # 归一化到[0,1]
    output_norm = torch.sigmoid(output) if output.min() < 0 or output.max() > 1 else output
    target_norm = target.float()

    # 使用target作为器官掩码（假设target是二值的分割标签）
    organ_mask = target_norm > 0.5

    # 确保掩码是二值的
    organ_mask = organ_mask.bool()

    # 如果没有器官区域，返回0
    if not organ_mask.any():
        return torch.tensor(0.0).to(output.device)

    # 提取器官区域
    organ_output = output_norm[organ_mask]
    organ_target = target_norm[organ_mask]

    # 计算器官区域的统计量
    mu_x = organ_output.mean()
    mu_y = organ_target.mean()

    sigma_x = organ_output.std()
    sigma_y = organ_target.std()
    sigma_xy = torch.mean((organ_output - mu_x) * (organ_target - mu_y))

    # 计算结构分量
    C3 = (0.03 * 1.0) ** 2
    eps = 1e-8
    structure = (2 * sigma_xy + C3) / (sigma_x ** 2 + sigma_y ** 2 + C3 + eps)

    return structure
