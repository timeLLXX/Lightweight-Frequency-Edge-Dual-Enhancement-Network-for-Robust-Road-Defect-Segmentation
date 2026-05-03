import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F

from dataset import FullDataset
from tqdm import tqdm
import medpy.metric.binary as metric
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings("ignore")


def calculate_metrics(pred, target, num_classes):
    pred = pred.cpu().numpy().astype(np.uint8)
    target = target.astype(np.uint8)

    iou_per_class = []
    dice_per_class = []

    for class_id in range(num_classes):
        pred_class = (pred == class_id).astype(np.uint8)
        target_class = (target == class_id).astype(np.uint8)

        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()

        iou = intersection / union if union != 0 else 1
        iou_per_class.append(iou)

        dice = metric.dc(pred_class, target_class) if pred_class.sum() > 0 and target_class.sum() > 0 else 0
        dice_per_class.append(dice)


    mean_iou = np.mean(iou_per_class)
    mean_dice = np.mean(dice_per_class)


    return mean_dice, mean_iou

from model.edgesamABCMiloracpr  import EdgeSAMUNet,fuse_model_conv_bn_full,fuse_repvgg_layers
# from model.test_1 import EdgeSAMUNet,fuse_all

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"D:\Desktop\SAM2-UNet\run\E_ABC_newloaradapter_alpha=0.5\best_model.pth",
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--save_path", type=str, default=r"D:\Desktop\SAM2-UNet\run\run_pic",
                    help="path to save the predicted masks")
parser.add_argument("--test_image_path", type=str, default=r"D:\Desktop\SAM2-UNet\new_2\test\images",
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, default=r"D:\Desktop\SAM2-UNet\new_2\test\labels",
                    help="path to the mask files for testing")


parser.add_argument("--num_classes", type=int, default=4, help="number classes")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loader = FullDataset(args.test_image_path, args.test_gt_path,'None', 'test')
test_loader=FullDataset(args.test_image_path, args.test_gt_path,None, 'test')
# test_loader=FullDataset(args.test_image_path, args.test_gt_path,[640,480], 'test')


model = EdgeSAMUNet(args.num_classes)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model=model.cuda()
fuse_repvgg_layers(model)
# --- 执行融合 ---
fuse_model_conv_bn_full(model)
# fuse_model(model)

# fuse_all_bn_in_model(model)
# fuse_all_lora_modules(model)
# fuse_repvgg_layers(model)

# fuse_all(model)




os.makedirs(args.save_path, exist_ok=True)
every_prediction_time = []
assd_scores, iou_scores, dice_losses, dice_scores, hausdorff_distances, visualizations = ([] for _ in range(6))

for i in tqdm(range(len(test_loader)), desc="Testing", ncols=100):
    with torch.no_grad():

        image, gt, name = test_loader[i]
        image = image.unsqueeze(0)
        image = image.to(device)
        start_prediction_time = time.time()
        res, _, _ = model(image)

        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        every_prediction_time.append(prediction_time)
        gt = np.asarray(gt, np.float32)


        res_gray = torch.argmax(F.softmax(res, dim=1), dim=1)

        dsc, iou_value = calculate_metrics(res_gray, gt, args.num_classes)


        iou_scores.append(iou_value)
        dice_scores.append(dsc)


        original_image, original_label, name = test_loader[i]
        res_gray = res_gray.cpu().numpy().squeeze().astype(np.uint8)  # 将 res 转换为二维数组，并转换为 uint8 类型


        res_gray_tensor = torch.tensor(res_gray).unsqueeze(0).unsqueeze(0).float()  # 转换为 (1, 1, H, W)

        res_gray_resized = res_gray_tensor.cpu().numpy().squeeze().astype(np.uint8)  # 再次转换回 numpy 数组

        visualizations.append((original_image, res_gray_resized, original_label, dsc))

        imageio.imsave(os.path.join(args.save_path, name[:-4] + "_gray.png"), res_gray_resized)

        # 获取最大值的索引
        res_color = torch.argmax(res, dim=1)

        # 确保 res 有正确的形状以进行插值
        if res_color.dim() == 3:  # 检查 res 是否只有三个维度 (N, C, H)
            res_color = res_color.unsqueeze(1)  # 添加一个维度变为 (N, 1, C, H)

        # 使用插值调整图像大小
        res_color = F.interpolate(res_color.float(), size=gt.shape[-2:], mode='bilinear', align_corners=None)

        # 去除批次维度并移动到 CPU 并转换为 numpy 数组
        res_color = res_color.squeeze().data.cpu().numpy()  # 形状变为 (526, 1024)

        # 归一化灰度值到 0-1 之间
        res_color = (res_color - res_color.min()) / (res_color.max() - res_color.min() + 1e-8)

        # 应用颜色映射
        # 这里我们使用 Matplotlib 的 'viridis' 颜色映射，你可以选择其他颜色映射
        colormap = plt.colormaps['viridis']
        res_colored = colormap(res_color)  # 形状变为 (1024,526,  4)，包含RGBA通道

        # 移除 alpha 通道（如果不需要）
        res_colored = res_colored[:, :, :3]  # 形状变为 (1024,526, 3)，包含RGB通道

        # 将颜色值转换为 uint8 类型
        res_colored = (res_colored * 255).astype(np.uint8)

        # 保存彩色图像
        imageio.imsave(os.path.join(args.save_path, name[:-4] + "_color.png"), res_colored)

        # print(os.path.join(args.save_path, name),"灰度及彩色图像已保存到：",args.save_path)
average_prediction_time = np.mean(every_prediction_time)
FPS=1/average_prediction_time

avg_iou = np.mean(iou_scores)
avg_dice = np.mean(dice_scores)

print(f"DSC={avg_dice:.4f}, IoU={avg_iou:.4f}, FPS={FPS:.4f}")

from thop import profile
# 定义输入数据的形状
image, _, _ = test_loader[0]
image = image.unsqueeze(0)
image = image.to(device)

# 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(image,))


print(f"模型的 FLOPs: {flops}")
print(f"模型的参数数量: {params}")