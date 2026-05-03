import torch
import torch.nn as nn
from thop import profile
import argparse
from dataset import FullDataset
from model.SAM2UNet import SAM2UNet


from model.edgesamABCMiloracpr import EdgeSAMUNet

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=r"D:\Desktop\SAM2-UNet\run\sam2unet_crack\best_model.pth",
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--test_image_path", type=str, default=r"D:\Desktop\SAM2-UNet\Cracks_and_Potholes_in_Road\test\image/",
                    help="path to the image files for testing")
parser.add_argument("--test_gt_path", type=str, default=r"D:\Desktop\SAM2-UNet\Cracks_and_Potholes_in_Road\test\mask/",
                    help="path to the mask files for testing")
parser.add_argument("--num_classes", type=int, default=4, help="number classes")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = FullDataset(args.test_image_path, args.test_gt_path,'None', 'test')

# model = EdgeSAMUNet(args.num_classes)
model = SAM2UNet(args.num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params} 个参数")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型可训练参数量: {trainable_params} 个参数")

model.load_state_dict(torch.load(args.checkpoint), strict=True)

model.eval()
model.cuda()
from edge_sam.modeling.rep_vit import Residual,RepVGGDW,Conv2d_BN,BN_Linear

# --- 步骤2：定义融合函数 ---
def fuse_repvgg_layers(module):
    for name, child in module.named_children():
        # 判断实例的类是否为 Residual 或 RepVGGDW
        if isinstance(child, (Residual, RepVGGDW,Conv2d_BN,BN_Linear)):
            fused_conv = child.fuse()
            setattr(module, name, fused_conv)
        else:
            # 递归调用融合函数，处理子模块
            fuse_repvgg_layers(child)

# --- 执行融合 ---
fuse_repvgg_layers(model)


# 定义输入数据的形状
image, _, _ = test_loader[0]
image = image.unsqueeze(0)
image = image.to(device)

# 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(image,))

print(f"模型的 FLOPs: {flops}")
print(f"模型的参数数量: {params}")


