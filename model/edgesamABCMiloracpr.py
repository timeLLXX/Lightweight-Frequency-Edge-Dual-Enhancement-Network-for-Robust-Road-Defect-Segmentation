import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import sys
sys.path.append("..")

from edge_sam import sam_model_registry
from edge_sam.modeling.rep_vit import Residual, RepVGGDW, Conv2d_BN, BN_Linear, RepViTBlock

from tools.EBFE import EBFE
from tools.EBFE import FourierUnit
from tools.EdgeEnhancer import EdgeEnhancer


# =========================
# Loss
# =========================
class structure_loss(nn.Module):
    def __init__(self, classes=None, eps: float = 1.0, kernel_size: int = 31):
        super().__init__()
        self.classes = classes
        self.eps = eps
        self.kernel = kernel_size
        self.pad = kernel_size // 2

    def forward(self, pred: torch.Tensor, mask: torch.Tensor):
        mask = mask.clone()

        B, C, H, W = pred.shape
        num_classes = C if self.classes is None else self.classes
        assert num_classes == C, f"num_classes({num_classes}) must equal pred channels({C})"

        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        elif mask.dim() != 3:
            raise ValueError(f"mask should be (B,H,W) or (B,1,H,W), got {mask.shape}")

        mask = mask.to(dtype=torch.long)
        mask = mask.clamp(0, num_classes - 1)

        m = mask.float().unsqueeze(1)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(m, kernel_size=self.kernel, stride=1, padding=self.pad) - m
        )

        ce = F.cross_entropy(pred, mask, reduction='none')
        weighted_ce = (weit.squeeze(1) * ce).sum() / (weit.sum() + 1e-6)

        prob = F.softmax(pred, dim=1)
        one_hot = F.one_hot(mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
        inter = ((prob * one_hot) * weit).sum(dim=(2, 3))
        union = ((prob + one_hot) * weit).sum(dim=(2, 3))
        iou = (inter + self.eps) / (union - inter + self.eps)
        iou_loss = 1 - iou.mean()

        return weighted_ce + iou_loss


# =========================
# Basic blocks
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.EdgeEnhancer = EdgeEnhancer(in_channels, norm=nn.BatchNorm2d, act=nn.ReLU)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(self.EdgeEnhancer(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



def milora_svd_init(W_2d: torch.Tensor, r: int, alpha: float, mode: str = "min"):
    # ---- 强制检查 W 必须是 2D ----
    if W_2d.dim() != 2:
        raise ValueError(f"[milora_svd_init] W must be 2D, got shape={tuple(W_2d.shape)}")
    print("[DEBUG] W_2d shape:", tuple(W_2d.shape))

    out_dim, in_dim = W_2d.shape
    max_r = min(out_dim, in_dim)
    r = int(min(r, max_r))
    if r < 1:
        scaling = 0.0
        A = W_2d.new_zeros((1, in_dim))
        B = W_2d.new_zeros((out_dim, 1))
        return A, B, scaling

    # SVD
    U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)  # U:[out,k], S:[k], Vh:[k,in]
    k = S.numel()
    r = min(r, k)
    if r < 1:
        scaling = 0.0
        A = W_2d.new_zeros((1, in_dim))
        B = W_2d.new_zeros((out_dim, 1))
        return A, B, scaling

    if mode == "min":
        U_sel = U[:, -r:]
        S_sel = S[-r:]
        V_sel = Vh[-r:, :]
    elif mode == "max":
        U_sel = U[:, :r]
        S_sel = S[:r]
        V_sel = Vh[:r, :]
    elif mode == "mid":
        mid_start = (k - r) // 2
        U_sel = U[:, mid_start:mid_start + r]
        S_sel = S[mid_start:mid_start + r]
        V_sel = Vh[mid_start:mid_start + r, :]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scaling = alpha / r
    S_sel = S_sel / max(1e-12, scaling)
    S_sqrt = torch.sqrt(S_sel.clamp_min(0))

    B = U_sel @ torch.diag(S_sqrt)      # [out, r]
    A = torch.diag(S_sqrt) @ V_sel      # [r, in]


    assert A.shape == (r, in_dim), f"A shape wrong: {A.shape}, expected {(r, in_dim)}"
    assert B.shape == (out_dim, r), f"B shape wrong: {B.shape}, expected {(out_dim, r)}"

    return A, B, scaling




class LoRAConv1x1(nn.Module):

    def __init__(self, conv: nn.Conv2d, r=8, alpha=0.5):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        assert conv.kernel_size == (1, 1), "LoRAConv1x1 expects 1x1 conv"

        self.base = conv
        for p in self.base.parameters():
            p.requires_grad = False

        in_ch = conv.in_channels
        out_ch = conv.out_channels

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)

        # 标准 LoRA 初始化：A 用 kaiming，B 用零（常见做法，起点更稳）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


class MiLoRAConv1x1(nn.Module):
    def __init__(self, conv: nn.Conv2d, r=8, alpha=0.5, svd_mode="min"):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        assert conv.kernel_size == (1, 1)

        self.base = conv
        for p in self.base.parameters():
            p.requires_grad = False

        with torch.no_grad():
            W = conv.weight.data                    # [out,in,1,1]
            out_ch, in_ch = W.shape[0], W.shape[1]  # ✅ 以 weight 为准
            W2d = W.reshape(out_ch, in_ch)          # ✅ 永远合法
            A, B, scaling = milora_svd_init(W2d.float(), r=r, alpha=alpha, mode=svd_mode)

        r_used = int(A.shape[0])
        self.r = r_used
        self.scaling = float(scaling)

        self.lora_A = nn.Conv2d(in_ch, r_used, 1, bias=False)
        self.lora_B = nn.Conv2d(r_used, out_ch, 1, bias=False)

        with torch.no_grad():
            self.lora_A.weight.copy_(A.view(r_used, in_ch, 1, 1).to(self.lora_A.weight.dtype))
            self.lora_B.weight.copy_(B.view(out_ch, r_used, 1, 1).to(self.lora_B.weight.dtype))

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling





def inject_adapter_to_backbone_1x1(model: nn.Module,
                                  adapter_type: str = "lora",
                                  r: int = 8,
                                  alpha: float = 0.5,
                                  milora_svd_mode: str = "min"):


    if adapter_type == "lora":
        adapter_cls = lambda conv: LoRAConv1x1(conv, r=r, alpha=alpha)
    elif adapter_type == "milora":
        adapter_cls = lambda conv: MiLoRAConv1x1(conv, r=r, alpha=alpha, svd_mode=milora_svd_mode)
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    def name_filter(full_name: str, module: nn.Module) -> bool:
        # 只注入 backbone
        if not (full_name.startswith("stage2.") or full_name.startswith("stage3.") or full_name.startswith("stage4.")):
            return False
        # 避免误注入到你现在已有的旧 LoRAAdapter 的 lora_A/lora_B（以后删了旧 adapter 就不会有）
        if ".lora_A" in full_name or ".lora_B" in full_name:
            return False
        # 只对 1×1 conv
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            return True
        return False

    def _replace(parent: nn.Module, prefix=""):
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and name_filter(full, child):
                #  关键：跳过 rank 太小的层
                max_r = min(child.in_channels, child.out_channels)
                if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                    W = child.weight
                    if W.shape[1] == 1 or W.shape[0] == 1:
                        print(f"[Skip MiLoRA low-rank] {full} weight={tuple(W.shape)}")
                        continue

                setattr(parent, child_name, adapter_cls(child))

            else:
                _replace(child, full)

    _replace(model)
    return model



def fuse_lora_conv1x1_inplace(module: nn.Module):

    for name, child in list(module.named_children()):
        if isinstance(child, (LoRAConv1x1, MiLoRAConv1x1)):
            base = child.base
            with torch.no_grad():
                W = base.weight.data.squeeze(-1).squeeze(-1)  # [out, in]
                A = child.lora_A.weight.data.squeeze(-1).squeeze(-1)  # [r, in]
                B = child.lora_B.weight.data.squeeze(-1).squeeze(-1)  # [out, r]
                deltaW = (B @ A) * child.scaling  # [out, in]
                W_new = W + deltaW.to(W.dtype)
                base.weight.data.copy_(W_new.view_as(base.weight.data))
            setattr(module, name, base)  # 用融合后的 base conv 替换 adapter
        else:
            fuse_lora_conv1x1_inplace(child)



def fuse_conv_bn(conv, bn):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    )

    w_conv = conv.weight.clone()
    mean = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(var + eps)
    w_bn = gamma / std

    fused_conv.weight.data = w_conv * w_bn.reshape([-1, 1, 1, 1])

    b_conv = conv.bias if conv.bias is not None else torch.zeros_like(mean)
    fused_conv.bias.data = beta - gamma * mean / std + w_bn * b_conv

    return fused_conv


def fuse_model_conv_bn_full(module):
    prev_name = None
    prev_module = None

    for name, child in list(module.named_children()):
        if isinstance(child, FourierUnit):
            continue

        if isinstance(prev_module, nn.Conv2d) and isinstance(child, nn.BatchNorm2d):
            fused = fuse_conv_bn(prev_module, child)
            setattr(module, prev_name, fused)
            delattr(module, name)
            fuse_model_conv_bn_full(fused)
            return

        elif isinstance(child, nn.Module):
            fuse_model_conv_bn_full(child)

        prev_name = name
        prev_module = child


def fuse_repvgg_layers(module):
    for name, child in module.named_children():
        if isinstance(child, (Residual, RepVGGDW, Conv2d_BN, BN_Linear, RepViTBlock)):
            fused_conv = child.fuse()
            setattr(module, name, fused_conv)
        else:
            fuse_repvgg_layers(child)



class EdgeSAMUNet(nn.Module):
    def __init__(self,
                 number_class=20,
                 checkpoint_path=r"",#root:edge_sam_3x.pth
                 model_type="edge_sam",
                 adapter_type="milora",   # "lora" or "milora"
                 lora_r=8,
                 lora_alpha=0.5,
                 milora_svd_mode="min"):
        super().__init__()

        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        del model.prompt_encoder
        del model.mask_decoder
        del model.image_encoder.fuse_stage2
        del model.image_encoder.fuse_stage3
        del model.image_encoder.neck

        self.stage1 = nn.Sequential(model.image_encoder.features[0])
        for p in self.stage1.parameters():
            p.requires_grad = False

        self.stage2 = nn.Sequential(
            model.image_encoder.features[1],
            model.image_encoder.features[2],
            model.image_encoder.features[3],
            model.image_encoder.features[4],
        )
        for p in self.stage2.parameters():
            p.requires_grad = False

        self.stage3 = nn.Sequential(*[model.image_encoder.features[i] for i in range(5, 9)])
        for p in self.stage3.parameters():
            p.requires_grad = False

        self.stage4 = nn.Sequential(*[model.image_encoder.features[i] for i in range(9, 25)])
        for p in self.stage4.parameters():
            p.requires_grad = False

        #  核心：在 backbone 内注入 1×1 LoRA/MiLoRA
        inject_adapter_to_backbone_1x1(
            self,
            adapter_type=adapter_type,
            r=lora_r,
            alpha=lora_alpha,
            milora_svd_mode=milora_svd_mode
        )

        # Decoder（保持不变）
        self.rfb1 = EBFE(48, 64)
        self.rfb2 = EBFE(96, 64)
        self.rfb3 = EBFE(192, 64)
        self.rfb4 = EBFE(384, 64)

        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)
        self.side1 = nn.Conv2d(64, number_class, kernel_size=1)
        self.side2 = nn.Conv2d(64, number_class, kernel_size=1)
        self.head = nn.Conv2d(64, number_class, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')

        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')

        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')

        return out, out1, out2



if __name__ == "__main__":
    with torch.no_grad():
        model = EdgeSAMUNet(number_class=20, adapter_type="milora", lora_r=8, lora_alpha=0.5).cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")

