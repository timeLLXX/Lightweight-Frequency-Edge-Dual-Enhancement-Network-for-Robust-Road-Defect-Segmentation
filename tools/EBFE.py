import torch
import torch.nn as nn




class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels * 2, out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度，强制 float32
            x = x.float()  # 强制类型转换
            ffted = torch.fft.rfft2(x, norm='ortho')  # [1, 64, H, W//2+1]

            x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)  # [B, C, H, Wf, 1]
            x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)  # [B, C, H, Wf, 1]
            ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)       # [B, C, H, Wf, 2]
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()         # [B, C, 2, H, Wf]
            ffted = ffted.view((batch, -1, *ffted.size()[3:]))        # [B, C*2, H, Wf]

            ffted = self.relu(self.bn(self.conv_layer(ffted)))       # [B, C*2, H, Wf]
            ffted = ffted.view((batch, -1, 2, *ffted.size()[2:]))     # [B, C, 2, H, Wf]
            ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()         # [B, C, H, Wf, 2]
            ffted = torch.view_as_complex(ffted)                     # [B, C, H, Wf]

            output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho') # [B, C, H, W]

        return x - output


class EBFE(nn.Module):
    def __init__(self, input_channels,output_channels, reduction_N=48):#64   7809
        """
            input_channels: 输入通道数（如RGB图像为3）
            reduction_N: 特征压缩维度（默认32）
        """
        super(TBFE, self).__init__()

        # 点卷积（通道降维）[2,4](@ref)
        # 1x1卷积核实现跨通道信息交互，类似SENet[2](@ref)的通道注意力机制
        self.point_wise = nn.Conv2d(input_channels, reduction_N,
                                    kernel_size=1, padding=0, bias=False)


        # 深度可分离卷积（空间特征提取）[2,4](@ref)
        # 3x3卷积核提取局部空间特征，BN+ReLU增强非线性表达能力
        self.depth_wise = nn.Sequential(
            nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(reduction_N),
            nn.ReLU(),
        )

        # 三维卷积（时序特征建模）[1,4](@ref)
        # 沿通道维度构建时序建模能力，kernel_size=(1,1,3)表示在通道维度进行时序卷积
        self.conv3D = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 3),  # (深度,高度,宽度)
            padding=(0, 0, 1),  # 保持时序维度尺寸不变
            stride=(1, 1, 1),
            bias=False
        )
        #一维卷积
        self.conv1d = nn.Conv1d(
            in_channels=reduction_N,
            out_channels=reduction_N,
            kernel_size=3,
            padding="same"  # 保持输出长度与输入一致（可选）
        )

        # 特征融合与恢复
        self.bn = nn.BatchNorm2d(3 * reduction_N)  # 融合双分支特征
        self.relu = nn.ReLU()

        # 投影层（恢复原始通道数）[2](@ref)
        # 1x1卷积实现通道维度变换，类似ResNet[2](@ref)的shortcut连接
        self.pro = nn.Conv2d(3* reduction_N, output_channels,
                             kernel_size=1, padding=0, bias=False)
        self.fuliye=FourierUnit(reduction_N, reduction_N)

    def forward(self, x):
        """前向传播过程（含维度变换注释）"""
        # 原始输入尺寸：(batch_size, input_channels, H, W)
        # 阶段1：通道压缩
        x_1 = self.point_wise(x)  # 输出尺寸：(B, reduction_N, H, W)
        # x_1 = x_1 + self.fuliye(x_1)

        # 阶段2：空间特征提取（含残差连接）[2](@ref)
        x_2 =x_1+ self.depth_wise(x_1)  # 输出尺寸保持(B, reduction_N, H, W)
        # x_2=x_2+self.fuliye(x_2)


        # 阶段3：时序特征建模[1,4](@ref)
        x_3 = x_1.unsqueeze(1)  # 增加时间维度：(B, 1, reduction_N, H, W)
        x_3 = self.conv3D(x_3)  # 3D卷积处理：(B, 1, reduction_N, H, W)
        x_3 = x_3.squeeze(1)  # 压缩时间维度：(B, reduction_N, H, W)
        x_3 = x_3 + self.fuliye(x_3)

        # 调整维度：将宽度(w)合并到batch维度
        x_reshaped = x_1.permute(0, 3, 1, 2)  # 变为 [batch, w, n, h]
        x_reshaped = x_reshaped.reshape(-1, x_1.shape[1], x_1.shape[2])  # 变为 [batch*w, n, h]

        # 执行卷积
        out = self.conv1d(x_reshaped)  # 输出形状 [batch*w, out_channels, h]

        # 恢复原始维度
        out = out.reshape(x_1.shape[0], x_1.shape[3], x_1.shape[1], x_1.shape[2])
        out = x_1 + out.permute(0, 2, 3, 1)  # 最终形状 [batch, out_channels, h_out, w]

        # 阶段4：特征融合
        x = torch.cat((x_2, x_3,out), dim=1)  # 通道拼接：(B, 2*reduction_N, H, W)

        x = self.bn(x)  # 标准化处理
        x = self.relu(x)  # 非线性激活
        x = self.pro(x)  # 通道恢复：(B, input_channels, H, W)
        return x

if __name__ == "__main__":
    model = TBFE(16,8)  # 实例化模块（模拟处理RGB图像）
    input = torch.randn(1, 16, 128, 128)  # 输入张量：(batch=1, channel=3, height=128, width=128)
    output = model(input)

    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')
