import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from TeVnet.sloss import TeVLoss


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, input_channels, inter_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels)
        )
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_feat, input_feat):
        # 关键修改：将input_feat下采样到gate_feat的尺寸
        input_feat_down = F.interpolate(
            input_feat,
            size=gate_feat.shape[2:],  # 目标尺寸：gate_feat的空间大小（如32x32）
            mode="bilinear",
            align_corners=True
        )  # 形状变为 (B, inter_channels, H_g, W_g)

        # 门控特征与下采样后的输入特征融合
        gate = self.gate_conv(gate_feat)  # (B, inter_channels, H_g, W_g)
        inp = self.input_conv(input_feat_down)  # (B, inter_channels, H_g, W_g)

        # 相加激活，生成注意力权重
        att = self.relu(gate + inp)
        att_weight = self.psi(att)  # (B, 1, H_g, W_g)

        # 加权后的输入特征（尺寸与gate_feat一致）
        return input_feat_down * att_weight  # 形状：(B, input_channels, H_g, W_g)


class UNetEncoder(nn.Module):
    def __init__(self, inc=3, content_ch=256):
        super().__init__()
        # 初始卷积（512x512 → 512x512）
        self.init_conv = nn.Sequential(
            nn.Conv2d(inc, 32, kernel_size=3, stride=1, padding=1),  # 单通道→32通道
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 下采样块1（512→256）
        self.down1 = ResidualBlock(32, 64, stride=2)  # 32→64通道，尺寸÷2
        # 下采样块2（256→128）
        self.down2 = ResidualBlock(64, 128, stride=2)  # 64→128通道，尺寸÷2
        # 下采样块3（128→64）
        self.down3 = ResidualBlock(128, 256, stride=2)  # 128→256通道，尺寸÷2
        # 下采样块4（64→32）
        self.down4 = ResidualBlock(256, content_ch, stride=2)  # 256→256通道，尺寸÷2

    def forward(self, x):
        # x: (B, 1, 512, 512) 输入图像
        x = self.init_conv(x)  # (B, 32, 512, 512)
        s1 = x  # 跳连1（512x512，32通道）

        x = self.down1(x)  # (B, 64, 256, 256)
        s2 = x  # 跳连2（256x256，64通道）

        x = self.down2(x)  # (B, 128, 128, 128)
        s3 = x  # 跳连3（128x128，128通道）

        x = self.down3(x)  # (B, 256, 64, 64)
        s4 = x  # 跳连4（64x64，256通道）

        deep_feat = self.down4(x)  # (B, 256, 32, 32) → 最深层特征
        return deep_feat, [s1, s2, s3, s4]  # 四层跳连特征


class UNetDecoder(nn.Module):
    def __init__(self, outc=6, content_ch=768):
        super().__init__()
        # 注意力门控
        self.att4 = AttentionGate(gate_channels=768, input_channels=256, inter_channels=128)  # 对应s4（64x64）
        self.att3 = AttentionGate(gate_channels=256, input_channels=128, inter_channels=64)  # 对应s3（128x128）
        self.att2 = AttentionGate(gate_channels=128, input_channels=64, inter_channels=32)  # 对应s2（256x256）
        self.att1 = AttentionGate(gate_channels=64, input_channels=32, inter_channels=16)  # 对应s1（512x512）

        # 上采样块1（32→64）：拼接最深层特征与s4
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(content_ch + 256, 256, kernel_size=4, stride=2, padding=1),  # 256+256→256通道，尺寸×2
            ResidualBlock(256, 256)
        )
        # 上采样块2（64→128）：拼接上采样1结果与s3
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=4, stride=2, padding=1),  # 256+128→128通道，尺寸×2
            ResidualBlock(128, 128)
        )
        # 上采样块3（128→256）：拼接上采样2结果与s2
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=4, stride=2, padding=1),  # 128+64→64通道，尺寸×2
            ResidualBlock(64, 64)
        )
        # 上采样块4（256→512）：拼接上采样3结果与s1
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1),  # 64+32→32通道，尺寸×2
            ResidualBlock(32, 32)
        )

        # 最终输出层（单通道红外图像）
        self.final_conv = nn.Conv2d(32, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, styled_feat, skip_feats):
        s1, s2, s3, s4 = skip_feats

        # 上采样1：32x32 → 64x64（与s4尺寸匹配）
        att4 = self.att4(styled_feat, s4)  # 注意力加权s4（64x64）
        x = torch.cat([styled_feat, att4], dim=1)  # 256 + 256 = 512通道
        x = self.up1(x)  # (B, 256, 64, 64)

        # 上采样2：64x64 → 128x128（与s3尺寸匹配）
        att3 = self.att3(x, s3)  # 注意力加权s3（128x128）
        x = torch.cat([x, att3], dim=1)  # 256 + 128 = 384通道
        x = self.up2(x)  # (B, 128, 128, 128)

        # 上采样3：128x128 → 256x256（与s2尺寸匹配）
        att2 = self.att2(x, s2)  # 注意力加权s2（256x256）
        x = torch.cat([x, att2], dim=1)  # 128 + 64 = 192通道
        x = self.up3(x)  # (B, 64, 256, 256)

        # 上采样4：256x256 → 512x512（与s1尺寸匹配）
        att1 = self.att1(x, s1)  # 注意力加权s1（512x512）
        x = torch.cat([x, att1], dim=1)  # 64 + 32 = 96通道
        x = self.up4(x)  # (B, 32, 512, 512)

        # 输出512x512单通道图像
        x = self.final_conv(x)  # (B, outc, 512, 512)

        # return torch.sigmoid(x) * 255.0
        x[:, 0, :, :] = torch.sigmoid(x[:, 0, :, :])
        x[:, 1, :, :] = F.relu(x[:, 1, :, :])

        return x


class UNetDecoderWithCycleGAN(nn.Module):
    def __init__(self, outc=3, content_ch=256):
        super().__init__()
        # 注意力门控
        self.att4 = AttentionGate(gate_channels=256, input_channels=256, inter_channels=128)  # 对应s4（64x64）
        self.att3 = AttentionGate(gate_channels=256, input_channels=128, inter_channels=64)  # 对应s3（128x128）
        self.att2 = AttentionGate(gate_channels=128, input_channels=64, inter_channels=32)  # 对应s2（256x256）
        self.att1 = AttentionGate(gate_channels=64, input_channels=32, inter_channels=16)  # 对应s1（512x512）

        # 上采样块1（32→64）：拼接最深层特征与s4
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(content_ch + 256, 256, kernel_size=4, stride=2, padding=1),  # 256+256→256通道，尺寸×2
            ResidualBlock(256, 256)
        )
        # 上采样块2（64→128）：拼接上采样1结果与s3
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=4, stride=2, padding=1),  # 256+128→128通道，尺寸×2
            ResidualBlock(128, 128)
        )
        # 上采样块3（128→256）：拼接上采样2结果与s2
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=4, stride=2, padding=1),  # 128+64→64通道，尺寸×2
            ResidualBlock(64, 64)
        )
        # 上采样块4（256→512）：拼接上采样3结果与s1
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=4, stride=2, padding=1),  # 64+32→32通道，尺寸×2
            ResidualBlock(32, 32)
        )

        # 最终输出层（单通道红外图像）
        self.final_conv = nn.Conv2d(32, outc, kernel_size=3, stride=1, padding=1)

    def forward(self, styled_feat, skip_feats):
        s1, s2, s3, s4 = skip_feats

        # 上采样1：32x32 → 64x64（与s4尺寸匹配）
        att4 = self.att4(styled_feat, s4)  # 注意力加权s4（64x64）
        x = torch.cat([styled_feat, att4], dim=1)  # 256 + 256 = 512通道
        x = self.up1(x)  # (B, 256, 64, 64)

        # 上采样2：64x64 → 128x128（与s3尺寸匹配）
        att3 = self.att3(x, s3)  # 注意力加权s3（128x128）
        x = torch.cat([x, att3], dim=1)  # 256 + 128 = 384通道
        x = self.up2(x)  # (B, 128, 128, 128)

        # 上采样3：128x128 → 256x256（与s2尺寸匹配）
        att2 = self.att2(x, s2)  # 注意力加权s2（256x256）
        x = torch.cat([x, att2], dim=1)  # 128 + 64 = 192通道
        x = self.up3(x)  # (B, 64, 256, 256)

        # 上采样4：256x256 → 512x512（与s1尺寸匹配）
        att1 = self.att1(x, s1)  # 注意力加权s1（512x512）
        x = torch.cat([x, att1], dim=1)  # 64 + 32 = 96通道
        x = self.up4(x)  # (B, 32, 512, 512)

        # 输出512x512单通道图像
        x = self.final_conv(x)  # (B, outc, 512, 512)

        return torch.sigmoid(x)


class TemperatureEmbedding(nn.Module):
    def __init__(self, cond_dim=1, content_ch=256):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Linear(256, content_ch)
        self.beta = nn.Linear(256, content_ch)

        self.temperature_scaler = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, temperature):

        if temperature.dtype != self.temperature_scaler.dtype:
            temperature = temperature.to(self.temperature_scaler.dtype)

        if temperature.dim() == 1:
            temperature = temperature.unsqueeze(1)

        batch_size = temperature.size(0)
        temperature = temperature.view(batch_size, -1)

        scaled_temperature = temperature * self.temperature_scaler
        t_feat = self.embedding(scaled_temperature)
        return self.gamma(t_feat), self.beta(t_feat)


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content_feat, gamma, beta):
        mean = content_feat.mean(dim=[2, 3], keepdim=True)
        std = content_feat.std(dim=[2, 3], keepdim=True) + self.eps
        return gamma.unsqueeze(2).unsqueeze(3) * (content_feat - mean) / std + beta.unsqueeze(2).unsqueeze(3)


class TeVNet(nn.Module):
    def __init__(self, model_path="TeVnet/models/MC/model_epoch_150.pth"):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = UNetEncoder(inc=3, content_ch=768).to(device)
        self.t_embed = TemperatureEmbedding(cond_dim=1, content_ch=768).to(device)
        self.adain = AdaIN().to(device)
        self.decoder = UNetDecoder(outc=6, content_ch=768).to(device)

        # 2. 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.t_embed.load_state_dict(checkpoint["t_embed"])
        self.decoder.load_state_dict(checkpoint["decoder"])

        # 设置为评估模式
        self.encoder.eval()
        self.t_embed.eval()
        self.decoder.eval()

        self.loss = TeVLoss()

    def tev_net(self, x, t):
        gamma, beta = self.t_embed(t)
        content_feat, skips = self.encoder(x)
        styled_feat = self.adain(content_feat, gamma, beta)
        re = self.decoder(styled_feat, skips)
        return re

    def loss_rec(self, pred, t):
        tev_pred = self.tev_net(pred, t)

        rec = self.loss.rec(preds=tev_pred, x=pred)
        pred_mean = torch.mean(pred, dim=1).unsqueeze(1)
        lo = torch.nn.functional.mse_loss(rec, pred_mean, reduction='none').mean([1, 2, 3])
        return lo

    def loss_tev(self, x, pred, t):
        tev_x = self.tev_net(x, t)
        tev_pred = self.tev_net(pred, t)
        lo = torch.nn.functional.mse_loss(tev_x, tev_pred, reduction='none').mean([1, 2, 3])
        return lo

    def forward(self, x, pred, t):
        loss_tev = self.loss_rec(pred, t) + self.loss_tev(x, pred, t)
        loss_tev = torch.clamp(loss_tev, min=0.0, max=1.0)
        return loss_tev.mean()



