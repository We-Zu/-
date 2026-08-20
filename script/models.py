import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FEATURE_DIM, NUM_ANCHORS, GRID_SIZE, DETECTION_BACKBONE_OUT_CHANNELS, DETECTION_NECK_OUT_CHANNELS, UNET_ENCODER_CHANNELS, UNET_BOTTLENECK_CHANNELS
import os
os.environ['TORCH_HUB_OFFLINE'] = '1'

import torchvision.models as models

class DetectionCNN(nn.Module):
    def __init__(self, num_classes, grid_size=GRID_SIZE, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        # 使用预训练ResNet18作为骨干
        resnet = models.resnet18(weights=None)
        # 使用完整的 ResNet18（去掉最后的 avgpool 和 fc）
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # 输出通道数从配置读取
        backbone_out_channels = DETECTION_BACKBONE_OUT_CHANNELS
        # neck层输出通道从配置读取
        neck_out_channels = DETECTION_NECK_OUT_CHANNELS
        self.neck = nn.Conv2d(backbone_out_channels, neck_out_channels, 1)
        self.conv_cls = nn.Conv2d(neck_out_channels, num_classes * num_anchors, 1)
        self.conv_reg = nn.Conv2d(neck_out_channels, 4 * num_anchors, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        cls_out = self.conv_cls(x)
        reg_out = self.conv_reg(x)
        B, _, H, W = cls_out.shape
        cls_out = cls_out.view(B, self.num_anchors, -1, H, W).permute(0, 1, 3, 4, 2)
        reg_out = reg_out.view(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2)
        return cls_out, reg_out

# ---------- 分割模型（U-Net，修改为可提取特征） ----------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        enc_channels = UNET_ENCODER_CHANNELS
        # 编码器
        self.enc1 = conv_block(in_channels, enc_channels[0])
        self.enc2 = conv_block(enc_channels[0], enc_channels[1])
        self.enc3 = conv_block(enc_channels[1], enc_channels[2])
        self.enc4 = conv_block(enc_channels[2], enc_channels[3])
        self.enc5 = conv_block(enc_channels[3], enc_channels[4])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(enc_channels[4], UNET_BOTTLENECK_CHANNELS)
        # 解码器（对称）
        dec_channels = enc_channels[::-1]  # [enc_channels[4], ..., enc_channels[0]]
        self.up5 = nn.ConvTranspose2d(UNET_BOTTLENECK_CHANNELS, dec_channels[0], 2, stride=2)
        self.dec5 = conv_block(enc_channels[4] + dec_channels[0], dec_channels[0])
        self.up4 = nn.ConvTranspose2d(dec_channels[0], dec_channels[1], 2, stride=2)
        self.dec4 = conv_block(enc_channels[3] + dec_channels[1], dec_channels[1])
        self.up3 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2)
        self.dec3 = conv_block(enc_channels[2] + dec_channels[2], dec_channels[2])
        self.up2 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.dec2 = conv_block(enc_channels[1] + dec_channels[3], dec_channels[3])
        self.up1 = nn.ConvTranspose2d(dec_channels[3], dec_channels[4], 2, stride=2)
        self.dec1 = conv_block(enc_channels[0] + dec_channels[4], dec_channels[4])
        self.out = nn.Conv2d(dec_channels[4], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        d5 = self.up5(b)
        d5 = torch.cat((e5, d5), dim=1)
        d5 = self.dec5(d5)
        d4 = self.up4(d5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

    def extract_features(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))
        # 分别进行平均池化和最大池化，然后拼接，特征维度变为 2 * FEATURE_DIM
        avg_feat = F.adaptive_avg_pool2d(b, (1, 1)).view(b.size(0), -1)
        max_feat = F.adaptive_max_pool2d(b, (1, 1)).view(b.size(0), -1)
        features = torch.cat([avg_feat, max_feat], dim=1)  # 维度变为 4096
        return features