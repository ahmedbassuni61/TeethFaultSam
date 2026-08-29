import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Mask_Refiner import (
    ConvBnRelu, 
    ResidualBlock, 
    SelfAttention, 
    EncoderBlock, 
    DecoderBlock, 
    FeatureFusionSimple
)

class LightweightSAMFeatureProcessor(nn.Module):
    """Generate multi-scale SAM features with half channel width"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.level3_conv = ConvBnRelu(in_channels, 128)
        self.level4_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.level2_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.level1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        level3 = self.level3_conv(x)  # [B, 128, 64, 64]
        level4 = self.level4_conv(x)  # [B, 256, 32, 32]

        level2_feat = self.level2_conv(x)  # [B, 64, 64, 64]
        level1_feat = self.level1_conv(x)  # [B, 32, 64, 64]
        level2 = F.interpolate(level2_feat, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 64, 128, 128]
        level1 = F.interpolate(level1_feat, scale_factor=4, mode='bilinear', align_corners=True)  # [B, 32, 256, 256]

        return {
            'level1': level1,
            'level2': level2,
            'level3': level3,
            'level4': level4
        }

class CalculusRefiner(nn.Module):
    """Lightweight binary ResUNet for calculus segmentation"""
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.num_classes = num_classes

        # Initial processors
        self.image_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.mask_processor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 2 channels for bg+calculus
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.sam_processor = LightweightSAMFeatureProcessor(256)

        # Encoders
        self.image_encoder1 = EncoderBlock(32, 32, use_attention=False, dropout_rate=dropout_rate)
        self.image_encoder2 = EncoderBlock(32, 64, use_attention=False, dropout_rate=dropout_rate)
        self.image_encoder3 = EncoderBlock(64, 128, use_attention=False, dropout_rate=dropout_rate)
        self.image_encoder4 = EncoderBlock(128, 256, use_attention=True, dropout_rate=dropout_rate)

        self.mask_encoder1 = EncoderBlock(32, 32, use_attention=False, dropout_rate=dropout_rate)
        self.mask_encoder2 = EncoderBlock(32, 64, use_attention=False, dropout_rate=dropout_rate)
        self.mask_encoder3 = EncoderBlock(64, 128, use_attention=False, dropout_rate=dropout_rate)
        self.mask_encoder4 = EncoderBlock(128, 256, use_attention=True, dropout_rate=dropout_rate)

        # Feature fusion
        self.fusion1 = FeatureFusionSimple(32, 32, 32, 32)
        self.fusion2 = FeatureFusionSimple(64, 64, 64, 64)
        self.fusion3 = FeatureFusionSimple(128, 128, 128, 128)
        self.fusion4 = FeatureFusionSimple(256, 256, 256, 256)

        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512, dropout_rate=dropout_rate)

        # Decoders
        self.decoder4 = DecoderBlock(512, 256, 256, use_attention=True, dropout_rate=dropout_rate)
        self.decoder3 = DecoderBlock(256, 128, 128, use_attention=False, dropout_rate=dropout_rate)
        self.decoder2 = DecoderBlock(128, 64, 64, use_attention=False, dropout_rate=dropout_rate)
        self.decoder1 = DecoderBlock(64, 32, 32, use_attention=False, dropout_rate=dropout_rate)

        # Final output
        self.final_conv = nn.Sequential(
            ConvBnRelu(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, images, calculus_mask, sam_embedding):
        """
        Args:
            images: [B, 3, H, W]
            calculus_mask: [B, 2, H, W]
            sam_embedding: [B, 256, 64, 64]
        """
        input_size = images.shape[2:]

        sam_features = self.sam_processor(sam_embedding)
        x_image = self.image_processor(images)
        x_mask = self.mask_processor(calculus_mask)

        # Stage 1
        e1_image = self.image_encoder1(x_image)
        e1_mask = self.mask_encoder1(x_mask)
        e1_fusion = self.fusion1(e1_image, e1_mask, sam_features['level1'])
        x_image = F.max_pool2d(e1_image, 2)
        x_mask = F.max_pool2d(e1_mask, 2)

        # Stage 2
        e2_image = self.image_encoder2(x_image)
        e2_mask = self.mask_encoder2(x_mask)
        e2_fusion = self.fusion2(e2_image, e2_mask, sam_features['level2'])
        x_image = F.max_pool2d(e2_image, 2)
        x_mask = F.max_pool2d(e2_mask, 2)

        # Stage 3
        e3_image = self.image_encoder3(x_image)
        e3_mask = self.mask_encoder3(x_mask)
        e3_fusion = self.fusion3(e3_image, e3_mask, sam_features['level3'])
        x_image = F.max_pool2d(e3_image, 2)
        x_mask = F.max_pool2d(e3_mask, 2)

        # Stage 4
        e4_image = self.image_encoder4(x_image)
        e4_mask = self.mask_encoder4(x_mask)
        e4_fusion = self.fusion4(e4_image, e4_mask, sam_features['level4'])

        # Bottleneck
        x = F.max_pool2d(e4_fusion, 2)
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder4(x, e4_fusion)
        x = self.decoder3(x, e3_fusion)
        x = self.decoder2(x, e2_fusion)
        x = self.decoder1(x, e1_fusion)

        # Output
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        logits = self.final_conv(x)
        return logits
