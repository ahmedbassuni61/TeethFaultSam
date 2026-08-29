import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sam2.build_sam import build_sam2
from model.sam2.utils.transforms import SAM2Transforms
from model.PEGnet import MultiLevelFeatureFusion
from model.calculus.calculus_prompt_generator import CalculusPEG
from model.calculus.calculus_head import CalculusRefiner
from model.calculus.lora import inject_lora, get_lora_params

class CalculusSegmentationSystem(nn.Module):
    """End-to-end calculus segmentation system"""
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.get('embed_dim', 256)
        self.num_queries = config.get('num_queries', 4)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.lora_rank = config.get('lora_rank', 4)
        
        # 1 & 2. Build SAM2 and freeze
        self.sam_model = build_sam2(
            config_file=config.get('sam_config', 'configs/sam2.1/sam2.1_hiera_l.yaml'),
            ckpt_path=config.get('sam_checkpoint', 'model/sam2/checkpoints/sam2.1_hiera_large.pt')
        )
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        # 3. Inject LoRA
        inject_lora(self.sam_model, rank=self.lora_rank)
        
        # 4. Feature Fusion
        self.feature_fusion = MultiLevelFeatureFusion(
            d_model=self.d_model,
            dropout=self.dropout_rate
        )
        
        # 5. Calculus PEG
        self.calculus_peg = CalculusPEG(
            d_model=self.d_model,
            num_queries=self.num_queries,
            num_layers=config.get('peg_layers', 3),
            dropout=self.dropout_rate
        )
        
        # 6. Calculus Refiner
        self.refine_net = CalculusRefiner(
            num_classes=2,
            dropout_rate=config.get('refiner_dropout', 0.4)
        )
        
        # 7. Store backbone feature sizes
        self.bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        
        # 8. SAM2Transforms
        self._transforms = SAM2Transforms(
            resolution=self.sam_model.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

    def process_images(self, images):
        B, _, H, W = images.shape
        orig_hw = [(H, W) for _ in range(B)]
        
        img_batch = self._transforms.transforms(images)
        
        backbone_out = self.sam_model.forward_image(img_batch)
        _, vision_feats, _, _ = self.sam_model._prepare_backbone_features(backbone_out)
        
        if self.sam_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]
        return image_embed, high_res_feats, orig_hw

    def generate_calculus_masks(self, prompt_embed, image_embed, high_res_feats, orig_hw):
        B = prompt_embed.shape[0]

        image_embeddings = torch.repeat_interleave(image_embed, self.num_queries, dim=0)
        high_res_features = [torch.repeat_interleave(feats, self.num_queries, dim=0) for feats in high_res_feats]
        
        sparse_embeddings = prompt_embed.reshape(-1, prompt_embed.shape[-1]).unsqueeze(1)
        
        dense_embeddings = self.sam_model.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B * self.num_queries, 
            -1, 
            self.sam_model.sam_prompt_encoder.image_embedding_size[0], 
            self.sam_model.sam_prompt_encoder.image_embedding_size[1]
        )

        low_res_masks, _, _, _ = self.sam_model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,  
            repeat_image=False,  
            high_res_features=high_res_features,
        )

        masks = F.interpolate(low_res_masks, orig_hw[0], mode="bilinear", align_corners=False)
        masks = masks.view(B, self.num_queries, masks.shape[2], masks.shape[3])

        # Combine into [B, 2, H, W]
        # Channel 1: max of 4 query logits
        calculus_masks, _ = torch.max(masks, dim=1, keepdim=True)
        
        # Channel 0: background logits
        fg_probs = torch.sigmoid(masks)
        p = torch.clamp(fg_probs.sum(dim=1, keepdim=True), 0, 1)
        eps = 1e-6
        # Background = log((1 - clamp(sum(sigmoid(masks)), 0, 1)) / (1 - (1-p) + eps)) -> which simplifies to
        # bg_prob / (1 - bg_prob + eps) + eps
        bg_prob = 1.0 - p
        bg_masks = torch.log(bg_prob / (1 - bg_prob + eps) + eps)
        
        sam_masks = torch.cat([bg_masks, calculus_masks], dim=1)  # [B, 2, H, W]

        return sam_masks

    def forward(self, images):
        image_embed, high_res_feats, orig_hw = self.process_images(images)

        seq_features, fused_features = self.feature_fusion(image_embed, high_res_feats)
        
        prompt_embed, confidence = self.calculus_peg(seq_features)
        
        sam_masks = self.generate_calculus_masks(prompt_embed, image_embed, high_res_feats, orig_hw)
        
        sam_probs = torch.sigmoid(sam_masks)
        
        refined_masks = self.refine_net(images, sam_probs, fused_features)
        
        return sam_masks, refined_masks, confidence
        
    def load_tooth_features(self, checkpoint_path):
        """Optional method to load pre-trained feature_fusion weights from a tooth checkpoint"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        # Filter for feature_fusion
        fusion_dict = {k.replace('feature_fusion.', ''): v for k, v in state_dict.items() if k.startswith('feature_fusion.')}
        if fusion_dict:
            self.feature_fusion.load_state_dict(fusion_dict, strict=True)
