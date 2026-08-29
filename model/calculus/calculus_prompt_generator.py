import torch
import torch.nn as nn
import math

class CalculusPEG(nn.Module):
    """Simplified prompt generator for calculus"""
    def __init__(self, d_model=256, nhead=8, num_queries=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.kaiming_uniform_(self.query_embed, a=math.sqrt(5))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.label_head = nn.Linear(d_model, 1)

    def forward(self, image_embed):
        """
        Args:
            image_embed: [B, H*W, C] - fused image embeddings
        Returns:
            prompt_embed: [B, 4, 256] - generated prompt embeddings
            confidence: [B, 4] - calculus presence confidence
        """
        B = image_embed.shape[0]
        device = image_embed.device
        
        query_embed = self.query_embed.to(device)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)

        prompt_embed = self.transformer_decoder(query_embed, image_embed)
        confidence = self.label_head(prompt_embed).squeeze(-1)
        
        return prompt_embed, confidence
