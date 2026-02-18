"""
Projection models for IPAdapter FaceID variants.
Contains the specialized projection architectures for FaceID models.
"""

import torch
import torch.nn as nn
from typing import Optional


class FeedForward(nn.Module):
    """Feed-forward network used in perceiver attention."""
    
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverAttention(nn.Module):
    """Perceiver-style cross-attention for combining embeddings."""
    
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: context tensor
            latents: query tensor
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.reshape(b, l, self.heads, -1).transpose(1, 2)
        k = k.reshape(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)

        # Attention
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out)


class FacePerceiverResampler(nn.Module):
    """
    Perceiver-based resampler for FaceID Plus models.
    Combines face embeddings with CLIP image embeddings via cross-attention.
    """
    
    def __init__(
        self,
        *,
        dim: int = 768,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        embedding_dim: int = 1280,
        output_dim: int = 768,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: Face embedding tokens (B, num_face_tokens, dim)
            x: CLIP image embeddings (B, num_image_tokens, embedding_dim)
        """
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class FaceIDProjectionModel(nn.Module):
    """
    Basic projection model for standard FaceID models.
    Projects face embeddings to the cross-attention dimension.
    """
    
    def __init__(
        self, 
        cross_attention_dim: int = 768, 
        id_embeddings_dim: int = 512, 
        num_tokens: int = 4
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_embeds: Face embeddings (B, id_embeddings_dim)
            
        Returns:
            Projected embeddings (B, num_tokens, cross_attention_dim)
        """
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class FaceIDPlusProjectionModel(nn.Module):
    """
    Advanced projection model for FaceID Plus models.
    Combines face embeddings with CLIP image embeddings via perceiver resampler.
    """
    
    def __init__(
        self, 
        cross_attention_dim: int = 768, 
        id_embeddings_dim: int = 512, 
        clip_embeddings_dim: int = 1280, 
        num_tokens: int = 4
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        # Face embedding projection
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

        # Perceiver resampler for combining with CLIP embeddings
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(
        self, 
        id_embeds: torch.Tensor, 
        clip_embeds: torch.Tensor, 
        scale: float = 1.0, 
        shortcut: bool = False
    ) -> torch.Tensor:
        """
        Args:
            id_embeds: Face embeddings (B, id_embeddings_dim)
            clip_embeds: CLIP image embeddings (B, seq_len, clip_embeddings_dim)
            scale: Scaling factor for FaceID v2 models
            shortcut: Whether to use shortcut connection (for FaceID v2)
            
        Returns:
            Combined embeddings (B, num_tokens, cross_attention_dim)
        """
        # Project face embeddings
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        
        # Combine with CLIP embeddings via perceiver resampler
        out = self.perceiver_resampler(x, clip_embeds)
        
        # Apply shortcut connection for FaceID v2
        if shortcut:
            out = x + scale * out
            
        return out


def create_faceid_projection_model(
    model_state_dict: dict,
    cross_attention_dim: int = 768,
    clip_embeddings_dim: int = 1280,
    is_sdxl: bool = False,
    is_plus: bool = False
) -> nn.Module:
    """
    Factory function to create appropriate FaceID projection model based on state dict.
    
    Args:
        model_state_dict: State dictionary from the model checkpoint
        cross_attention_dim: Cross-attention dimension
        clip_embeddings_dim: CLIP embeddings dimension  
        is_sdxl: Whether this is an SDXL model
        is_plus: Whether this is a Plus model
        
    Returns:
        Appropriate projection model instance
    """
    # Determine model type from state dict structure
    has_perceiver = any("perceiver_resampler" in key for key in model_state_dict.keys())
    
    if is_plus and has_perceiver:
        # FaceID Plus model with perceiver resampler
        model = FaceIDPlusProjectionModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,  # Standard face embedding size
            clip_embeddings_dim=clip_embeddings_dim,
            num_tokens=16 if is_plus else 4
        )
    else:
        # Standard FaceID model
        model = FaceIDProjectionModel(
            cross_attention_dim=cross_attention_dim,
            id_embeddings_dim=512,  # Standard face embedding size
            num_tokens=16 if is_plus else 4
        )
    
    return model