import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from .ttt_block import TTT

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = TTT(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, state=None, learning_rate=1.0, return_grad_norm=False):
        b, n, c = x.shape
        h = w = int(n ** 0.5)
        # CPE positional embedding
        res = self.cpe(x.reshape(b, h, w, c).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        x = x + res
        
        # TTT update (where memory is formed)
        attn_out, new_state, metrics = self.attn(self.norm1(x), h, w, state=state, 
                                               learning_rate=learning_rate, 
                                               return_grad_norm=return_grad_norm)
        # Note: metrics is a dictionary here
            
        x = x + self.drop_path(attn_out)
        
        # MLP processing
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x, new_state, metrics

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, states=None, learning_rate=1.0, return_grad_norm=False):
        """
        states: List of dicts, one for each block in the transformer depth.
        """
        x = self.patch_embed(x)
        
        new_states = []
        all_metrics = []
        
        for i, block in enumerate(self.blocks):
            block_state = states[i] if states else None
            x, block_new_state, block_metrics = block(x, state=block_state, 
                                                    learning_rate=learning_rate, 
                                                    return_grad_norm=return_grad_norm)
            new_states.append(block_new_state)
            all_metrics.append(block_metrics)
            
        x = self.norm(x)
        features = x.mean(dim=1) # Global average pooling
        logits = self.head(features)
        
        return logits, features, new_states, all_metrics

def vittt_tiny(pretrained=False, **kwargs):
    model = VisionTransformer(embed_dim=192, depth=12, num_heads=3, **kwargs)
    return model

def vittt_small(pretrained=False, **kwargs):
    model = VisionTransformer(embed_dim=384, depth=12, num_heads=6, **kwargs)
    return model

def vittt_base(pretrained=False, **kwargs):
    model = VisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)
    return model
