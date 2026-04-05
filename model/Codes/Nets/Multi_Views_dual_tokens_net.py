import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from einops import rearrange, repeat
from copy import deepcopy
from entmax import EntmaxBisect # 假设您已经安装了这个库


# 定义标准的位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model) 
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


POSITIVE_TO_NEGATIVE_MAP = {
    # 示例:
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5,
    16: 6, 17: 7, 9: 2, 10: 5,
    18: 8,
}

class BasicTransformerBlock(nn.Module):
    """一个标准的Transformer编码器层"""
    def __init__(self, dim, n_heads, dim_feedforward, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, src_key_padding_mask=None):
        # MultiheadAttention 期望 (Query, Key, Value)
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + attn_output
        x = x + self.ff(self.norm2(x))
        return x

# --- 2. 核心：交错式时空块 ---

class InterleavedSpatioTemporalBlock(nn.Module):  # 存在残差问题 需要后续解决
    """
    在每个块内部，先进行时序注意力，再进行空间注意力。
    [CLS] Tokens 全程参与。
    """
    def __init__(self, dim, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.temporal_transformer = BasicTransformerBlock(dim, n_heads, dim_feedforward, dropout)
        self.spatial_transformer = BasicTransformerBlock(dim, n_heads, dim_feedforward, dropout)

    def forward(self, x, t, h, w):
        # 输入 x: (B, S, D)，其中 S = 18 + H*W
        b, s, d = x.shape
        num_cls_tokens = s - (h * w)

        # --- a. 时序建模 ---
        # 变换视角: (B, S, D) -> (B, S, T, D/T) -> (B*S, T, D/T)
        # 这是一个近似，因为D中混合了时序信息，我们无法完美分离
        # 更精确的做法是让输入就是5D的 (B, C, T, H, W)
        # 让我们严格按照维度变换来
        # 输入 x 应该是 (B, C, T, H, W)
        
        # 假设我们传入的是一个5D张量 x: (B, C, T, H, W)
        b, c, t, h, w = x.shape
        x_in = x
        
        # 变换视角进行时序注意力: b c t h w -> (b h w) t c
        x_temporal = rearrange(x, 'b c t h w -> (b h w) t c')
        x_temporal = self.temporal_transformer(x_temporal)
        x_temporal = rearrange(x_temporal, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
        x = x_in + x_temporal # 残差连接
        
        # --- b. 空间建模 ---
        x_in_spatial = x
        # 变换视角进行空间注意力: b c t h w -> (b t) (h w) c
        x_spatial = rearrange(x, 'b c t h w -> (b t) c (h w)')
        x_spatial = x_spatial.permute(0, 2, 1) # -> (b t) (h w) c
        x_spatial = self.spatial_transformer(x_spatial)
        x_spatial = x_spatial.permute(0, 2, 1) # -> (b t) c (h w)
        x_spatial = rearrange(x_spatial, '(b t) c (h w) -> b c t h w', t=t, h=h, w=w)
        x = x_in_spatial + x_spatial # 第二次残差连接
        
        return x

class MultiViewDualTokensFusionSize(nn.Module):
    def __init__(self, view_num_classes=18, d_model=512, 
                 view_nhead=8, view_encoder_layers=6,
                 case_num_classes=4,
                 fusion_layers=2, fusion_nhead = 8,
                 max_views=5, # 支持的最大视角数
                 dropout=0.1,
                 spatial_size=4,):
        super().__init__()
        
        # --- 参数定义 ---
        self.view_num_classes = view_num_classes
        self.case_num_classes = case_num_classes
        self.num_patch_tokens = spatial_size * spatial_size

        # --- 1. 共享CNN特征提取器 ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        
        # --- 2. 视角级编码器组件 ---
        self.cls_tokens = nn.Parameter(torch.randn(1, self.view_num_classes, d_model))
        self.pos_encoder_patch = nn.Parameter(torch.randn(1, self.num_patch_tokens, d_model))
        # self.pos_encoder_temporal = PositionalEncoding(d_model, dropout, max_len=32)
        self.pos_encoder_temporal = PositionalEncoding(d_model, dropout, max_len=32)
        
        # 视角内的时空交错编码器
        self.transformer_blocks = nn.ModuleList(
            [InterleavedSpatioTemporalBlock(d_model, view_nhead, d_model * 4, dropout) for _ in range(view_encoder_layers)]
        )
        
        # --- 3. 跨视角融合编码器组件 ---
        self.case_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
       # 【【【核心修正】】】
        # 正确计算融合阶段位置编码的最大长度
        # 应该是 1 (case_cls) + max_views * 18 (所有视角的所有cls_token)
        max_fusion_sequence_len = 1 + max_views * self.view_num_classes
        self.view_pos_embedding = nn.Parameter(torch.randn(1, max_fusion_sequence_len, d_model))
        
        # 专门用于融合的Transformer
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model, fusion_nhead, d_model * 4, dropout, batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=fusion_layers)
        
        # --- 4. 最终分类头 ---
        self.case_head_norm = nn.LayerNorm(d_model)
        self.case_head_projection = nn.Linear(d_model, self.case_num_classes)

    def forward(self, x_case, num_views_per_sample, masks):
        b, v, t, c, h, w = x_case.shape
        # print('x_case',x_case.max(),x_case.min())
        # --- a. 并行单视角编码 ---
        
        # 1. CNN特征提取: (B,V,T,C,H,W)->(B*V,T,16,D)
        x_reshaped = rearrange(x_case, 'b v t c h w -> (b v t) c h w')
        pooled_maps = self.pool(self.projection(self.cnn_backbone(x_reshaped)))
        patch_tokens = rearrange(pooled_maps, '(b v t) d h w -> (b v) t (h w) d', b=b, v=v, t=t)

        patch_tokens = patch_tokens + self.pos_encoder_patch
        view_cls_tokens_expanded = self.cls_tokens.expand(b*v, -1, -1).unsqueeze(1).expand(-1, t, -1, -1)
        current_sequence = torch.cat([view_cls_tokens_expanded, patch_tokens], dim=2)
        
        # --- ▼▼▼【核心修正：正确处理传入的3D mask】▼▼▼ ---
        spatial_attn_mask = None
        if masks is not None:
            # masks 输入形状: (B, V, NumPatches)
            
            # 1. 我们需要将其 reshape 以匹配 (B*V,) 的批次维度
            # (B, V, NumPatches) -> (B*V, NumPatches)
            masks_reshaped_bv = rearrange(masks, 'b v s -> (b v) s')
            
            # 2. 【重要】: 空间注意力作用于 T 个时间步上，所以我们需要将这个mask复制T次
            # (B*V, NumPatches) -> (B*V, T, NumPatches) -> (B*V*T, NumPatches)
            masks_reshaped_bvt = repeat(masks_reshaped_bv, 'bv s -> (bv t) s', t=t)
            
            # 3. 为 18 个 [CLS] token 添加 False (不mask)
            cls_mask = torch.zeros(b*v*t, self.view_num_classes, dtype=torch.bool, device=x_case.device)
            
            # 4. 最终的空间注意力 mask
            # 形状: (B*V*T, 18 + num_patches)
            spatial_attn_mask = torch.cat([cls_mask, masks_reshaped_bvt], dim=1)

        # 3. 通过时空交错编码器
        s_inner = current_sequence.shape[2] # 34
        for block in self.transformer_blocks:
            # 时序注意力
            x_for_temporal = rearrange(current_sequence, 'bv t s d -> (bv s) t d')
            x_for_temporal = self.pos_encoder_temporal(x_for_temporal)
            temporal_out = block.temporal_transformer(x_for_temporal)
            current_sequence = rearrange(temporal_out, '(bv s) t d -> bv t s d', s=s_inner)
            
            # 空间注意力
            x_for_spatial = rearrange(current_sequence, 'bv t s d -> (bv t) s d')
            spatial_out = block.spatial_transformer(x_for_spatial, src_key_padding_mask=spatial_attn_mask)
            current_sequence = rearrange(spatial_out, '(bv t) s d -> bv t s d', t=t)
            # print('current_sequence',current_sequence[0])
        # --- b. 单视角特征提炼 ---
        
        # 1. 提取最终的 view_cls_tokens
        # (B*V, T, 34, D) -> (B*V, T, 18, D)
        final_view_cls = current_sequence[:, :, :self.view_num_classes, :]
        
        # 【【【核心修改】】】
        # 2. 不再对18个token求平均，而是只在时间维度(T)上求平均
        # (B*V, T, 18, D) -> (B*V, 18, D)
        view_features_set = final_view_cls.mean(dim=1)
        
        # --- c. 跨视角融合 ---
        # 【【【核心修改】】】
        # 1. 将每个视角的18个token直接展平拼接，形成一个长序列
        # (B*V, 18, D) -> (B, V * 18, D)
        view_features_sequence = rearrange(view_features_set, '(b v) s d -> b (v s) d', b=b)
        
        # 2. 拼接病例级[CLS] token
        # (B, 1, D)
        case_cls = self.case_cls_token.expand(b, -1, -1)
        # (B, 1 + V*18, D)
        fusion_sequence = torch.cat([case_cls, view_features_sequence], dim=1)
        
        # 3. 添加视角/token位置编码
        # 注意：view_pos_embedding的尺寸需要足够大以覆盖所有token
        num_fusion_tokens = fusion_sequence.shape[1]
        fusion_sequence = fusion_sequence + self.view_pos_embedding[:, :num_fusion_tokens, :]
        
        # 4. 构建padding mask (逻辑更复杂，需要按token屏蔽)
        fusion_mask = None
        if num_views_per_sample is not None:
            # (B, V*18)
            view_token_mask = repeat(
                torch.arange(v, device=x_case.device) >= num_views_per_sample.unsqueeze(1),
                'b v -> b (v s)', s=self.view_num_classes
            )
            cls_padding_mask = torch.zeros(b, 1, dtype=torch.bool, device=x_case.device)
            # (B, 1 + V*18)
            fusion_mask = torch.cat([cls_padding_mask, view_token_mask], dim=1)
        
        # 5. 通过融合Transformer
        fused_output = self.fusion_transformer(fusion_sequence, src_key_padding_mask=fusion_mask)
        
        # --- d. 最终分类 ---
        final_case_token = fused_output[:, 0, :]
        logits = self.case_head_projection(self.case_head_norm(final_case_token))
        
        return logits