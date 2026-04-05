import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
from einops import rearrange
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
        # self.ff = nn.Sequential(
        #     nn.Linear(dim, dim_feedforward),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, dim),
        #     nn.Dropout(dropout)
        # )

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # MultiheadAttention 期望 (Query, Key, Value)
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=src_key_padding_mask)
        # x = x + attn_output
        # x = x + self.ff(self.norm2(x))
        # 第一个残差连接
        x = x + attn_output
        
        # --- 2. 前馈网络部分 (Post-LN) ---
        # a. 暂存残差连接的输入
        x_ffn_in = x
        
        # b. 通过FFN
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        # c. 第二个残差连接
        x = x + x_ffn_in
        
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

class ResnetTransformerDualTokensTemporalSpatialDecouplesize(nn.Module):
    def __init__(self, num_classes=18, d_model_cnn=512, 
                 nhead=8, num_layers=6, dropout=0.1, spatial_resolution=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # --- CNN 部分 ---
        # 【【【核心修改 1】】】 将空间尺寸定义为类属性，避免硬编码
        self.spatial_size = spatial_resolution # 调整尺寸
        self.num_patch_tokens = self.spatial_size * self.spatial_size  # 5x5 = 25

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model_cnn, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((self.spatial_size, self.spatial_size))
        
        # --- Transformer 部分 ---
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, d_model_cnn))

        #【核心修改 3】】】 空间位置编码适配新的 patch token 数量 (25)
        self.pos_encoder_patch = nn.Parameter(torch.randn(1, self.num_patch_tokens, d_model_cnn))
        # 时序位置编码
        # self.pos_encoder_temporal = PositionalEncoding(d_model_cnn, dropout, max_len=16)
        self.pos_encoder_temporal = PositionalEncoding(d_model_cnn, dropout, max_len=32)

        # 堆叠交错的时空块
        self.transformer_blocks = nn.ModuleList(
            [InterleavedSpatioTemporalBlock(d_model_cnn, nhead, d_model_cnn * 4, dropout) for _ in range(num_layers)] # 看看512和2048的组合是否可行
        )
        
        # --- 分类头 ---
        self.cls_head_norm = nn.LayerNorm(d_model_cnn)
        self.cls_head_projection = nn.Linear(d_model_cnn, 1)

    def forward(self, x, labels_18cls, doppler_mask=None):
        b, t, c, h_in, w_in = x.shape

        # --- 1. CNN 特征提取 ---
        x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
         # with torch.no_grad():
        feature_maps = self.cnn_backbone(x_reshaped)
        projected_maps = self.projection(feature_maps)
        pooled_maps = self.pool(projected_maps) # (B*T, 512, 4, 4)


        # --- 2. 【分支1】为损失计算准备 raw_strings ---
        raw_strings_for_loss = rearrange(pooled_maps.detach(), '(b t) c h w -> b (h w) t c', t=t)
        # --- 3. 准备 Transformer 初始输入 ---
        # a. Patch Tokens: (B*T, C, H*W) -> (B, T, H*W, C)
        patch_tokens = rearrange(pooled_maps, '(b t) c h w -> b t (h w) c', t=t)

        # b. Class Tokens: (1, 18, C) -> (B, T, 18, C)
        cls_tokens = self.class_tokens.unsqueeze(1).expand(b, t, -1, -1)
        
        # --- 4. 【核心修正】在循环前，一次性注入所有位置编码 ---
        # a. 添加空间位置编码
        # self.pos_encoder_patch (1, 16, C) 会被广播到 (B, T, 16, C)
        patch_tokens = patch_tokens + self.pos_encoder_patch.unsqueeze(1)
        
        # b. 拼接 [CLS] 和 Patch
        initial_x = torch.cat([cls_tokens, patch_tokens], dim=2) # (B, T, 34, C)

        # c. 添加时序位置编码
        s = initial_x.shape[2] # 序列长度 S (34)
        x_temp_for_pe = rearrange(initial_x, 'b t s c -> (b s) t c')
        x_temp_for_pe = self.pos_encoder_temporal(x_temp_for_pe)
        current_x = rearrange(x_temp_for_pe, '(b s) t c -> b t s c', s=s)

        # --- 4. 核心：交错式 Transformer 编码 ---
        all_layer_cls_tokens = []
        all_layer_spatial_scores = []
        
        # 【新增】用于累加解耦损失的变量
        total_decouple_loss = torch.tensor(0.0, device=x.device)

        for block in self.transformer_blocks:
            s = current_x.shape[2] # 序列长度 (34)
            x_in_block = current_x.clone()
            
            # a. 时序注意力
            # (B, T, S, C) -> (B*S, T, C)
            x_temp = rearrange(current_x, 'b t s c -> (b s) t c')
            # x_temp = self.pos_encoder_temporal(x_temp)
            x_temp = block.temporal_transformer(x_temp)
            current_x = rearrange(x_temp, '(b s) t c -> b t s c', s=s)
            # current_x = current_x + x_in_block # 第一个残差连接

            # b. 空间注意力
            # (B, T, S, C) -> (B*T, S, C)
            x_spat = rearrange(current_x, 'b t s c -> (b t) s c')
            # 添加空间位置编码 (只给patch部分)
            # x_spat[:, self.num_classes:, :] += self.pos_encoder_patch

            # --- 【核心修改】构建并应用空间掩码 ---
            # 我们需要一个 (B*T, S) 的掩码
            # doppler_mask 是 (B, 16)，对应16个“串”Token (Patch Tokens)
            # 我们需要把它扩展到 (B, T, 16)，然后再 reshape
            
            if doppler_mask is not None:
                cls_mask = torch.zeros(b, t, self.num_classes, dtype=torch.bool, device=x.device)
                patch_mask = doppler_mask.unsqueeze(1).expand(-1, t, -1)
                spatial_padding_mask = torch.cat([cls_mask, patch_mask], dim=2)
                spatial_padding_mask = rearrange(spatial_padding_mask, 'b t s -> (b t) s')
            else:
                spatial_padding_mask = None

            # 5. 将掩码传递给空间Transformer
            x_spat = block.spatial_transformer(x_spat, src_key_padding_mask=spatial_padding_mask)
            current_x = rearrange(x_spat, '(b t) s c -> b t s c', t=t)
            # current_x = current_x + x_in_block # 第二个残差连接 (是否需要？可以实验)
                    
            # --- 【核心】c. 记录当前层的空间注意力分数 ---
            # 1. 提取当前层输出的 cls 和 patch tokens
            current_cls_temporal = current_x[:, :, :self.num_classes, :]
            current_patch_temporal = current_x[:, :, self.num_classes:, :]

            # import pdb; pdb.set_trace()
            # 2. 在时间维度上聚合，得到用于空间分析的“摘要”Token
            current_cls_spatial = current_cls_temporal.mean(dim=1) # (B, 18, C)
            current_patch_spatial = current_patch_temporal.mean(dim=1) # (B, 16, C)

            # 【核心修正 1：防止 FP16 下除零溢出】强制设置 eps=1e-6 
            norm_tokens = F.normalize(current_cls_spatial, p=2, dim=-1, eps=1e-6)
            
            similarity_matrix = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))
            ground_truth = torch.arange(self.num_classes, device=x.device)

            # 在批次维度上重复，然后计算交叉熵
            loss_decouple_layer_i = F.cross_entropy(
                similarity_matrix.view(-1, self.num_classes),
                ground_truth.repeat(b)
            )
            # 3. 累加到总解耦损失中
            total_decouple_loss += loss_decouple_layer_i
            # 4. 计算 cls-to-patch 的注意力分数并保存
            # scores = torch.einsum('bic,bpc->bip', current_cls_spatial, current_patch_spatial)
            # print('scores',scores[0])
            
            with torch.autocast(device_type='cuda', enabled=False):
                scale_factor = current_cls_spatial.size(-1) ** 0.5
                # 步骤A: 强制提取至 FP32 精度域
                cls_fp32 = current_cls_spatial.float()
                patch_fp32 = current_patch_spatial.float()
                
                # 步骤B: 必须在点积前除以 scale_factor，抑制乘法结果级数
                cls_scaled = cls_fp32 / scale_factor
                
                # 步骤C: 在 FP32 环境下执行爱因斯坦求和
                scores = torch.einsum('bic,bpc->bip', cls_scaled, patch_fp32)
            # --- ▲▲▲ 修正结束 ▲▲▲ ---

        all_layer_spatial_scores.append(scores)
        
        # 4. 同时保存当前层的 CLS Tokens (聚合后)
        all_layer_cls_tokens.append(current_cls_spatial)

        # --- 5. 提取最终信息用于分类 ---
        final_output_cls_tokens = all_layer_cls_tokens[-1] # (B, 18, C)

        # --- 6. 分类 ---
        normed_cls_tokens = self.cls_head_norm(final_output_cls_tokens)
        projected_scores = self.cls_head_projection(normed_cls_tokens)
        predicted_scores = projected_scores.squeeze(-1)
        # predicted_scores = final_output_cls_tokens.mean(dim=-1)

        return {
            "predicted_scores": predicted_scores,
            "final_cls_tokens": final_output_cls_tokens,
            "all_layer_cls_tokens": all_layer_cls_tokens, # 列表，每个元素是 (B, 18, C)
            "raw_strings_for_loss": raw_strings_for_loss,
            "all_layer_attention_scores": all_layer_spatial_scores, # 列表，每个元素是 (B, 18, 25)
            # 【核心修改】直接返回计算好的总解耦损失
            "total_decouple_loss": total_decouple_loss,
        }