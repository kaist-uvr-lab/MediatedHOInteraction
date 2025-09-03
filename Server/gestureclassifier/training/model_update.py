import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import math


class Attention(nn.Module):
    def __init__(self, attention_dim, gru_dim=128):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(gru_dim, gru_dim, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output


# ----------------------------------------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Self-Attention mechanism for sequence modeling"""

    def __init__(self, hidden_dim, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Linear layers for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Generate Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)  # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, seq_len, hidden_dim]

        # Output projection with residual connection
        output = self.output(context) + x
        output = self.norm(output)

        return output


# ----------------------------------------------------------------------------------------------------------------------
class EnhancedDeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Graph Convolutional Network
        # Input: 78 dimensions = 21*3 coords + 15 angles
        self.gcn = HandMANOGCN(num_features, gcn_output_dim)

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(gcn_output_dim, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(256)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )

        # Streamlined classifier (balance between depth and speed)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),  # avg + max + attention pooling
            nn.Dropout(0.3),
            nn.Linear(256 * 3, 256, bias=False),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes, bias=False)
        )

        # # 수정: 15클래스에 최적화된 3층 분류기
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(256 * 3),
        #     nn.Dropout(0.35),
        #     nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
        #     nn.BatchNorm1d(384),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(384, 192),  # 점진적 감소
        #     nn.BatchNorm1d(192),
        #     nn.GELU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
        #     nn.BatchNorm1d(64),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 15)  # 최종 15클래스
        # )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # Apply GCN
        x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_gcn)
        output = self.gru_dropout(self.gru_norm(output))

        # Multi-head self-attention (no masking needed for fixed length)
        attn_output, _ = self.multihead_attention(output, output, output)
        attn_output = self.attention_norm(attn_output + output)  # Residual connection

        # Feature enhancement with skip connection
        enhanced = self.feature_enhance(attn_output)
        enhanced = enhanced + attn_output  # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]

        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]

        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v2(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21):
        super(EnhancedDeepGRU_v2, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints


        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(256)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # Apply GCN
        x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_gcn)
        output = self.gru_dropout(self.gru_norm(output))

        # Multi-head self-attention (no masking needed for fixed length)
        attn_output, _ = self.multihead_attention(output, output, output)
        attn_output = self.attention_norm(attn_output + output)  # Residual connection

        # Feature enhancement with skip connection
        enhanced = self.feature_enhance(attn_output)
        enhanced = enhanced + attn_output  # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]

        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]

        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedDeepGRU_v3(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v3, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # 3. *** Spatial Multi-head Self-Attention ***
        spatial_attn_output, _ = self.spatial_attention(output, output, output)
        spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        # temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(spatial_output)
        enhanced = enhanced + spatial_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v4(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v4, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # # Lightweight Multi-head Attention (2 heads)
        # self.spatial_attention = nn.MultiheadAttention(
        #     embed_dim=256,
        #     num_heads=2,
        #     dropout=0.15,
        #     batch_first=True
        # )
        # self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # # 3. *** Spatial Multi-head Self-Attention ***
        # spatial_attn_output, _ = self.spatial_attention(output, output, output)
        # spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(temporal_output)
        enhanced = enhanced + temporal_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v5(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v5, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # # Lightweight Multi-head Attention (2 heads)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # # Feature enhancement (lightweight)
        # self.feature_enhance = nn.Sequential(
        #     nn.Linear(256, 192, bias=False),
        #     nn.GELU(),
        #     nn.Dropout(0.15),
        #     nn.Linear(192, 256, bias=False)
        # )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # 3. *** Spatial Multi-head Self-Attention ***
        spatial_attn_output, _ = self.spatial_attention(output, output, output)
        spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        temporal_output = self.temporal_attention(spatial_output)  # [batch_size, 10, 256]

        # # 5. Feature enhancement with skip connection
        # enhanced = self.feature_enhance(temporal_output)
        # enhanced = enhanced + temporal_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = temporal_output.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(temporal_output)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(temporal_output * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v6(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v6, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output, _ = self.gru2(output)
        output = self.gru_dropout(self.gru_norm(output))

        # 3. *** Spatial Multi-head Self-Attention ***
        spatial_attn_output, _ = self.spatial_attention(output, output, output)
        spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(temporal_output)
        enhanced = enhanced + temporal_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v7(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EnhancedDeepGRU_v7, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 256, 2, batch_first=True)
        self.gru2 = nn.GRU(256, 128, 2, batch_first=True)
        # self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128*2),
            nn.Dropout(0.5),
            nn.Linear(128*2, 192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, hidden = self.gru2(output)
        # output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedDeepGRU_v8(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EnhancedDeepGRU_v8, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        # self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(256, 256)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256*2),
            nn.Dropout(0.5),
            nn.Linear(256*2, 192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, hidden = self.gru2(output)
        # output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v3_1(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v3_1, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 256, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(512)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.spatial_attention_norm = nn.LayerNorm(512)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=512, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 512, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.GELU(),
            nn.Linear(256, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(512 * 3),
            nn.Dropout(0.35),
            nn.Linear(512 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # 3. *** Spatial Multi-head Self-Attention ***
        spatial_attn_output, _ = self.spatial_attention(output, output, output)
        spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        # temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(spatial_output)
        enhanced = enhanced + spatial_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedDeepGRU_v3_2(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v3_2, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 256, 2, batch_first=True, dropout=0.2)
        self.gru2 = nn.GRU(256, 256, 2, batch_first=True, dropout=0.1)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Lightweight Multi-head Attention (2 heads)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=2,
            dropout=0.15,
            batch_first=True
        )
        self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output, _ = self.gru2(output)
        output = self.gru_dropout(self.gru_norm(output))

        # 3. *** Spatial Multi-head Self-Attention ***
        spatial_attn_output, _ = self.spatial_attention(output, output, output)
        spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        # temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(spatial_output)
        enhanced = enhanced + spatial_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedDeepGRU_v4_2(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v4_2, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 256, 2, batch_first=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # # Lightweight Multi-head Attention (2 heads)
        # self.spatial_attention = nn.MultiheadAttention(
        #     embed_dim=256,
        #     num_heads=2,
        #     dropout=0.15,
        #     batch_first=True
        # )
        # self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # # 3. *** Spatial Multi-head Self-Attention ***
        # spatial_attn_output, _ = self.spatial_attention(output, output, output)
        # spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(temporal_output)
        enhanced = enhanced + temporal_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v4_1(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v4_1, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # # Lightweight Multi-head Attention (2 heads)
        # self.spatial_attention = nn.MultiheadAttention(
        #     embed_dim=256,
        #     num_heads=2,
        #     dropout=0.15,
        #     batch_first=True
        # )
        # self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.max_pool = nn.AdaptiveMaxPool1d(1)
        #
        # # Learnable attention pooling (more efficient for fixed length)
        # self.attention_pool = nn.Sequential(
        #     nn.Linear(256, 64, bias=False),
        #     nn.GELU(),
        #     nn.Linear(64, 1, bias=False)
        # )

        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.35),
            nn.Linear(256, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 15)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        # # 3. *** Spatial Multi-head Self-Attention ***
        # spatial_attn_output, _ = self.spatial_attention(output, output, output)
        # spatial_output = self.spatial_attention_norm(spatial_attn_output + output)  # Residual

        # 4. *** 새로 추가: Temporal Self-Attention ***
        # 시간적 의존성을 더 잘 캡처하기 위한 추가 attention layer
        temporal_output = self.temporal_attention(output)  # [batch_size, 10, 256]

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(temporal_output)
        enhanced = enhanced + temporal_output    # Skip connection      # 16 10 256

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # # 2. Max pooling
        # max_pooled = self.max_pool(enhanced_transposed).squeeze(-1)  # [batch_size, 256]
        # # 3. Learnable attention pooling (no masking needed)
        # attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        # attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        # attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        # pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(avg_pooled)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU_v4_3(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=128):
        super(EnhancedDeepGRU_v4_3, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(128)
        self.gru_dropout = nn.Dropout(0.2)

        # # Lightweight Multi-head Attention (2 heads)
        # self.spatial_attention = nn.MultiheadAttention(
        #     embed_dim=256,
        #     num_heads=2,
        #     dropout=0.15,
        #     batch_first=True
        # )
        # self.spatial_attention_norm = nn.LayerNorm(256)

        # Temporal attention 추가
        # self.temporal_attention = SelfAttention(hidden_dim=256, dropout=0.1)

        self.attention = Attention(128)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )

        # # Efficient pooling for fixed sequence length (10)
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.max_pool = nn.AdaptiveMaxPool1d(1)
        #
        # # Learnable attention pooling (more efficient for fixed length)
        # self.attention_pool = nn.Sequential(
        #     nn.Linear(256, 64, bias=False),
        #     nn.GELU(),
        #     nn.Linear(64, 1, bias=False)
        # )


        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.Linear(256, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  # 최종 15클래스
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_padding_mask(self, x, lengths):
        """Create padding mask for variable length sequences"""
        batch_size, max_len = x.shape[0], x.shape[1]
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).expand(batch_size, max_len, max_len).to(x.device)

    def forward(self, x_padded, lengths=None):
        """
        Args:
            x_padded: [batch_size, 10, 78] - Fixed sequence length of 10
            lengths: Not needed for fixed length sequences
        """
        batch_size = x_padded.shape[0]

        # # Apply GCN
        # x_gcn = self.gcn(x_padded)

        # Bidirectional GRU (no packing needed for fixed length)
        output, hidden = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        attn_output = self.attention(output, hidden[-1:])

        # 5. Feature enhancement with skip connection
        enhanced = self.feature_enhance(attn_output)
        enhanced = enhanced + attn_output    # Skip connection

        # Efficient pooling for fixed sequence length
        # Transpose for pooling: [batch_size, hidden_dim, seq_len]
        # enhanced_transposed = enhanced.transpose(1, 2)  # [batch_size, 256, 10]
        #
        # # 1. Average pooling
        # avg_pooled = self.adaptive_pool(enhanced).squeeze(-1)  # [batch_size, 256]
        # # 2. Max pooling
        # max_pooled = self.max_pool(enhanced).squeeze(-1)  # [batch_size, 256]
        # # 3. Learnable attention pooling (no masking needed)
        # attention_scores = self.attention_pool(enhanced)  # [batch_size, 10, 1]
        # attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        # attention_pooled = torch.sum(enhanced * attention_weights, dim=1)  # [batch_size, 256]
        #
        # # Combine all pooling strategies
        # pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(enhanced)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
# Example usage and model information
def create_fast_model(num_features=78, num_classes=15, num_joints=21, gcn_output_dim=128):
    """
    Create a fast enhanced DeepGRU model with self-attention for real-time gesture recognition

    Args:
        num_features: Total input features (78 = 21*3 coords + 15 angles)
        num_classes: Number of gesture classes to predict
        num_joints: Number of hand joints (default 21 for MANO model)
        gcn_output_dim: Output dimension for GCN layer
    """
    model = EnhancedDeepGRU_v4(
        num_features=num_features,
        num_classes=num_classes,
        # num_joints=num_joints,
        # gcn_output_dim=gcn_output_dim
    )

    print(f"Fast model with self-attention created with {model.get_num_params():,} parameters")
    return model


def create_model(num_features=78, num_classes=15, num_joints=21, gcn_output_dim=128):
    """
    Backward compatibility - creates fast model by default
    """
    return create_fast_model(num_features, num_classes, num_joints, gcn_output_dim)

