import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import math


# ----------------------------------------------------------------------------------------------------------------------
class GraphConvLayer(nn.Module):
    """Single Graph Convolutional Layer"""

    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        # Graph convolution
        output = torch.matmul(adj, support)
        return output + self.bias


# ----------------------------------------------------------------------------------------------------------------------
class HandMANOGCN(nn.Module):
    """Lightweight Graph Convolutional Network based on MANO hand model topology"""

    def __init__(self, input_dim, output_dim):
        super(HandMANOGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # MANO hand model adjacency matrix (21 joints)
        self.register_buffer('adjacency_matrix', self._create_mano_adjacency())

        # Single GCN layer operating on joint coordinates
        self.gcn = GraphConvLayer(3, output_dim)  # 3D coordinates per joint

        # Angle processing branch
        self.angle_processor = nn.Sequential(
            nn.Linear(15, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

        # Fusion layer
        self.fusion = nn.Linear(output_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def _create_mano_adjacency(self):
        """Create adjacency matrix based on MANO hand topology (21 joints)"""
        adj = torch.zeros(21, 21)

        # Wrist connections to finger bases
        adj[0, [1, 5, 9, 13, 17]] = 1
        adj[[1, 5, 9, 13, 17], 0] = 1

        # Finger bone connections
        fingers = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
        for start, end in fingers:
            for i in range(start, end):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1

        # Add self-connections
        adj += torch.eye(21)

        # Normalize adjacency matrix
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-6)

        return adj

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, 78] where 78 = 21*3 + 15
        Returns:
            Output features [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Split joint coordinates and angles
        joint_coords = x[:, :, :63]  # 21*3 = 63
        joint_angles = x[:, :, 63:]  # 15 angles

        # Process joint coordinates through GCN
        joint_coords_reshaped = joint_coords.view(batch_size * seq_len, 21, 3)
        joint_features = self.gcn(joint_coords_reshaped, self.adjacency_matrix)
        joint_features_pooled = joint_features.mean(dim=1)  # Global pooling

        # Process joint angles
        joint_angles_flat = joint_angles.view(batch_size * seq_len, 15)
        angle_features = self.angle_processor(joint_angles_flat)

        # Fuse features
        combined = torch.cat([joint_features_pooled, angle_features], dim=1)
        fused = self.fusion(combined)
        fused = self.norm(F.gelu(fused))

        # Reshape back
        output = fused.view(batch_size, seq_len, self.output_dim)
        return output


# ----------------------------------------------------------------------------------------------------------------------
class FastMultiHeadAttention(nn.Module):
    """Optimized 4-Head Attention mechanism for speed"""

    def __init__(self, attention_dim, num_heads=4):
        super(FastMultiHeadAttention, self).__init__()
        assert attention_dim % num_heads == 0

        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Single linear layer for Q, K, V computation
        self.qkv = nn.Linear(attention_dim, attention_dim * 3, bias=False)
        self.output = nn.Linear(attention_dim, attention_dim)

        # Simplified GRU
        self.gru = nn.GRU(attention_dim, attention_dim, 1, batch_first=True, bias=False)
        self.norm = nn.LayerNorm(attention_dim)

    def forward(self, input_padded, hidden):
        """
        Args:
            input_padded: [batch_size, seq_len, attention_dim]
            hidden: [1, batch_size, attention_dim]
        """
        batch_size, seq_len, _ = input_padded.shape

        # Expand hidden: [batch_size, 1, attention_dim]
        hidden_expanded = hidden.transpose(0, 1)

        # Generate Q from hidden, K,V from input
        qkv_hidden = self.qkv(hidden_expanded)
        qkv_input = self.qkv(input_padded)

        # Split Q, K, V
        Q = qkv_hidden[:, :, :self.attention_dim]
        K = qkv_input[:, :, self.attention_dim:2 * self.attention_dim]
        V = qkv_input[:, :, 2 * self.attention_dim:]

        # Reshape for multi-head: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.attention_dim)

        # Output projection with residual
        context = self.output(context) + hidden_expanded
        context = self.norm(context)

        # GRU processing
        aux_context, _ = self.gru(context, hidden)

        # Combine contexts
        output = torch.cat([aux_context, context], dim=2).squeeze(1)
        return output


# ----------------------------------------------------------------------------------------------------------------------
class EnhancedDeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, num_joints=21, gcn_output_dim=64, num_heads=4):
        super(EnhancedDeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_joints = num_joints

        # Graph Convolutional Network
        # Input: 78 dimensions = 21*3 coords + 15 angles
        self.gcn = HandMANOGCN(num_features, gcn_output_dim)

        # GRU Encoder (reduced for speed)
        self.gru1 = nn.GRU(gcn_output_dim, 256, 1, batch_first=True, bias=False)
        self.gru2 = nn.GRU(256, 128, 1, batch_first=True, bias=False)

        # Multi-Head Attention
        self.attention = FastMultiHeadAttention(128, num_heads)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes, bias=False)
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

    def forward(self, x_padded):
        """
        Args:
            x_padded: [batch_size, seq_len, 78]
        """
        # Apply GCN
        x_gcn = self.gcn(x_padded)

        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        # Pack sequences
        x_packed = packer(x_gcn, x_lengths, batch_first=True)

        # GRU encoding
        output, _ = self.gru1(x_packed)
        output, hidden = self.gru2(output)

        # Unpack for attention
        output_padded, _ = padder(output, batch_first=True)

        # Apply attention
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classification
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
# Example usage and model information
def create_fast_model(num_features=78, num_classes=15, num_joints=21, gcn_output_dim=64, num_heads=4):
    """
    Create a fast enhanced DeepGRU model for real-time gesture recognition

    Args:
        num_features: Total input features (78 = 21*3 coords + 15 angles)
        num_classes: Number of gesture classes to predict
        num_joints: Number of hand joints (default 21 for MANO model)
        gcn_output_dim: Output dimension for GCN layer
        num_heads: Number of attention heads (default 4 for speed)
    """
    model = EnhancedDeepGRU(
        num_features=num_features,
        num_classes=num_classes,
        num_joints=num_joints,
        gcn_output_dim=gcn_output_dim,
        num_heads=num_heads
    )

    print(f"Fast model created with {model.get_num_params():,} parameters")
    return model


def create_model(num_features=78, num_classes=15, num_joints=21, gcn_output_dim=64, num_heads=4):
    """
    Backward compatibility - creates fast model by default
    """
    return create_fast_model(num_features, num_classes, num_joints, gcn_output_dim, num_heads)


# ----------------------------------------------------------------------------------------------------------------------
# Test function to verify the model works
def test_model():
    """Test function to verify model functionality"""
    print("Testing Enhanced DeepGRU model...")

    # Create model
    model = create_fast_model(num_features=78, num_classes=10)
    model.eval()

    # Test data
    batch_size = 4
    max_seq_len = 20

    # Random input: [batch_size, seq_len, 78]
    x = torch.randn(batch_size, max_seq_len, 78)
    lengths = torch.randint(10, max_seq_len + 1, (batch_size,))

    print(f"Input shape: {x.shape}")
    print(f"Sequence lengths: {lengths}")

    try:
        with torch.no_grad():
            output = model(x, lengths)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print("✅ Model test passed!")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False


# For backward compatibility, keep the original classes
class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded, x_lengths):
        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output


# Run test if this file is executed directly
if __name__ == "__main__":
    test_model()