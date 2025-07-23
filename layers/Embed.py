import torch
import torch.nn as nn
from torch import Tensor

"""
Patch Embedding for Time Series (Summary):
- Splits a long time series into fixed-size patches (windows), possibly overlapping.
- Each patch is projected into a higher-dimensional embedding space.
- This allows transformer models to process time series as a sequence of tokens, similar to NLP.
- PatchEmbedding, ReplicationPad1d, and TokenEmbedding are the key components for this process.

Positional Embeddings in This Codebase:
- LLMs (like GPT, Llama, BERT) have their own positional embedding mechanisms built-in, which are applied to their input tokens.
- If you concatenate time series patch embeddings with LLM token embeddings, the LLM's positional encodings will be applied to the whole sequence (prompt + patches).
- You could add additional positional encodings to the time series patches before feeding them to the LLM, but this risks double-encoding position unless handled carefully.
"""

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding projects each input patch (or feature vector) into a higher-dimensional embedding space using a 1D convolution.
    For time series, this allows each patch (or timestep) to be represented as a d_model-dimensional vector.
    Args:
        c_in: Number of input channels (patch length or feature count)
        d_model: Output embedding dimension
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # Use circular padding for time series continuity
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: [batch, patch_len, features] or [batch, patch_len]
        # Permute to [batch, features, patch_len] for Conv1d, then back
        # Input: [B, patch_len, c_in] -> Conv1d expects [B, c_in, patch_len]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # [B, patch_len, d_model]
        return x

class ReplicationPad1d(nn.Module):
    """
    ReplicationPad1d pads the input sequence at the end by replicating the last value.
    This ensures that when extracting patches, no data is lost at the sequence boundary.
    Args:
        padding: Tuple (left_pad, right_pad) - only right_pad is used here.
    """
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        # input: [batch, n_vars, seq_len]
        # Replicate the last value along the sequence dimension for right padding
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])  # [batch, n_vars, right_pad]
        output = torch.cat([input, replicate_padding], dim=-1)  # [batch, n_vars, seq_len + right_pad]
        return output

class PatchEmbedding(nn.Module):
    """
    PatchEmbedding splits a time series into overlapping or non-overlapping patches (windows),
    then projects each patch into a d_model-dimensional embedding using TokenEmbedding.
    This is inspired by Vision Transformers (ViT) and allows transformers to process time series as a sequence of tokens.

    Steps:
    1. Pads the input sequence at the end so all patches fit (ReplicationPad1d).
    2. Uses torch.unfold to extract patches of length patch_len with a given stride.
    3. Reshapes the tensor to merge batch and variable dimensions for efficient embedding.
    4. Applies TokenEmbedding to each patch, projecting it to d_model dimensions.
    5. Applies dropout for regularization.
    6. Returns the embedded patches and the number of variables.

    Args:
        d_model: Output embedding dimension
        patch_len: Length of each patch (window)
        stride: Step size between patches
        dropout: Dropout rate

    Input shape: [batch, n_vars, seq_len]
    Output shape: [batch * n_vars, num_patches, d_model], n_vars
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        # Pad the sequence at the end to ensure all patches fit
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Project each patch to d_model dimensions
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, n_vars, seq_len]
        n_vars = x.shape[1]  # Number of variables/features
        x = self.padding_patch_layer(x)  # [batch, n_vars, seq_len + stride]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [batch, n_vars, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [batch * n_vars, num_patches, patch_len]
        x = self.value_embedding(x)  # [batch * n_vars, num_patches, d_model]
        return self.dropout(x), n_vars
