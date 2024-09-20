import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Custom Transformer model for text generation.

    Attributes:
        v_size (int): Vocabulary size.
        d_model (int): Dimensionality of the model.
        n_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_seq_length (int): Maximum sequence length.
    """

    def __init__(self, v_size, d_model, n_heads, num_layers, max_seq_length):
        """
        Initializes the transformer model with embedding, positional encoding, and transformer layers.

        Args:
            v_size (int): Vocabulary size for embedding layer.
            d_model (int): Dimensionality of the transformer.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of layers in the transformer.
            max_seq_length (int): Maximum sequence length for positional encoding.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(v_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model))
        self.transformer = nn.Transformer(
            d_model, n_heads, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, v_size)

    def forward(self, source, target):
        """
        Forward pass of the transformer model.

        Args:
            source (torch.Tensor): Input tensor containing source sequence.
            target (torch.Tensor): Input tensor containing target sequence.

        Returns:
            torch.Tensor: Output tensor containing predicted tokens.
        """
        src = self.embedding(source) + \
            self.positional_encoding[:, :source.size(1)]
        tgt = self.embedding(target) + \
            self.positional_encoding[:, :target.size(1)]
        output = self.transformer(src, tgt)
        return self.fc_out(output)
