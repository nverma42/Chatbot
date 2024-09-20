import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, v_size, d_model, n_heads, num_layers, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(v_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model))
        self.transformer = nn.Transformer(
            d_model, n_heads, num_layers, num_layers)
        self.fc_out = nn.Linear(d_model, v_size)

    def forward(self, source, target):
        src = self.embedding(source) + \
            self.positional_encoding[:, :source.size(1)]
        tgt = self.embedding(target) + \
            self.positional_encoding[:, :target.size(1)]
        output = self.transformer(src, tgt)
        return self.fc_out(output)
