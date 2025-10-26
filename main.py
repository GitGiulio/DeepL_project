import numpy as np
#import matplotlib.pyplot as plt
import sklearn
import torch
from torch import nn
from mlxtend.data import mnist_data
import math


"""HYPER PARAMETERS"""
learning_rate = 0.001
test_size = 0.3
"""HYPER PARAMETERS"""

X,y = mnist_data()

Xtrain, Xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=test_size,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ChessPlayerTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(ChessPlayerTransformer, self).__init__()

        # Embedding layer for moves or board states
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding to retain move order
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [batch_size, seq_len, embed_dim]

        # Use the representation of the final token or mean pooling
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        return self.fc(x)  # [batch_size, num_classes] # this is returning the same output of a classification net ( N nodes each with the "probability" of the inputo to be of the corresponding class)

model = ChessPlayerTransformer(vocab_size=10, embed_dim=64, num_heads=8, num_layers=2, dropout=0.1) # TODO understand param this model was made by GPT-4 as an example

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.functional.cross_entropy

randomImput = torch.randn((3, 20))

randomImput.to(device)

print(randomImput)

print(model(randomImput))