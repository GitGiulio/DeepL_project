import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device {device}")
# --- THE RNN MODEL ---
# https://www.nature.com/articles/s41598-025-88378-6
# Applying attention to LSTM outputs
class RecurrentNN(nn.Module):
    def __init__(self, dir, dropout, lstm_layers, dim_embedded, dim_hidden_layer, dim_out):
        super(RecurrentNN, self).__init__()

        # lookup table for the tokens
        self.table = nn.Embedding( 
            num_embeddings=dir,
            embedding_dim=dim_embedded,  # size of embeddings
            padding_idx=2  # telling torch 0's are padding, not actual moves
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=dim_embedded, 
            hidden_size=dim_hidden_layer,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True, # backward and forward
            dropout=dropout 
        )

        # Fully connected layers with ReLU and dropout
        self.FC = nn.Sequential(
            nn.Linear(2*dim_hidden_layer, dim_hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden_layer, dim_out)
        )
        
        self.attention = nn.Linear(2*dim_hidden_layer, 1)

        self.norm = nn.LayerNorm(2 * dim_hidden_layer)

    def forward(self, x):
        x = self.table(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # Attempted, but not much difference
        # context, _ = self.multiheadattention(lstm_out, lstm_out, lstm_out)
        # context_vector = context.mean(dim=1)

        return self.FC(context_vector)

torch.backends.cudnn.enabled=False

model = RecurrentNN( # Building model
    dir=10,
    dropout=0.1,
    lstm_layers=2,
    dim_embedded=150,
    dim_hidden_layer=150,
    dim_out=20
).to(device)

print(summary(model=model, input_size=(150, 101), dtypes=[torch.long]))