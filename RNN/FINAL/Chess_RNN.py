from torch import nn
import torch

'''
@author Nikan Mahdavi Tabatabaei, Mathijs Tobé

The initial RNN is created by Nikan, but mainly improved to perform well with 4 players. 
Afterwards, the RNN is mostly changed to adapt better to bigger datasets, with extra regularization measures and attention.
A short description of the initial RNN is given in the report.

https://www.nature.com/articles/s41598-025-88378-6 - Source for how applying attention to LSTM outputs can give better performance
'''
class Chess_RNN(nn.Module):
    # Params should make sense, worth mentioning is the dim_embedded, which controls how big the embedding table should be
    # The dim_hidden_layer drastically increases or decreases the number of trainable params of the LSTM
    # dim_out should always be equal to the number of players that need to be classified. 
    def __init__(self, dir, dropout, lstm_layers, dim_embedded, dim_hidden_layer, dim_out):
        super(Chess_RNN, self).__init__()

        # embedding lookup table for the tokens
        self.table = nn.Embedding( 
            num_embeddings=dir,
            embedding_dim=dim_embedded,  
            padding_idx=2  # telling torch 0's are padding, not actual moves
        )

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=dim_embedded, 
            hidden_size=dim_hidden_layer,
            num_layers=lstm_layers,
            batch_first=True,  # we pass the data with batch as the first dimension, so True is necessary.
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
        
        # Used for creating a context vector
        self.attention = nn.Linear(2*dim_hidden_layer, 1)
        self.att_dropout = nn.Dropout(dropout)
        
        # The output of the LSTM is normalized, 
        self.norm = nn.LayerNorm(2 * dim_hidden_layer)

    def forward(self, x):
        x = self.table(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        
        raw_attention = self.attention(lstm_out).squeeze(-1)
        raw_attention = self.att_dropout(raw_attention)
        attention_weights = torch.softmax(raw_attention, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        return self.FC(context_vector)
