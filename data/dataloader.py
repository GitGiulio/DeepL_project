import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ChessDataset(Dataset):
    def __init__(self, df, target_col):
        # Map player names to integer indices
        self.player_to_idx = {name: idx for idx, name in enumerate(df[target_col].unique())}
        self.labels = torch.tensor(df[target_col].map(self.player_to_idx).values, dtype=torch.long)

        # Build deterministic move-to-index mapping
        all_moves = set()
        for m in df['list_of_moves']:
            moves_list = ast.literal_eval(m)
            all_moves.update(moves_list)
        self.move_to_idx = {move: idx+1 for idx, move in enumerate(sorted(all_moves))}


        # Encode all move sequences as tensors
        self.moves = []
        for m in df['list_of_moves']:
            moves_list = ast.literal_eval(m)
            encoded = [self.move_to_idx[x] for x in moves_list]
            self.moves.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.moves[idx], self.labels[idx]


# Collate function for DataLoader
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)  # pad to max length in batch
    labels = torch.stack(labels)
    return sequences_padded, labels


def make_dataloaders(csv_path, target='black_name', splits=(0.8, 0.1, 0.1),
                     batch_size=32, seed=42):
    df = pd.read_csv(csv_path)
    dataset = ChessDataset(df, target_col=target)
    num_classes = len(dataset.player_to_idx)

    # Split dataset
    total_size = len(dataset)
    lengths = [int(total_size * s) for s in splits]
    lengths[-1] = total_size - sum(lengths[:-1])  # fix rounding
    torch.manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, lengths)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, dataset.player_to_idx, num_classes




# Test

"""
train_loader, val_loader, test_loader, player_to_idx, num_classes = make_dataloaders(
    "deep_learning/games.csv",
    target="black_name",
    batch_size=32
)

# Iterate one batch
for X_batch, Y_batch in train_loader:
    print("X_batch shape:", X_batch.shape)  # [batch_size, max_seq_len_in_batch]
    print("Y_batch shape:", Y_batch.shape)  # [batch_size]
    break
    

"""
