import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import nn
import numpy as np

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

LIST_OF_PLAYERS = ["Carlsen, Magnus",
                   "Cramling Bellon, Anna",
                   "Caruana, Fabiano",
                   "Nepomniachtchi, Ian",
                   "Firouzja, Alireza",
                   "Giri, Anish",
                   "Niemann, Hans",
                   "Cramling, Pia",
                   "Nakamura, Hikaru",
                   "Botez, Alexandra",
                   "Botez, Andrea",
                   "Belenkaya, Dina",
                   "So, Wesley",]


#csv_path = r"/content/filtered_games.csv"
csv_path = r"C:\Users\giuli\PycharmProjects\DeepL_project_test\data\filtered_games.csv"
batch_size = 16                   # Batch size for DataLoader
train_ratio = 0.9                # Train split ratio
val_ratio = 0.05                 # Validation split ratio
test_ratio = 0.05                # Test split ratio
shuffle = True                   # Shuffle data before splitting
num_epochs = 10

df = pd.read_csv(csv_path)

df_white = df[df['white_name'].isin(LIST_OF_PLAYERS)].copy()
df_black = df[df['black_name'].isin(LIST_OF_PLAYERS)].copy()

for col in ['game_type', 'time_control', 'moves']:
    df_white[col] = df_white[col].fillna('').astype(str)
    df_black[col] = df_black[col].fillna('').astype(str)

y_white = pd.Series(df_white['white_name'])
y_black = pd.Series(df_black['black_name'])
y = pd.concat([y_white, y_black], axis=0, ignore_index=True, sort=False)

df_white['transformer_input'] = ("1 " + df_white['game_type'] + " " + df_white['time_control'] + " " + df_white['moves'])

X_white = pd.Series(df_white['transformer_input'])
df_black['transformer_input'] = ("2 " + df_black['game_type'] + " " + df_black['time_control'] + " " + df_black['moves'])

X_black = pd.Series(df_black['transformer_input'])
X = pd.concat([X_white, X_black], axis=0,ignore_index=True, sort=False)

if len(X) != len(y):
    raise ValueError('X and y must have same length')
print(X.shape)
print(y.shape)


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), shuffle=shuffle)

val_size = val_ratio / (val_ratio + test_ratio)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_size), shuffle=shuffle)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


class ChessDataset_transformer(Dataset):
    """
    TODO
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = [self.player_to_idx(name) for name in labels]  # Keep as list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text
        encoding = self.tokenizer(
            self.texts.iloc[idx],
            padding='max_length',  # or use DataCollator for dynamic padding
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}  # remove batch dim

        # Convert label to tensor here (per batch)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def player_to_idx(self, name):
        if name in LIST_OF_PLAYERS:
            return LIST_OF_PLAYERS.index(name)
        else:
            return len(LIST_OF_PLAYERS)

model_name = "Waterhorse/chessgpt-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=14)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

print(model)

for param in model.base_model.parameters():
    param.requires_grad = False


for param in model.score.parameters():
    param.requires_grad = True


# model.classifier = MultiTaskHead(model.config.hidden_size) TODO this is how we add a custom classifier

#def tokenize(batch):
#    return tokenizer(batch, truncation=True, padding="max_length", max_length=512)


torch.manual_seed(42)

train_dataset = ChessDataset_transformer(X_train, y_train, tokenizer, max_length=128)
val_dataset = ChessDataset_transformer(X_val, y_val, tokenizer, max_length=128)
test_dataset = ChessDataset_transformer(X_test, y_test, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader))


optimizer = AdamW(model.parameters(), lr=5e-5)

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer: nn.Module,
                   device: torch.device):
    estimatedLabels_train = []
    trueLabels_train = []
    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # --- TRAINING ---
    model.train()
    c = 0
    for batch in train_loader:
        c += 1
        if c % 1 == 0:
            print(f"fatti {c * 8}")
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        batch_losses_train.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO classification
        # for now just an argmax
        # yhat = torch.argmax(y_pred, dim=1)
        # estimatedLabels_train.append(yhat.cpu())
        # trueLabels_train.append(ybatch.cpu())

    # all_estimated = torch.cat(estimatedLabels_train, dim=0).numpy().flatten()
    # all_true = torch.cat(trueLabels_train, dim=0).numpy().flatten()
    avg_train_loss = np.mean(batch_losses_train)

    # --- VALIDATION ---
    model.eval()

    estimatedLabels_val = []
    # We may need the values in the array below to calculate a PR curve
    # To optimize F1-scores. But at this point unsure whether we will use F1 scores.
    estimatedLabels_val_raw_logits = []
    trueLabels_val = []
    batch_losses_val = []

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # estimatedLabels_val_raw_logits.append(outputs.cpu())

            batch_losses_val.append(loss.item())

            # TODO classification
            # for now just an argmax
            yhat = torch.argmax(outputs, dim=1)
            estimatedLabels_val.append(yhat.cpu())
            trueLabels_val.append(batch["labels"].cpu())

            # print(y_pred[:5], ybatch[:5])

    # Validation loss will be the total validation loss over all batches divided by the number of batches.
    avg_val_loss = np.mean(batch_losses_val)  # we can do this because reduction =' mean'

    # After validation, do scheduler step
    # all_estimated = torch.cat(estimatedLabels_val, dim=0).numpy().flatten()
    # all_true = torch.cat(trueLabels_val, dim=0).numpy().flatten()

    return avg_train_loss, avg_val_loss


'''
Main test function. Independent of epoch. 

Returns the test loss.
'''


def test(test_loader: DataLoader,
         model: nn.Module,
         loss_fn: nn.Module,
         device: torch.device):
    model.eval()

    estimatedLabels = []
    trueLabels = []
    batch_losses_test = []

    with torch.no_grad():
        for xbatch, ybatch in test_loader:
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            y_pred = model(xbatch)

            loss = loss_fn(y_pred, ybatch)

            batch_losses_test.append(loss.item())

            # TODO classification
            # for now just an argmax
            yhat = torch.argmax(y_pred, dim=1)
            estimatedLabels.append(yhat.cpu())
            trueLabels.append(ybatch.cpu())

    # Validation loss will be the total validation loss over all batches divided by the number of batches.
    avg_test_loss = np.mean(batch_losses_test)

    return avg_test_loss

for epoch in range(num_epochs):
  avg_train_loss, avg_val_loss = train_validate(train_loader,val_loader,model, optimizer,device)
  print(f"avg_train_loss {avg_train_loss}")
  print(f"avg_val_loss {avg_val_loss}")

"""
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(loss.item())
        loss.backward()
        optimizer.step()
"""