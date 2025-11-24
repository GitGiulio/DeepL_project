import torch
import transformers
from transformers import AutoTokenizer, AutoModel
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

LIST_OF_PLAYERS = ['ArasanX','MassterofMayhem',
                   'JelenaZ','lestri',
                   'doreality','therealYardbird',
                   'Chesssknock','No_signs_of_V',
                   'Recobachess','drawingchest',
                   'kasparik_garik', 'ChainsOfFantasia',
                   'Consent_to_treatment','Alexandr_KhleBovich',
                   'unknown-maestro_2450', 'gefuehlter_FM',
                   'gmmitkov', 'positionaloldman',
                   "Carlsen, Magnus","Nakamura, Hikaru"]


csv_path = r"filtered_games_new.csv"
batch_size = 256
train_ratio = 0.9
val_ratio = 0.05
test_ratio = 0.05
shuffle = True
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

df_white_black = pd.concat([df_white, df_black], axis=0, ignore_index=True, sort=False)
df_white_black['white_elo'] = df_white_black['white_elo'].replace('NOT_FOUND', 1000).astype(float)
df_white_black['black_elo'] = df_white_black['black_elo'].replace('NOT_FOUND', 1000).astype(float)

elos = np.vstack([df_white_black['white_elo'].to_numpy(), df_white_black['black_elo'].to_numpy()]).T.astype(np.float32)

df_white['transformer_input'] = ("1 " + df_white['game_type'] + " " + df_white['time_control'] + " " + df_white['moves'])

X_white = pd.Series(df_white['transformer_input'])
df_black['transformer_input'] = ("2 " + df_black['game_type'] + " " + df_black['time_control'] + " " + df_black['moves'])

X_black = pd.Series(df_black['transformer_input'])
X = pd.concat([X_white, X_black], axis=0,ignore_index=True, sort=False)

if len(X) != len(y):
    raise ValueError('X and y must have same length')
print(X.shape)
print(y.shape)


X_train, X_temp, y_train, y_temp, elos_train, elos_temp = train_test_split(X, y,elos, test_size=(1 - train_ratio), shuffle=shuffle)

val_size = val_ratio / (val_ratio + test_ratio)
X_val, X_test, y_val, y_test, elos_val, elos_test = train_test_split(X_temp, y_temp,elos_temp, test_size=(1 - val_size), shuffle=shuffle)

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
    def __init__(self, texts, labels, elos, tokenizer, max_length=512):
        self.texts = texts
        self.class_labels = [self.player_to_idx(name) for name in labels]
        self.regression_labels = elos
        self.tokenizer = tokenizer
        self.max_length = max_length  # TODO CHANGE TO FEED the new model

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
        item['class_labels'] = torch.tensor(self.class_labels[idx], dtype=torch.long)
        item['regression_labels'] = torch.tensor(self.regression_labels[idx], dtype=torch.float32)
        return item

    def player_to_idx(self, name):
        if name in LIST_OF_PLAYERS:
            return LIST_OF_PLAYERS.index(name)
        else:
            return len(LIST_OF_PLAYERS)

model_name = "Waterhorse/chessgpt-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

chessgpt = AutoModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
chessgpt.config.pad_token_id = tokenizer.eos_token_id
chessgpt.resize_token_embeddings(len(tokenizer))

print(chessgpt)

def masked_mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


class SharedTwoLayerMLP(nn.Module):
    def __init__(self, in_dim, shared_dim1=1024, shared_dim2=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, shared_dim1)
        self.fc2 = nn.Linear(shared_dim1, shared_dim2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))  # shared layer 1
        x = self.dropout(self.act(self.fc2(x)))  # shared layer 2
        return x                                  # shared representation


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes=20, head_dim=256, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, head_dim)
        self.out = nn.Linear(head_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc(x)))
        logits = self.out(x)
        return logits


class RegressionHead(nn.Module):
    def __init__(self, in_dim, out_dim=2, head_dim=256, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, head_dim)
        self.out = nn.Linear(head_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc(x)))
        values = self.out(x)
        return values


class MultiTaskTransformer(nn.Module):
    def __init__(self, base_model, num_classes=20, num_regression=2,
                 shared_dim1=1024, shared_dim2=512, head_dim=256, dropout=0.1,
                 use_cls_if_available=True):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        self.shared = SharedTwoLayerMLP(
            in_dim=hidden_size,
            shared_dim1=shared_dim1,
            shared_dim2=shared_dim2,
            dropout=dropout
        )

        self.classifier = ClassificationHead(
            in_dim=shared_dim2,
            num_classes=num_classes,
            head_dim=head_dim,
            dropout=dropout
        )
        self.regressor = RegressionHead(
            in_dim=shared_dim2,
            out_dim=num_regression,
            head_dim=head_dim,
            dropout=dropout
        )

        self.use_cls_if_available = use_cls_if_available

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden = outputs.last_hidden_state
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            pooled = masked_mean_pooling(last_hidden, attention_mask)

        shared_repr = self.shared(pooled)

        class_logits = self.classifier(shared_repr)
        reg_values = self.regressor(shared_repr)

        return {
            "classification": class_logits,
            "regression": reg_values
        }


model = MultiTaskTransformer(
    base_model=chessgpt,
    num_classes=20,
    num_regression=2,
    shared_dim1=1024,
    shared_dim2=512,
    head_dim=256,
    dropout=0.1
)

for p in model.base_model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if "shared" in name or "classifier" in name or "regressor" in name:
        p.requires_grad = True

print(model)

train_dataset = ChessDataset_transformer(X_train, y_train, elos_train, tokenizer, max_length=128)
val_dataset = ChessDataset_transformer(X_val, y_val, elos_val, tokenizer, max_length=128)
test_dataset = ChessDataset_transformer(X_test, y_test, elos_test, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader))

optimizer = AdamW(model.parameters(), lr=9e-4)

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer,
                   device: torch.device):
    correct = 0
    total = 0
    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    model.train()
    c = 0
    for batch in train_loader:
        c += 1
        if c % 10 == 0:
            print(f"fatti {c * batch_size}")
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        class_logits = outputs["classification"]
        reg_values = outputs["regression"]

        ce_loss = torch.nn.functional.cross_entropy(class_logits, class_labels)
        mse_loss = torch.nn.functional.mse_loss(reg_values, reg_labels)

        loss = ce_loss + 0.2 * mse_loss  # TODO weight regression appropriately
        print(loss)
        batch_losses_train.append(loss)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(class_logits, dim=1)
        correct += (preds == class_labels).sum().item()
        total += class_labels.size(0)

    accuracy = correct / total
    print(f"Training Accuracy: {accuracy:.4f}")

    avg_train_loss = np.mean(batch_losses_train)

    model.eval()

    correct = 0
    total = 0
    batch_losses_val = []

    for batch in validation_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs["classification"].argmax(dim=1)

        ce_loss = torch.nn.functional.cross_entropy(preds, class_labels)
        mse_loss = torch.nn.functional.mse_loss(outputs["regression"], reg_labels)

        loss = ce_loss + 0.2 * mse_loss
        print(loss.item())
        batch_losses_val.append(loss.item())

        correct += (preds == class_labels).sum().item()
        total += class_labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    avg_val_loss = np.mean(batch_losses_val)

    return avg_train_loss, avg_val_loss


def test(test_loader: DataLoader,
         model: nn.Module,
         device: torch.device):
    model.eval()

    correct = 0
    total = 0
    batch_losses_test = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        ce_loss = torch.nn.functional.cross_entropy(preds, class_labels)
        mse_loss = torch.nn.functional.mse_loss(outputs["regression"], reg_labels)

        loss = ce_loss + 0.2 * mse_loss
        print(loss.item())
        batch_losses_test.append(loss.item())

        correct += (preds == class_labels).sum().item()
        total += class_labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    avg_test_loss = np.mean(batch_losses_test)

    return avg_test_loss


for epoch in range(num_epochs):
    avg_train_loss, avg_val_loss = train_validate(train_loader,val_loader,model,optimizer,device)
    print(f"avg_train_loss {avg_train_loss}")
    print(f"avg_val_loss {avg_val_loss}")

nul = test(test_loader,model,device)
