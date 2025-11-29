import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import nn
import numpy as np
from torch.amp import autocast, GradScaler
import json
from transformers import get_scheduler
from sklearn.metrics import f1_score

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

LIST_OF_PLAYERS = ['ArasanX', 'MassterofMayhem',
                   'JelenaZ', 'lestri',
                   'doreality', 'therealYardbird',
                   'Chesssknock', 'No_signs_of_V',
                   'Recobachess', 'drawingchest',
                   'kasparik_garik', 'ChainsOfFantasia',
                   'Consent_to_treatment', 'Alexandr_KhleBovich',
                   'unknown-maestro_2450', 'gefuehlter_FM',
                   'gmmitkov', 'positionaloldman',
                   "Carlsen, Magnus", "Nakamura, Hikaru"]

csv_path = r"filtered_games_new.csv"
batch_size = 128
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

df_white['transformer_input'] = (
            "1 " + df_white['game_type'] + " " + df_white['time_control'] + " " + df_white['moves'])

X_white = pd.Series(df_white['transformer_input'])
df_black['transformer_input'] = (
            "2 " + df_black['game_type'] + " " + df_black['time_control'] + " " + df_black['moves'])

X_black = pd.Series(df_black['transformer_input'])
X = pd.concat([X_white, X_black], axis=0, ignore_index=True, sort=False)

if len(X) != len(y):
    raise ValueError('X and y must have same length')
print(X.shape)
print(y.shape)

X_train, X_temp, y_train, y_temp, elos_train, elos_temp = train_test_split(X, y, elos, test_size=(1 - 0.9),
                                                                           shuffle=True)

val_size = 0.05 / (0.9 + 0.05)
X_val, X_test, y_val, y_test, elos_val, elos_test = train_test_split(X_temp, y_temp, elos_temp,
                                                                     test_size=(1 - val_size))

# normalization of regression targets to have the loss in a closer scale to the classification loss

mean_elo = np.mean(elos_train)
std_elo = np.std(elos_train)
elos_train = (elos_train - mean_elo) / std_elo
elos_val = (elos_val - mean_elo) / std_elo
elos_test = (elos_test - mean_elo) / std_elo

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


class ChessDataset_transformer(Dataset):
    """
    This is a custom dataset that enables the feeding of chess games as input for our transformer-based model
    It tokenize the input text and transform both classification labels and elo-regression labels in torch.tensors
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
            return -1  # note that this never happens


model_name = "/model/snapshots/e498922d792f3fd7c07471a498ad0a79e0f0b0a0"  # now I am using a local version of the model
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

chessgpt = AutoModel.from_pretrained(model_name, local_files_only=True, low_cpu_mem_usage=True)

tokenizer.pad_token = tokenizer.eos_token
chessgpt.config.pad_token_id = tokenizer.eos_token_id


# print(chessgpt)

def masked_mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


class SharedTwoLayerMLP(nn.Module):
    """
    This is the first part of the multitask MLP that we added to the pretrained-transformer, and is shared between both tasks
    It consists of 2 fully connected layers, each followed by GELU activation function
    """

    def __init__(self, in_dim, shared_dim1, shared_dim2, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, shared_dim1)
        self.fc2 = nn.Linear(shared_dim1, shared_dim2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()  # GELU is the same act fun of the transformer, but we can also try simple RELU

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return x


class ClassificationHead(nn.Module):
    """
    This is the Classification specific end of the network,
    it consists of 1 fully connected layer, followed by a GELU activation function,
    then another fully connected layer that ends in N neurons (where N is the number of classes)
    """

    def __init__(self, in_dim, num_classes, head_dim, dropout):
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
    """
    This is the Regression specific end of the network,
    it consists of 1 fully connected layer, followed by a GELU activation function,
    then another fully connected layer that ends in 2 neurons since we need to predict 2 ELO values
    """

    def __init__(self, in_dim, out_dim, head_dim, dropout):
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
    """
    In this class we put all the previous parts together, after the pretrained transformer
    """

    def __init__(self, base_model, num_classes, num_regression, shared_dim1, shared_dim2, head_dim, dropout,
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


""" 
This is the actual initialization of the model, we specify 21 classes, since we have 20 players that the network must classify.
Then we have 2 regression values, because the model should also predict the elo of both players (white and black)
"""
model = MultiTaskTransformer(
    base_model=chessgpt,
    num_classes=20,
    num_regression=2,
    shared_dim1=512,
    shared_dim2=256,
    head_dim=128,
    dropout=0.1
)

# model.load_state_dict(torch.load("best_model.pth")) <- this let us restart training from the last best model (not always a good idea)

for p in model.base_model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if "shared" in name or "classifier" in name or "regressor" in name:
        p.requires_grad = True

train_dataset = ChessDataset_transformer(X_train, y_train, elos_train, tokenizer, max_length=128)
val_dataset = ChessDataset_transformer(X_val, y_val, elos_val, tokenizer, max_length=128)
test_dataset = ChessDataset_transformer(X_test, y_test, elos_test, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader))

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=8,
                          num_training_steps=187)  # those steps have been calculated based on the number of data, batch size, accumulator size and n. epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)


def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer,
                   scheduler,
                   device: torch.device,
                   alpha=0.001,
                   accumulation_step=32):
    scaler = GradScaler()

    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # used for f1 score and accuracy metrics
    pred_labels_train = []
    true_labels_train = []
    pred_regression_train = []
    true_regression_train = []

    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            class_logits = outputs["classification"]
            class_regr = outputs["regression"]

        class_regr = torch.nan_to_num(class_regr, nan=0.5, posinf=1, neginf=0)
        reg_labels = torch.nan_to_num(reg_labels, nan=0.5, posinf=1, neginf=0)

        ce_loss = torch.nn.functional.cross_entropy(class_logits.float(), class_labels)
        huber_loss = torch.nn.functional.smooth_l1_loss(class_regr.float(), reg_labels.float(), beta=1.0)

        loss = ce_loss + alpha * huber_loss
        loss = loss / accumulation_step

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_step == 0 or i == len(train_loader) - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            print(f"Done {i} Batches of {len(train_loader)}")
            true_labels_train_f1 = torch.cat(true_labels_train, dim=0).detach().cpu().numpy().flatten()
            pred_labels_train_f1 = torch.cat(pred_labels_train, dim=0).detach().cpu().numpy().flatten()
            print(f"class_loss = {ce_loss.item()}")
            print(f"macro_f1 = {f1_score(true_labels_train_f1, pred_labels_train_f1, average='macro')}")
            print(f"weighted_f1 = {f1_score(true_labels_train_f1, pred_labels_train_f1, average='weighted')}")
            print(f"regression_loss = {huber_loss.item()}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # Check params are attached and require grad
            bad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            print("Params with no grad after backward:", bad)

            # Sanity: verify optimizer has trainable params
            print("Optimizer param groups:", sum(p.requires_grad for g in optimizer.param_groups for p in g['params']))
            print("LRs after scheduler:", [g['lr'] for g in optimizer.param_groups])

        batch_losses_train.append(loss.item())

        preds = torch.argmax(class_logits, dim=1)

        pred_labels_train.append(preds)
        true_labels_train.append(class_labels)
        pred_regression_train.append(class_regr)
        true_regression_train.append(reg_labels)

    # Format useful lists for calculation of metrics
    pred_labels_train = torch.cat(pred_labels_train, dim=0).detach().cpu().numpy().flatten()
    true_labels_train = torch.cat(true_labels_train, dim=0).detach().cpu().numpy().flatten()
    pred_regression_train = torch.cat(pred_regression_train, dim=0).cpu().detach().numpy().flatten()
    true_regression_train = torch.cat(true_regression_train, dim=0).cpu().detach().numpy().flatten()

    avg_train_loss = np.mean(batch_losses_train)

    model.eval()

    batch_losses_val = []

    # used for f1 score and accuracy metrics
    pred_labels_val = []
    true_labels_val = []
    pred_regression_val = []
    true_regression_val = []

    for batch in validation_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with autocast(device.type):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                class_logits = outputs["classification"]
                class_regr = outputs["regression"]

            class_regr = torch.nan_to_num(class_regr, nan=0.5, posinf=1, neginf=0)
            reg_labels = torch.nan_to_num(reg_labels, nan=0.5, posinf=1, neginf=0)

            ce_loss = torch.nn.functional.cross_entropy(class_logits.float(), class_labels)
            huber_loss = torch.nn.functional.smooth_l1_loss(class_regr.float(), reg_labels.float(), beta=1.0)
            loss = ce_loss + alpha * huber_loss

        print(loss.item())
        batch_losses_val.append(loss.item())

        preds = outputs["classification"].argmax(dim=1)
        pred_labels_val.append(preds)
        true_labels_val.append(class_labels)
        pred_regression_val.append(class_regr)
        true_regression_val.append(reg_labels)

    avg_val_loss = np.mean(batch_losses_val)

    # Format useful lists for calculation of metrics
    pred_labels_val = torch.cat(pred_labels_val, dim=0).cpu().detach().numpy().flatten()
    true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy().flatten()
    pred_regression_val = torch.cat(pred_regression_val, dim=0).cpu().detach().numpy().flatten()
    true_regression_val = torch.cat(true_regression_val, dim=0).cpu().detach().numpy().flatten()

    return avg_train_loss, avg_val_loss, \
        (pred_labels_train, true_labels_train), \
        (pred_labels_val, true_labels_val), \
        (pred_regression_train, true_regression_train), \
        (pred_regression_val, true_regression_val)


def test(test_loader: DataLoader,
         model: nn.Module,
         device: torch.device,
         alpha=0.001):
    model.eval()

    batch_losses_test = []
    pred_labels_test = []  # used for f1 score and accuracy metrics
    true_labels_test = []  # used for f1 score and accuracy metrics
    pred_regression_test = []
    true_regression_test = []

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            class_logits = outputs["classification"]
            class_regr = outputs["regression"]

        class_regr = torch.nan_to_num(class_regr, nan=0.5, posinf=1, neginf=0)
        reg_labels = torch.nan_to_num(reg_labels, nan=0.5, posinf=1, neginf=0)

        ce_loss = torch.nn.functional.cross_entropy(class_logits.float(), class_labels)
        huber_loss = torch.nn.functional.smooth_l1_loss(class_regr.float(), reg_labels.float(), beta=1.0)

        loss = ce_loss + alpha * huber_loss
        print(loss.item())
        batch_losses_test.append(loss.item())

        preds = outputs["classification"].argmax(dim=1)

        pred_labels_test.append(preds)
        true_labels_test.append(class_labels)
        pred_regression_test.append(class_regr)
        true_regression_test.append(reg_labels)

    # Format useful lists for calculation of metrics
    pred_labels_test = torch.cat(pred_labels_test, dim=0).cpu().detach().numpy().flatten()
    true_labels_test = torch.cat(true_labels_test, dim=0).cpu().detach().numpy().flatten()
    pred_regression_test = torch.cat(pred_regression_test, dim=0).cpu().detach().numpy().flatten()
    true_regression_test = torch.cat(true_regression_test, dim=0).cpu().detach().numpy().flatten()

    avg_test_loss = np.mean(batch_losses_test)

    return avg_test_loss, (pred_labels_test, true_labels_test), (pred_regression_test, true_regression_test)


def unfreeze_last_layer(model):
    """
    This function unfreezes the last layer of the transformer,
    this is because slightly changing the representation of the last layer
     after a few epochs of training can be very beneficial for the accuracy.
    This is due to the fact that the representation in the last layer is the most task-specific.
    """
    last_block = model.base_model.layers[-1]
    for p in last_block.parameters():
        p.requires_grad = True

    for p in model.base_model.final_layer_norm.parameters():
        p.requires_grad = True

    print(f"Unfroze last transformer layer.")


"""
Here we have the training loop, where we finetune the added multitask-MLP for 4 epochs,
 to then unfreeze the last layer of the transformer, and continue with the other 6 epochs.
We have a dataset of almost 400000 chess games, which are slow to feed to such a big model, therefore we had to settle for this.

After the training is done we save the weights of the best performing model (on the validation set) and we test it on the test set.

The train_validate and test functions return all the useful information regarding the performance of the model,
 that can be processed and visualized in a second moment.
"""

epochs_non_improved = 0
best_val_loss = float("inf")

metrics_dict = {
    "avg_train_loss": [],
    "avg_val_loss": [],
    "pred_labels_train": [],
    "true_labels_train": [],
    "pred_labels_val": [],
    "true_labels_val": [],
    "pred_regression_train": [],
    "true_regression_train": [],
    "pred_regression_val": [],
    "true_regression_val": [],
}

for epoch in range(10):
    avg_train_loss, avg_val_loss, \
        (pred_labels_train, true_labels_train), \
        (pred_labels_val, true_labels_val), \
        (pred_regression_train, true_regression_train), \
        (pred_regression_val, true_regression_val) = train_validate(train_loader, val_loader, model, optimizer,
                                                                    scheduler, device)

    metrics_dict["avg_train_loss"].append(avg_train_loss)
    metrics_dict["avg_val_loss"].append(avg_val_loss)
    metrics_dict["pred_labels_train"].append(pred_labels_train)
    metrics_dict["true_labels_train"].append(true_labels_train)
    metrics_dict["pred_labels_val"].append(pred_labels_val)
    metrics_dict["true_labels_val"].append(true_labels_val)
    metrics_dict["pred_regression_train"].append(pred_regression_train)
    metrics_dict["true_regression_train"].append(true_regression_train)
    metrics_dict["pred_regression_val"].append(pred_regression_val)
    metrics_dict["true_regression_val"].append(true_regression_val)

    with open("metrics.json", "w") as f:
        json.dump(metrics_dict, f)

    print(f"avg_train_loss {avg_train_loss}")
    print(f"avg_val_loss {avg_val_loss}")
    if epochs_non_improved == 2:
        print(f"Epoch {epoch + 1}\n-----------------------------------------------")
        # print(f"Train Loss: {avg_train_loss:>7f}\tTrain Accuracy: {(100 * train_accuracy):>0.1f}%")
        # print(f"Validation Loss: {avg_val_loss:>7f}\tValidation Accuracy: {(100 * avg_val_accuracy):>0.1f}%")
        print(f"------------------------------------------------------------")
        break
    elif best_val_loss > avg_val_loss:
        epochs_non_improved = 0
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
    else:
        # print(f"NOT improved {epochs_non_improved} epochs")
        epochs_non_improved += 1

    if epoch == 4:
        unfreeze_last_layer(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        head_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                if name.startswith("shared") or name.startswith("classifier") or name.startswith("regressor"):
                    head_params.append(p)
                else:
                    backbone_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": 1e-4, "weight_decay": 0.01},  # heads: faster LR
            {"params": backbone_params, "lr": 2e-5, "weight_decay": 0.01},  # unfrozen GPT-NeoX blocks: smaller LR
        ])

        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=3,
                                  num_training_steps=250)  # again as above, calculated by me

model.load_state_dict(torch.load("best_model.pth"))  # I load the model that performed better on validation

avg_test_loss, (pred_labels_test, true_labels_test), (pred_regression_test, true_regression_test) = test(test_loader,
                                                                                                         model,
                                                                                                         device)  # I test only the best model

test_metrics_dict = {
    "avg_test_loss": avg_test_loss,
    "pred_labels_test": pred_labels_test,
    "true_labels_test": true_labels_test,
    "pred_regression_test": pred_regression_test,
    "true_regression_test": true_regression_test,
}

with open("test_metrics.json", "w") as f:
    json.dump(test_metrics_dict, f)


