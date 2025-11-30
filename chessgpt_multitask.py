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

csv_path = r"/filtered_games_new.csv"
batch_size = 128
df = pd.read_csv(csv_path)

def stratified_random_subset(df, label_col="class", per_class=32, seed=77):
    """
    This function is used to take for each class the same number of games
    """
    subset_rows = []
    for cls, group in df.groupby(label_col):
        take = min(per_class, len(group))
        if cls in LIST_OF_PLAYERS:
            subset_rows.append(group.sample(n=take, random_state=seed))
    return pd.concat(subset_rows).sample(frac=1, random_state=seed).reset_index(drop=True)


w_in = df['white_name'].isin(set(LIST_OF_PLAYERS))
b_in = df['black_name'].isin(set(LIST_OF_PLAYERS))

df['class'] = np.where(
    w_in,                         # if white_name in L
    df['white_name'],             # -> choose white_name
    np.where(
        b_in,                     # else, if black_name in L
        df['black_name'],         # -> choose black_name
        'NONE'                    # else -> NONE
    )
)

df = stratified_random_subset(df, label_col="class", per_class=8020) # I take 8020 games for each player, since this is how many we have for the one with the least

# In this I have the biggest possible Dataset while avoiding class imbalance


df_white = df[df['white_name'].isin(LIST_OF_PLAYERS)].copy()
df_black = df[df['black_name'].isin(LIST_OF_PLAYERS)].copy()

for col in ['game_type', 'time_control', 'moves']:
    df_white[col] = df_white[col].fillna('').astype(str)
    df_black[col] = df_black[col].fillna('').astype(str)

y_white = pd.Series(df_white['white_name'])
y_black = pd.Series(df_black['black_name'])
y = pd.concat([y_white, y_black], axis=0, ignore_index=True, sort=False)

df_white_black = pd.concat([df_white, df_black], axis=0, ignore_index=True, sort=False)

df_white_black['white_elo'] = df_white_black['white_elo'].replace('NOT_FOUND', np.nan).astype(float)
df_white_black['black_elo'] = df_white_black['black_elo'].replace('NOT_FOUND', np.nan).astype(float)
df_white_black['white_elo'].fillna(df_white_black['white_elo'].mean(), inplace=True)
df_white_black['black_elo'].fillna(df_white_black['black_elo'].mean(), inplace=True)

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

X_train, X_temp, y_train, y_temp, elos_train , elos_temp,  = train_test_split(X, y, elos,test_size=0.15,stratify=y,shuffle=True,random_state=77)

X_val, X_test, y_val, y_test, elos_val, elos_test = train_test_split(X_temp, y_temp, elos_temp,test_size=2/3,stratify=y_temp,shuffle=True,random_state=77)

def pretty_counts(n_total, n_train, n_val, n_test):
    pct = lambda n: 100.0 * n / n_total
    print(f"Total: {n_total:,}")
    print(f"Train: {n_train:,} ({pct(n_train):.2f}%)")
    print(f"Val:   {n_val:,} ({pct(n_val):.2f}%)")
    print(f"Test:  {n_test:,} ({pct(n_test):.2f}%)")

pretty_counts(len(y), len(y_train), len(y_val), len(y_test))

# normalization of regression targets to have the loss in a closer scale to the classification loss

mean_elo = np.mean(elos_train)
std_elo = np.std(elos_train)
elos_train = (elos_train - mean_elo) / std_elo
elos_val = (elos_val - mean_elo) / std_elo
elos_test = (elos_test - mean_elo) / std_elo

print(f"|-|-|-|-|-|mean_elo: {mean_elo}|-|-|-|-|")
print(f"|-|-|-|-|-|std_elo: {std_elo}|-|-|-|-|")


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

def unfreeze_last_layers(model):
    """
    This function unfreezes the last layer of the transformer,
    this is because slightly changing the representation of the last layer
     after a few epochs of training can be very beneficial for the accuracy.
    This is due to the fact that the representation in the last layer is the most task-specific.
    """
    third_last = model.base_model.layers[-3]
    for p in third_last.parameters():
        p.requires_grad = True

    second_last = model.base_model.layers[-2]
    for p in second_last.parameters():
        p.requires_grad = True

    last_block = model.base_model.layers[-1]
    for p in last_block.parameters():
        p.requires_grad = True

    for p in model.base_model.final_layer_norm.parameters():
        p.requires_grad = True

    print(f"Unfroze last transformer layer.")

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
This is the actual initialization of the model, we specify 20 classes, since we have 20 players that the network must classify.
Then we have 2 regression values, because the model should also predict the elo of both players (white and black)
"""
model = MultiTaskTransformer(
    base_model=chessgpt,
    num_classes=20,
    num_regression=2,
    shared_dim1=512,
    shared_dim2=256,
    head_dim=128,
    dropout=0.0
)

for p in model.base_model.parameters():
    p.requires_grad = False

train_dataset = ChessDataset_transformer(X_train, y_train, elos_train, tokenizer, max_length=256)
val_dataset = ChessDataset_transformer(X_val, y_val, elos_val, tokenizer, max_length=256)
test_dataset = ChessDataset_transformer(X_test, y_test, elos_test, tokenizer, max_length=256)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader))

unfreeze_last_layers(model)

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
    {"params": head_params, "lr": 5e-4, "weight_decay": 0.00001},  # heads: faster LR
    {"params": backbone_params, "lr": 1e-4, "weight_decay": 0.00001},  # unfrozen GPT-NeoX blocks: smaller LR
])

scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=8, num_training_steps=2500)  # those steps have been calculated based on the number of data, batch size, accumulator size and n. epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

print(f"-------LOADING MODEL--------------")
model.load_state_dict(torch.load("best_model.pth")) # <- this let us restart training from the last best model (not always a good idea)
print(f"-------LOADING MODEL DONE---------")


def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer,
                   scheduler,
                   device: torch.device,
                   alpha=0.2,
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

        class_regr = torch.nan_to_num(class_regr, nan=0.0, posinf=3, neginf=-15)
        reg_labels = torch.nan_to_num(reg_labels, nan=0.0, posinf=3, neginf=-15)

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

        batch_losses_train.append(loss.item() * accumulation_step)

        preds = torch.argmax(class_logits, dim=1)

        pred_labels_train.append(preds)
        true_labels_train.append(class_labels)
        pred_regression_train.append(class_regr)
        true_regression_train.append(reg_labels)

    # Format useful lists for calculation of metrics
    pred_labels_train = torch.cat(pred_labels_train, dim=0).detach().cpu().numpy().flatten()
    true_labels_train = torch.cat(true_labels_train, dim=0).detach().cpu().numpy().flatten()
    pred_regression_train = torch.cat(pred_regression_train, dim=0).cpu().detach().to(torch.float32).numpy().flatten()
    true_regression_train = torch.cat(true_regression_train, dim=0).cpu().detach().to(torch.float32).numpy().flatten()

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

            class_regr = torch.nan_to_num(class_regr, nan=0.0, posinf=3, neginf=-15)
            reg_labels = torch.nan_to_num(reg_labels, nan=0.0, posinf=3, neginf=-15)

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
         alpha=0.2):
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

        class_regr = torch.nan_to_num(class_regr, nan=0.0, posinf=3, neginf=-15)
        reg_labels = torch.nan_to_num(reg_labels, nan=0.0, posinf=3, neginf=-15)

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



def to_serializable(obj):
    """
    Recursively convert NumPy arrays / Tensors inside lists/dicts to JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [to_serializable(x) for x in obj.tolist()]
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().float().tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


"""
Here we have the training loop, where we train the added multitask-MLP and the last 2 layers of the pretrained transformer.
We have a dataset of almost 140000 chess games, which are slow to feed to such a big model, therefore we had to settle for only 15 epochs,
but already with this we see the model settle around some metrics.

After the training is done we save the weights of the best performing model (on the validation set) and we test it on the test set (10% of the games).

The train_validate and test functions return all the useful information regarding the performance of the model,
 that are saved in a json file each epoch, so we can process and visualize them in a second moment.
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

for epoch in range(15):
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

    with open(f"./metrics/metrics_epoch_{epoch}.json", "w") as f:
        json.dump(to_serializable(metrics_dict), f, indent=2)

    print(f"avg_train_loss {avg_train_loss}")
    print(f"avg_val_loss {avg_val_loss}")

    if best_val_loss > avg_val_loss:
        epochs_non_improved = 0
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
    else:
        # print(f"NOT improved {epochs_non_improved} epochs")
        epochs_non_improved += 1
        if epochs_non_improved == 2:
            break

model.load_state_dict(torch.load(r"./best_models/best_model.pth"))  # I load the model that performed better on validation to test it

avg_test_loss, (pred_labels_test, true_labels_test), (pred_regression_test, true_regression_test) = test(test_loader,model,device)  # I test only the best model

test_metrics_dict = {
    "avg_test_loss": avg_test_loss,
    "pred_labels_test": pred_labels_test,
    "true_labels_test": true_labels_test,
    "pred_regression_test": pred_regression_test,
    "true_regression_test": true_regression_test,
}


with open(r"./metrics/test_metrics.json", "w") as f:
    json.dump(to_serializable(test_metrics_dict), f, indent=2)
