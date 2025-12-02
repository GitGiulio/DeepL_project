# import all libraries
import numpy as np
# import sklearn.model_selection
from sklearn.model_selection import train_test_split
# import os
# from tqdm import tqdm
# from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
# import torch.optim as optim
# import torchvision.transforms as transforms
from collections import Counter
# from pathlib import Path
# from io import BytesIO
import itertools
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
# import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
# import matplotlib.pyplot as plt
import pandas as pd
# from tabulate import tabulate
from pathlib import Path
# for logging
import mlflow
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(f"Using device {device}")

# import games in csv
# data_path = "C:\\Users\\mathi\\Documents\\University\\Aarhus University\\MSc Computer Engineering\\Semester 1\\Deep Learning\\project\\DeepL_project\\data\\filtered_games_20_players.csv"
# project_root = "/teamspace/studios/this_studio/DeepL_project"
data_path = "/filtered_games_new.csv"

# Data loading
print("Loading data...")
data = pd.read_csv(data_path) # loading into dataframe

print("Data loaded...")
#Added first move indicating to predict either white or black
#White = 0
#Black = 1

# data = data[:200]
# print("Only using the first 200 rows for this test")

# --- SETTINGS ---
NEW_LIST_OF_PLAYERS_MANUAL = [
    'ArasanX','MassterofMayhem','JelenaZ','lestri','doreality','therealYardbird',
    'Chesssknock','No_signs_of_V','Recobachess','drawingchest','kasparik_garik',
    'ChainsOfFantasia','Consent_to_treatment','Alexandr_KhleBovich','unknown-maestro_2450',
    'gefuehlter_FM','gmmitkov','positionaloldman',"Carlsen, Magnus","Nakamura, Hikaru"
]

# --- DATA PREPARATION ---
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
GAME_LENGTH = 100

# --- ENSURE STRINGS ---
data["white_name"] = data["white_name"].astype(str)
data["black_name"] = data["black_name"].astype(str)

# --- PLAYER MATCH FUNCTION ---
def player_match(name: str):
    lowered = name.lower()
    for player in NEW_LIST_OF_PLAYERS_MANUAL:
        if player.lower() == lowered:
            return player
    return None

# --- CREATE WHITE AND BLACK LABELS ---
data["WhiteLabel"] = data["white_name"].apply(player_match)
data["BlackLabel"] = data["black_name"].apply(player_match)

# Keep games where at least one player is in the manual list
mask = data["WhiteLabel"].notna() | data["BlackLabel"].notna()
data = data[mask].reset_index(drop=True)

# --- MAP PLAYERS TO INTEGER LABELS ---
encodep = dict(zip(NEW_LIST_OF_PLAYERS_MANUAL, range(len(NEW_LIST_OF_PLAYERS_MANUAL))))
decodep = {v: k for k, v in encodep.items()}

data["WhiteLabelID"] = data["WhiteLabel"].map(encodep)
data["BlackLabelID"] = data["BlackLabel"].map(encodep)

# --- TOKENIZATION ---
cleaner = str.maketrans({"[": "", "]": "", "'": "", ",": ""})
all_step = [k for s in data["list_of_moves"] for k in s.translate(cleaner).split()]
frequency = Counter(all_step)
dir = {"<PAD>": 2, "<UNK>": 3}
side_tokens = {"white": 0, "black": 1}
dir.update(side_tokens)
dir.update({move: len(dir) + i for i, move in enumerate(frequency)})

# --- STEP ENCODING ---
def step_encode(step, side_token=None):
    cleaned = step.translate(cleaner)
    tokening = cleaned.split()[:GAME_LENGTH]  # truncate only
    vector = list(map(lambda i_token: dir.get(i_token, 1), tokening))
    if side_token is not None:
        vector = [side_token] + vector
    # do NOT pad here
    return vector

# --- DATASET ---
class GameSequence(Dataset):
    def __init__(self, df):
        self.samples = []
        self.moves = df["list_of_moves"].to_list()
        self.white_labels = df["WhiteLabelID"].to_list()
        self.black_labels = df["BlackLabelID"].to_list()

        for i in range(len(df)):
            if not pd.isna(self.white_labels[i]):
                self.samples.append((self.moves[i], 0, int(self.white_labels[i])))
            if not pd.isna(self.black_labels[i]):
                self.samples.append((self.moves[i], 1, int(self.black_labels[i])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        moves, side_token, label = self.samples[idx]
        x = torch.tensor(step_encode(moves, side_token=side_token), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

gs_data = GameSequence(data)

# --- Basic stats ---
total_samples = len(gs_data)
num_white = sum(1 for s in gs_data.samples if s[1] == 0)
num_black = sum(1 for s in gs_data.samples if s[1] == 1)

print(f"Total games in original data: {len(data)}")
print(f"Total samples (should be <= 2x games): {total_samples}")
print(f"White samples: {num_white}")
print(f"Black samples: {num_black}")

# --- Check first few samples ---
for i in range(min(6, total_samples)):
    moves, side_token, label = gs_data.samples[i]
    side_name = "White" if side_token == 0 else "Black"
    label_name = decodep[label]
    print(f"Sample {i}: side={side_name}, label={label_name}, first 10 tokens={step_encode(moves, side_token=side_token)[:10]}")
    
# --- THE RNN MODEL ---
# https://www.nature.com/articles/s41598-025-88378-6
# Applying attention to LSTM outputs
class RecurrentNN(nn.Module):
    def __init__(self, dir, dropout, lstm_layers, dim_embedded, dim_hidden_layer, dim_out):
        print(dim_out)
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

# --- COLLATE FUNCTION (dynamic padding per batch) ---
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = [torch.tensor(x, dtype=torch.long) if not isinstance(x, torch.Tensor) else x for x in xs]
    padded_x = pad_sequence(xs, batch_first=True, padding_value=2)
    ys = torch.tensor(ys, dtype=torch.long)
    return padded_x, ys

labels = [s[2] for s in gs_data.samples]  # s[2] is the label in (moves, side_token, label)

train_indices, val_test_indices = train_test_split(
    range(len(gs_data)),
    test_size=VALIDATION_SPLIT + TEST_SPLIT,
    stratify=labels,  # stratify using pre-expanded sample labels
    random_state=123
)

validation_indices, test_indices = train_test_split(
    val_test_indices,
    test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT),
    stratify=[labels[i] for i in val_test_indices],
    random_state=123
)

train_data = Subset(gs_data, train_indices)
validation_data = Subset(gs_data, validation_indices)
test_data = Subset(gs_data, test_indices)
print("Data subsets created...")

# Cross Entropy loss (ideal and simple for classification tasks)
loss_fn = nn.CrossEntropyLoss()

# For RNN's, ADAM is the way to go.
def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer,
                   scheduler,
                   device: torch.device):
    model.train() # training mode activation before updating gradients

    # Initialize variables
    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # used for f1 score and accuracy metrics
    pred_labels_train = []
    true_labels_train = []

    for xbatch, ybatch in train_loader: # iterating batches
        xbatch = xbatch.to(device)
        ybatch = ybatch.to(device)

        # reset from last batch
        optimizer.zero_grad()
        output = model(xbatch)
        loss = loss_fn(output, ybatch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)

        batch_losses_train.append(loss.item())
        pred_labels_train.append(preds)
        true_labels_train.append(ybatch)

    # Format useful lists for calculation of metrics
    pred_labels_train = torch.cat(pred_labels_train, dim=0).detach().cpu().numpy().flatten()
    true_labels_train = torch.cat(true_labels_train, dim=0).detach().cpu().numpy().flatten()

    avg_train_loss = np.mean(batch_losses_train)

    # VALIDATE
    model.eval()

    # Initialize variables
    batch_losses_val = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # used for f1 score and accuracy metrics
    pred_labels_val = []
    true_labels_val = []

    with torch.no_grad(): # without gradient update for evaluation
        for xbatch, ybatch in validation_loader:
            xbatch = xbatch.to(device)
            ybatch = ybatch.to(device)

            output = model(xbatch)
            loss = loss_fn(output, ybatch)

            batch_losses_val.append(loss.item())

            preds = torch.argmax(output, dim=1)

            pred_labels_val.append(preds)
            true_labels_val.append(ybatch)

    avg_val_loss = np.mean(batch_losses_val)

    # Format useful lists for calculation of metrics
    pred_labels_val = torch.cat(pred_labels_val, dim=0).cpu().detach().numpy().flatten()
    true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy().flatten()

    return avg_train_loss, avg_val_loss, \
        (pred_labels_train, true_labels_train), \
        (pred_labels_val, true_labels_val)

def test(test_loader: DataLoader,
         model: nn.Module,
         device: torch.device):
    # Now we test on the test data at the end

    model.eval()

    # Initialize variables
    batch_losses_test = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # used for f1 score and accuracy metrics
    pred_labels_test = []
    true_labels_test = []

    with torch.no_grad(): # without gradient update for evaluation
        for xbatch, ybatch in test_loader:
            xbatch = xbatch.to(device)
            ybatch = ybatch.to(device)

            output = model(xbatch)
            loss = loss_fn(output, ybatch)

            batch_losses_test.append(loss.item())

            preds = torch.argmax(output, dim=1)

            pred_labels_test.append(preds)
            true_labels_test.append(ybatch)

    avg_test_loss = np.mean(batch_losses_test)

    # Format useful lists for calculation of metrics
    pred_labels_test = torch.cat(pred_labels_test, dim=0).cpu().detach().numpy().flatten()
    true_labels_test = torch.cat(true_labels_test, dim=0).cpu().detach().numpy().flatten()

    return avg_test_loss, (pred_labels_test, true_labels_test)


'''
To calculate different kind of metrics based on:
- Average (train/validation/test) loss
- Predicted labels for training/validation/test
- True labels for training/validation/test

Calculates:
- Classification metrics
    - F1 scores (macro, weighted, per-class)
    - Confusion matrix (and prints)
    - Classification report
    - Balanced accuracy score
- Regression metrics
    - Mean absolute error
    - (root) mean squared error
    - R2 score

Shows:
    - Confusion matrix
    - Predicted vs True regression plot
'''
def calculateMetrics(avg_loss : np.floating, predicted_labels : np.ndarray, true_labels : np.ndarray):
    # --- CLASSIFICATION ---
    '''
    Macro F1 = The average f1 score over all classes, treating each class equally.
    This score becomes more relevant when some players have very few games.
    '''
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

    '''
    Weighted F1 = Same as Macro F1, but is weighted by class frequency. It doesn't punish too hard for players with few games.
    '''
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")

    '''
    Per-class F1 scores, this doesn't average over all classes and shows how different players compare
    Is not printed, because classification_report already does it in a nice way, but wanted to include here
    Because it shows the relevance.
    '''
    per_class_f1 = f1_score(true_labels, predicted_labels, average=None)

    '''
    A full on confusion matrix of shape NxN.
    '''
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    '''
    Balanced accuracy = Each class contributes equally to the accuracy, better than the usual way of calculating accuracy: correct / total
    '''
    bal_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    print(
        f"--- CLASSIFICATION METRICS --- \n"
        f"F1 scores: [Macro={macro_f1:.3f}, Weighted={weighted_f1:.3f}] \n"
        f"Balanced Accuracy = {bal_accuracy:.3f}\n"
        f"Average loss = {avg_loss:.5f}")

    report = classification_report(true_labels, predicted_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    # print(tabulate(df.round(3), headers='keys', tablefmt="pretty"))
    # print(df.round(3))

    # Don't show plots for now...
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(conf_matrix, annot=False, cmap="Blues")
    # plt.title("Confusion matrix")
    # plt.show()
    
    return macro_f1, weighted_f1, bal_accuracy


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

'''
Function used for the grid search algorithm, input = grid search array
Initialization steps unaffected by the runGridPoint are NOT ran again. 
Runs the whole RNN loop (from creating data loaders to training the model, with specified hyperparamaters)
'''
def runGridPoint(grid_search_array : list, id : int, early_stop_patience=100, epochs=3, logging=True, workerID='A'):
    # Affected by Grid Search
    BATCH_SIZE = int(grid_search_array[0])
    DROPOUT = float(grid_search_array[1])
    HIDDEN_LAYER_DIM = grid_search_array[2]
    LSTM_LAYERS = int(grid_search_array[3])
    LEARNING_RATE = float(grid_search_array[4])
    EMBEDDED_LAYER_DIM = int(grid_search_array[5])
    
    # LEARNING_RATE = 0.001
    USE_LOGGING = logging

    # Learning rate increases/decreases based on batch size
    # if BATCH_SIZE == 256:
    #     LEARNING_RATE = 0.01    
    # elif BATCH_SIZE == 128:
    #     LEARNING_RATE = 0.005
    # elif BATCH_SIZE == 64:
    #     LEARNING_RATE = 0.001

    # Unaffected by grid search
    EPOCHS = epochs
    
    # EARLY STOPPING
    early_stop_counter = 0  # do not change
    early_stop_best_loss = torch.inf
    early_stop_best_model_state = None
    PATIENCE = early_stop_patience  # after how many epochs of no decrease in loss should we stop
    DELTA = 0.0  # if the loss decreases with maximum this delta, do not reset the counter
    
    # Reproducability
    torch.manual_seed(123)
    
    if USE_LOGGING:
        r_name = f"RUN {id:04d}"
        with mlflow.start_run(run_name=r_name):
            print(f"Starting new run {id}...")
            TRAIN_DATALOADER = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            VALIDATION_DATALOADER = DataLoader(validation_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
            TEST_DATALOADER = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

            mlflow.log_param('batch_size', BATCH_SIZE)
            mlflow.log_param('dropout', DROPOUT)
            mlflow.log_param('hidden_layer_dim', HIDDEN_LAYER_DIM)
            mlflow.log_param('lstm_layers', LSTM_LAYERS)
            mlflow.log_param('learning_rate', LEARNING_RATE)
            mlflow.log_param('embedding_dim', EMBEDDED_LAYER_DIM)

            model = RecurrentNN( # Building model
                dir=len(dir),
                dropout=DROPOUT,
                lstm_layers=LSTM_LAYERS,
                dim_embedded=EMBEDDED_LAYER_DIM,
                dim_hidden_layer=HIDDEN_LAYER_DIM,
                dim_out=len(encodep)).to(device)
            
            optimizer = torch.optim.Adam(params=list(model.parameters()), lr=LEARNING_RATE)
                
            # --- TRAINING, VALIDATION ---
            print(f"Beginning training... using {device} device")
            for iEpoch in range(EPOCHS):
                t_loss, v_loss, (pltrain, tltrain), (plval, tlval)\
                = train_validate(train_loader=TRAIN_DATALOADER,
                                validation_loader=VALIDATION_DATALOADER,
                                model=model,
                                optimizer=optimizer,
                                scheduler=None,
                                device=device)

                print(f"Epoch {iEpoch}, training metrics:")
                t_macro_f1, t_weighted_f1, t_balanced_acc = calculateMetrics(t_loss, pltrain, tltrain)

                print(f"Epoch {iEpoch}, validation metrics:")
                v_macro_f1, v_weighted_f1, v_balanced_acc = calculateMetrics(v_loss, plval, tlval)

                mlflow.log_metric('train/loss', t_loss, step=iEpoch)
                mlflow.log_metric('train/macro_f1', t_macro_f1, step=iEpoch)
                mlflow.log_metric('train/weighted_f1', t_weighted_f1, step=iEpoch)
                mlflow.log_metric('train/balanced_acc', t_balanced_acc, step=iEpoch)
                mlflow.log_metric('val/loss', v_loss, step=iEpoch)
                mlflow.log_metric('val/macro_f1', v_macro_f1, step=iEpoch)
                mlflow.log_metric('val/weighted_f1', v_weighted_f1, step=iEpoch)
                mlflow.log_metric('val/balanced_acc', v_balanced_acc, step=iEpoch)
            
                metrics_dict = {
                    "avg_train_loss": t_loss,
                    "avg_val_loss": v_loss,
                    "pred_labels_train": pltrain,
                    "true_labels_train": tltrain,
                    "pred_labels_val": plval,
                    "true_labels_val": tlval,
                }
                
                folder_path = f"/metrics/run_{id:04d}_{workerID}"
                os.makedirs(folder_path, exist_ok=True)
                
                with open(f"/metrics/run_{id:04d}_{workerID}/epoch_{iEpoch}.json", "w") as f:
                    json.dump(to_serializable(metrics_dict), f, indent=2)

                early_stop_best_model_state = model.state_dict()

                # -- EARLY STOPPING CHECK --
                if v_loss < early_stop_best_loss - DELTA:
                    # A better loss was found, so reset counter and save model state
                    early_stop_best_loss = v_loss
                    early_stop_counter = 0
                    # Save the best model so we can restore it later and get the best model performance to use the test data for.
                    early_stop_best_model_state = model.state_dict()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= PATIENCE:
                        print(f"Early stopping...")

                        # Restore the best model which was saved earlier.
                        model.load_state_dict(early_stop_best_model_state)
                        break

            # torch.save(model.state_dict(), f"/best_models/RNN_best_model_run{id:04d}_{workerID}.pth")

            # --- TESTING ---

            avg_test_loss, (pred_labels_test, true_labels_test) = test(test_loader=TEST_DATALOADER,
                                model=model,
                                device=device)
            print(f"Testing metrics:")
            t_macro_f1, t_weighted_f1, t_balanced_acc = calculateMetrics(avg_test_loss, pred_labels_test, true_labels_test)
            
            test_metrics_dict = {
                "avg_test_loss": avg_test_loss,
                "pred_labels_test": pred_labels_test,
                "true_labels_test": true_labels_test,
            }

            with open(f"/metrics/run_{id:04d}_{workerID}/test_metrics.json", "w") as f:
                json.dump(to_serializable(test_metrics_dict), f, indent=2)

            mlflow.log_metric('test/loss', avg_test_loss)
            mlflow.log_metric('test/macro_f1', t_macro_f1)
            mlflow.log_metric('test/weighted_f1', t_weighted_f1)
            mlflow.log_metric('test/balanced_acc', t_balanced_acc)
    else:
        print(f"Starting new run, NO LOGGING...")
        TRAIN_DATALOADER = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        VALIDATION_DATALOADER = DataLoader(validation_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        TEST_DATALOADER = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        model = RecurrentNN( # Building model
            dir=len(dir),
            dropout=DROPOUT,
            lstm_layers=LSTM_LAYERS,
            dim_embedded=EMBEDDED_LAYER_DIM,
            dim_hidden_layer=HIDDEN_LAYER_DIM,
            dim_out=len(encodep)).to(device)
        
        optimizer = torch.optim.Adam(params=list(model.parameters()), lr=LEARNING_RATE)
            
        # --- TRAINING, VALIDATION ---
        print(f"Beginning training... using {device} device")
        for iEpoch in range(EPOCHS):
            t_loss, v_loss, (pltrain, tltrain), (plval, tlval)\
            = train_validate(train_loader=TRAIN_DATALOADER,
                            validation_loader=VALIDATION_DATALOADER,
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            device=device)

            print(f"Epoch {iEpoch}, training metrics:")
            t_macro_f1, t_weighted_f1, t_balanced_acc = calculateMetrics(t_loss, pltrain, tltrain)

            print(f"Epoch {iEpoch}, validation metrics:")
            v_macro_f1, v_weighted_f1, v_balanced_acc = calculateMetrics(v_loss, plval, tlval)
            
            early_stop_best_model_state = model.state_dict()
            
            metrics_dict = {
                    "avg_train_loss": t_loss,
                    "avg_val_loss": v_loss,
                    "pred_labels_train": pltrain,
                    "true_labels_train": tltrain,
                    "pred_labels_val": plval,
                    "true_labels_val": tlval,
                }
                
            folder_path = f"/metrics/run_{r_name}"
            os.makedirs(folder_path, exist_ok=True)
            
            with open(f"/metrics/run_{r_name}/epoch_{iEpoch}.json", "w") as f:
                json.dump(to_serializable(metrics_dict), f, indent=2)

            # -- EARLY STOPPING CHECK --
            if v_loss < early_stop_best_loss - DELTA:
                # A better loss was found, so reset counter and save model state
                early_stop_best_loss = v_loss
                early_stop_counter = 0
                # Save the best model so we can restore it later and get the best model performance to use the test data for.
                early_stop_best_model_state = model.state_dict()
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print(f"Early stopping...")

                    # Restore the best model which was saved earlier.
                    model.load_state_dict(early_stop_best_model_state)
                    break

        # --- TESTING ---
        avg_test_loss, (pred_labels_test, true_labels_test) = test(test_loader=TEST_DATALOADER,
                            model=model,
                            device=device)
        print(f"Epoch {iEpoch}, testing metrics:")
        t_macro_f1, t_weighted_f1, t_balanced_acc = calculateMetrics(avg_test_loss, pred_labels_test, true_labels_test)
        
        test_metrics_dict = {
            "avg_test_loss": avg_test_loss,
            "pred_labels_test": pred_labels_test,
            "true_labels_test": true_labels_test,
            }

        with open(f"/metrics/run_{r_name}/test_metrics.json", "w") as f:
            json.dump(to_serializable(test_metrics_dict), f, indent=2)



'''
This code thinks about how to implement grid search (for our RNN model), given the lecture
given by Kaare about Hyperparameter Search. 

Notes on the lecture / things to keep in mind:
- Random grid search is probably not the way to go since big gaps can occur
- Rotated grid search sounds really promising, also given the paper "Rotated Grid Search for Hyperparamater Optimization" by Allawala et al.
    -> However, it's complicating to think about rotations in >2D space
- Make sure to search the full range, there will be regions of trainability, which we can search better afterwards
- Models than end low often drop in early epochs. Hence,
    -> Optimize searching algorithm by not training unpromising hyperparameters
- Learning rate scheduler is probably a good idea in the end if we are plateauing
    - Cyclical schedulers seem very promising, but for RNN's it may be too complex, as training in general does not take that long

Hyperparameter specific notes:
- Bigger learning rate is already regularization form
    - Decrease other regularization methods (dropout, batch normalization), works vice versa too!
- Batch size -> If using batch_loss as guidance, set reduction to mean. 
    - Also, store number of time passed when comparing batch sizes, smaller batch sizes usually take less epochs

'''
# --- GRID SEARCH PARAMETERS ---

# Questions:
# Should we tokenize on a random 80% split, or should we tokenize within the fold?

# These will have huge gaps, to see where the most potential lies.

# A
# initial_grid_search_parameters = {
#     'batch_size': [32, 128, 256],
#     'dropout': [0.1, 0.4],
#     'hidden_layer_dim': [64],
#     'lstm_layers': [2, 3],
#     'learning_rate': [1e-4, 1e-3, 1e-2],
#     'embedded_layer_dim': [64, 128],
# }

# B
# initial_grid_search_parameters = {
#     'batch_size': [32, 128, 256],
#     'dropout': [0.1, 0.4],
#     'hidden_layer_dim': [128],
#     'lstm_layers': [2, 3],
#     'learning_rate': [1e-4, 1e-3, 1e-2],
#     'embedded_layer_dim': [64, 128],
# }

# C
initial_grid_search_parameters = {
    'batch_size': [32, 128, 256],
    'dropout': [0.1, 0.4],
    'hidden_layer_dim': [256],
    'lstm_layers': [2, 3],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'embedded_layer_dim': [64, 128],
}

# USED LATER
# concentrated_grid_search_parameters = {
#     'batch_size': [64, 128, 256],
#     'dropout': [0.2, 0.3, 0.4, 0.5],
#     'hidden_layer_dim': [64, 128, 256],
#     'lstm_layers': [2, 3],
# }

def return_combinations(dict_hyperparameters):
    # We want to create all possible combinations, but systematically so we can interrupt if we don't see good results
    all_parameters = ([list(x) for x in dict_hyperparameters.values()])

    return list(itertools.product(*all_parameters))

# print(len(return_combinations(initial_grid_search_parameters)))

# Run MLFlow locally on port 5000, set IP address here:
mlflow.set_tracking_uri(
    "http://84.238.41.58:5000"
)

# mlflow.set_experiment("RNN_OvernightGridSearch_A")
# mlflow.set_experiment("RNN_OvernightGridSearch_B")
mlflow.set_experiment("RNN_OvernightGridSearch_C")

for i, comb in enumerate(return_combinations(initial_grid_search_parameters)):
    runGridPoint(comb, id=i, early_stop_patience=100, epochs=3, logging=True, workerID='C')

# runGridPoint([64, 0.2, 64, 2, 1e-3, 64], id=3, early_stop_patience=10, epochs=2, logging=True, workerID='A')