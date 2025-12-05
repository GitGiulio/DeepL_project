from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import mlflow
import random

# Local dependencies
from RNN_grid_searches import return_rotated_combinations, return_combinations, run_grid_point
import Chess_RNN

'''
@author Mathijs Tobé, Nikan Mahdavi Tabatabaei, Peter Normann Diekema 

This is the file that is ran by LUMI, using all of the libraries and functions written and splitted into the
other python scripts within this folder. The dataloading and tokenization are handled in this script.

Everything was brought together in this script.

Important note: The final version of this code is adjusted to be able to run on LUMI (path names, devices, etc.)
BUT: Old artifacts left from runs on other devices are left to show you what we have done.
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We loved MLFlow! Thanks for the recommendation. @author Mathijs Tobé kept his laptop on 24/7 for about a week to
# have the server running while running the big transformer models but also the RNNs on LUMI. 
mlflow.set_tracking_uri(
    "http://hahabutimnotshowingmyIP:5000"  # port you can have :P 
)

print(f"Using device {device}")

# Import games CSV, this really depends on where the code was run (locally, colab, lightning.ai, LUMI)

# data_path = "C:\\Users\\mathi\\Documents\\University\\Aarhus University\\MSc Computer Engineering\\Semester 1\\Deep Learning\\project\\DeepL_project\\data\\filtered_games_20_players.csv"
# project_root = "/teamspace/studios/this_studio/DeepL_project"
# data_path = "/filtered_games_new.csv"
data_path = "/filtered_games_100_players.csv"

# Data loading
print("Loading data...")
data = pd.read_csv(data_path) # loading into dataframe
print("Data loaded...")


# The commented list below contains 20 players, mainly used for testing in the initial phases of the model.
# NEW_LIST_OF_PLAYERS_MANUAL = [
#     'ArasanX','MassterofMayhem','JelenaZ','lestri','doreality','therealYardbird',
#     'Chesssknock','No_signs_of_V','Recobachess','drawingchest','kasparik_garik',
#     'ChainsOfFantasia','Consent_to_treatment','Alexandr_KhleBovich','unknown-maestro_2450',
#     'gefuehlter_FM','gmmitkov','positionaloldman',"Carlsen, Magnus","Nakamura, Hikaru"
# ]

# Contains 100 players, filtered so that they are NOT BOTS
NEW_LIST_OF_PLAYERS_MANUAL = ['ArasanX', 'MassterofMayhem', 'JelenaZ', 'lestri', 'doreality', 'therealYardbird', 'Chesssknock',
'No_signs_of_V', 'Recobachess', 'drawingchest', 'kasparik_garik', 'ChainsOfFantasia','Consent_to_treatment',
'Alexandr_KhleBovich', 'unknown-maestro_2450', 'gefuehlter_FM', 'gmmitkov', 'positionaloldman','Consent_to_treatment',
'Gyalog75','chargemax23','Boreminator','sotirakis','cn_ua','anhao','manuel-abarca','Chess_diviner',
'Toro123','Odirovski','manneredmonkey','Viktor_Solovyov','Stas-2444','Zhigalko, Sergei','AKS-Mantissa',
'vistagausta','Romsta','Aborigen100500','JoeAssaad','bodoque50','doreality1991','Niper13','Violet_Pride','Ivanoblitz','Atalik, Suat',
'iakov98','AlexD64','satlan','Bakayoyo','athena-pallada','Pblu35','okriak','morus22','Corre_por_tu_vida','Attila76',
'Karlos_ulsk','www68','Podrebo','papasi','crackcubano','Chessibague','Konstrictor','EleKtRoMaGnIt','snayperchess','ZhohoFF',
'dont_blunder','Ute-Manfred','Konkurs_prognozov','Mischuk_D','kenkons','notkevich','Elda64','Konnov, Oleg','DrawDenied_Twitch',
'Vnebo','Leviathan64','VonSinnen','SpiderMoves','econpower','Napo18','KQRBNPkqrbnp','kirlianitalian','DOCTAL','bingo95',
'smurf42','IDISJUDA','Lightlike','Enialios','miki_train','Nguyen, Duc Hoa','FlaggingIsADisease','MrSOK','wuzhaobing',
'Barry_Lindon','cutemouse83','Rumple_DarkOne','ElexusChess','Herushkanamura','Yarebore',"Carlsen, Magnus","Nakamura, Hikaru"]

# --- DATA PREPARATION ---
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# How long should games be when passed to the RNN
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

# We tokenize with all available data, because then we are pretty sure all possible moves are tokenized.
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
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        moves, side_token, label = self.samples[idx]
        x = torch.tensor(step_encode(moves, side_token=side_token), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# An option for if that data must be balanced, and if so, how many games to pick PER PLAYER.
balanced = False
N = 8020  # this is the minimum number in the current dataset with 20 players
all_samples = []  # will be the balanced data

random.seed(123)  # for consistent testing over many different runs

for player in NEW_LIST_OF_PLAYERS_MANUAL:
    pid = encodep[player]

    player_games = data[(data["WhiteLabelID"] == pid) | (data["BlackLabelID"] == pid)]
    player_samples = []

    for _, row in player_games.iterrows():
        moves = row["list_of_moves"]

        # White
        if row["WhiteLabelID"] == pid:
            player_samples.append((moves, 0, pid))
        elif row["BlackLabelID"] == pid:  # player is Black
            player_samples.append((moves, 1, pid))
    
    random.shuffle(player_samples)

    if balanced:
        all_samples.extend(player_samples[:N])
    else:
        all_samples.extend(player_samples)

# Returns the dataset containing all the tokenized games.
gs_data = GameSequence(all_samples)

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
    
# --- COLLATE FUNCTION (dynamic padding per batch) ---
'''
@author Peter Normann Diekema
'''
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

mlflow.set_experiment("RNN_Grid_Search")


# How the initial grid search was run:

# --- INITIAL GRID SEARCH ---

# This leads to 3*2*3*2*3*2 = 216 combinations, which was separated into 3 workers
initial_grid_search_parameters = {
    'batch_size': [32, 128, 256],
    'dropout': [0.1, 0.4],
    'hidden_layer_dim': [64, 128, 256],
    'lstm_layers': [2, 3],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'embedded_layer_dim': [64, 128],
}

# Should speak for itself
# We ran the initial grid search on 3 epochs only, because of limited resources on LUMI.
# On average, 3 epochs took about 8 minutes to run.
# In the actual run, the combinations were split up in 3 'workers', all working simultaneously, with different 'workerID's, but uploading
# to the same experiment name on MLFlow.

WEIGHT_DECAY = 0.05
SCHEDULER_FACTOR = 0.33

for i, comb in enumerate(return_combinations(initial_grid_search_parameters)):
    comb.append(WEIGHT_DECAY)  # weight_decay
    comb.append(SCHEDULER_FACTOR)  # learning_rate scheduler factor (divide by 3 if loss does not decrease in 2 epochs)
    run_grid_point(
        grid_search_array=comb,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        collate_fn=collate_fn,
        RNN_model=Chess_RNN.Chess_RNN,
        device=device,
        len_dir=len(dir),
        classes_len=len(NEW_LIST_OF_PLAYERS_MANUAL),
        early_stop_patience=3,  # no early stopping, does not reach this
        epochs=3,
        seed=123,
        id=i,
        workerID='A'
    )

# --- ROTATED GRID SEARCH ---
hyper_ranges = {
    'batch_size': (64, 256),
    'hidden_layer_dim': (40, 160),
    'embedding_size': (90, 150),
}

# All keys in the rotated grid should be rounded to nearest integer.
integer_keys = ['batch_size', 'hidden_layer_dim', 'embedding_size']

# Generate rotated grid
grid_combos = return_rotated_combinations(hyper_ranges, m=6, theta_deg=20, integer_keys=integer_keys)


'''
# A lot of scrap code showing how we ran grid searches with multiple workers on LUMI
# Meaning to run 'sbatch run_rnn.sh' various times, but every time slightly changing the variables

# 120 possibilities, have 4 workers all doing 30 possibilities. 
workerA = grid_combos[1:30]
workerB = grid_combos[30:60]
workerC = grid_combos[60:90]
workerD = grid_combos[90:]

print("STARTING ROTATED GRID SEARCH FOR WORKER D!")

counter = 90

mlflow.set_experiment("Rotated_D")

for i in workerD:
    comb = list(i.values())
    runGridPoint(comb, id=counter, early_stop_patience=3, epochs=15, workerID='D')
    counter += 1
batch_size, hidden_layer, embedded layer

# Some individual tests after the rotated grid search, to see performance, take this with a grain of salt
# Since the actual results are presented nicely in the report.

# runGridPoint([32, 20, 60], id=100, early_stop_patience=5, epochs=30, workerID='Z') # terrible
# runGridPoint([64, 12, 80], id=101, early_stop_patience=5, epochs=30, workerID='Z') # terrible 
# runGridPoint([64, 16, 80], id=102, early_stop_patience=5, epochs=30, workerID='Z') # terrible
# runGridPoint([64, 20, 80], id=103, early_stop_patience=5, epochs=30, workerID='Z') # terrible
# runGridPoint([150, 150, 150], id=104, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.01  -- TERRIBLE
# runGridPoint([64, 150, 150], id=105, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.01 --- TERRIBLE 
# runGridPoint([64, 70, 150], id=106, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.01
# runGridPoint([64, 80, 80], id=107, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.01
# runGridPoint([64, 50, 100], id=108, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.01
# runGridPoint([64, 150, 150], id=109, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1 --- NOT THE WORST
# runGridPoint([32, 100, 100], id=110, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1 --- NOT THE WORST
# runGridPoint([48, 256, 128], id=111, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1
# runGridPoint([48, 256, 256, 2], id=112, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.2
# runGridPoint([64, 384, 128], id=113, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1
# runGridPoint([32, 100, 100, 3], id=114, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1
# runGridPoint([64, 256, 128, 3], id=115, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1
# runGridPoint([32, 100, 100, 2], id=116, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1, factor scheduler 0.5 --- NOT THE WORST
# runGridPoint([32, 100, 100, 2], id=117, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1, dropout 0.2, factor 0.5
# runGridPoint([48, 256, 128, 2], id=118, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1, dropout 0.2, factor 0.5
# runGridPoint([32, 256, 128, 2], id=119, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1, dropout 0.2, factor 0.5
# runGridPoint([32, 256, 86, 2], id=120, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.1, dropout 0.2, factor 0.5
# runGridPoint([48, 256, 128, 2], id=121, early_stop_patience=15, epochs=30, workerID='Z')  # ADAMW, weight decay 0.05 on linear layers, not embeddings or norm
# runGridPoint([32, 256, 128, 2], id=122, early_stop_patience=15, epochs=30, workerID='Z')  # ADAMW, weight decay 0.05 on linear layers, not embeddings or norm, dropout 0.2, factor 0.5
# runGridPoint([32, 256, 86, 2], id=123, early_stop_patience=5, epochs=30, workerID='Z')  # ADAMW, weight decay 0.05 on linear layers, not embeddings or norm, dropout 0.2, factor 0.5
# runGridPoint([48, 256, 128, 2], id=124, early_stop_patience=15, epochs=30, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([32, 100, 100, 2], id=125, early_stop_patience=15, epochs=30, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([32, 128, 64, 2], id=126, early_stop_patience=15, epochs=50, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([32, 256, 128, 2], id=127, early_stop_patience=15, epochs=50, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([64, 256, 128, 1], id=128, early_stop_patience=15, epochs=50, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([64, 256, 128, 1], id=129, early_stop_patience=15, epochs=50, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([128, 350, 64, 1], id=130, early_stop_patience=15, epochs=50, workerID='Z')  # ADAMW, weight decay 0.15 on linear layers, not embeddings or norm, dropout 0.4, factor 0.3
# runGridPoint([64, 80, 80, 2], id=201, early_stop_patience=10, epochs=30, workerID='Z')

# runGridPoint([64, 128, 128, 1], id=202, early_stop_patience=10, epochs=30, workerID='Z')

# Again, unbalanced data now
# runGridPoint([64, 128, 128, 2], id=203, early_stop_patience=10, epochs=30, workerID='Z')

# runGridPoint([64, 90, 90, 2], id=204, early_stop_patience=10, epochs=30, workerID='Z')

# runGridPoint([16, 80, 80, 1], id=205, early_stop_patience=10, epochs=30, workerID='Z')

# runGridPoint([128, 150, 150, 2], id=206, early_stop_patience=10, epochs=30, workerID='Z')
'''

