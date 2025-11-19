'''
Author: Mathijs Tobe

This file focuses solely on independent training and visualisation, 
including some accuracy metrics. It also splits the data into training/validation,
but assumes the data is splitted into trainX, trainy, testX, testy already.
Most of the training loop is based on my Assignment 2 submission. 

Notes / findings:
- Top-k accuracy was proposed by many other students in our pitch feedback,
would this require a top-k loss function too? Where are our priorities?
- 
'''
import torch
import numpy as np
from train_val_test_funcs import train_validate, test
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# --- UNIMPORTANT DECLARATIONS OUTSIDE OF THE SCOPE OF THIS FILE ---
# Most of the things defined here will probably be defined somewhere earlier
# But it just shows what this code needs in order to run successfully
device = torch.device('cpu')
trainX, trainy, testX, testy = [], [], [], []
model = torch.nn.Transformer().to(device)  # random
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# --- IMPORTANT CONSTANTS THAT YOU CAN ACTUALLY TWEAK ---
# all_games (100%)
# ├─ no_test ((1-TEST_SPLIT) * 100%)
# │  ├─ training ((1-VALIDATION_SPLIT) * 100%)
# │  └─ validation (VALIDATION_SPLIT * 100%)
# └─ test (TEST_SPLIT * 100%)
VALIDATION_SPLIT = 0.2  # hence, will be a percentage of the no_test data

# As stated in slide 42 of lecture 11, make sure to use 'reduction='mean' for the loss function when comparing this hyperparameter
# If it is 'sum', a larger batch size automatically means a larger loss because there are more samples
BATCH_SIZE = 16

# This is to enable better hyperparameter tuning
SHUFFLE_SEED = 9999  

EPOCHS = 10

# EARLY STOPPING
early_stop_counter = 0  # do not change
early_stop_best_loss = torch.inf
early_stop_best_model_state = None
PATIENCE = 5  # after how many epochs of no decrease in loss should we stop
DELTA = 1e-3  # if the loss decreases with maximum this delta, do not reset the counter

# Other hyperparameters
USE_LEARNING_RATE_SCHEDULER = False
if not USE_LEARNING_RATE_SCHEDULER:
    scheduler = None

USE_TOP_K_ACCURACY = True
K_ACCURACY = 3

# --- SETTING UP TRAINING / VALIDATION DATA ---
n_no_test = trainX.shape[0]
n_validation = int(n_no_test * VALIDATION_SPLIT)

# Shuffle
torch.manual_seed(SHUFFLE_SEED)
indices = torch.randperm(n_no_test)

# Split
validation_indices = indices[:n_validation]
train_indices = indices[n_validation:]

x_val, y_val = trainX[validation_indices], trainy[validation_indices]
x_train, y_train = trainX[train_indices], trainy[train_indices]

# DATALOADERS ARE SUPPOSED TO BE DEFINED SOMEWHERE ELSE PROBABLY
# DEPENDS ON DATA :)
# Setup data loaders for given BATCH_SIZE
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = torch.utils.data.TensorDataset(x_val, y_val)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(testX, testy)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# TODO visualizing the training data based on player / total number of moves or something similar

# --- TRAINING, VALIDATION ---
train_losses, val_losses = [], []

for iEpoch in range(EPOCHS):
    avg_train_loss, avg_val_loss \
        = train_validate(train_loader=train_loader, 
                        validation_loader=validation_loader,
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        scheduler=scheduler)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {iEpoch}:\
            \n<Insert accuracy function values here>\
            \nTraining loss {avg_train_loss:.3f}, Validation loss {avg_val_loss:.3f}")

    # -- EARLY STOPPING CHECK --
    if avg_val_loss < early_stop_best_loss - DELTA:
        # A better loss was found, so reset counter and save model state
        early_stop_best_loss = avg_val_loss
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
avg_test_loss = test(test_loader=test_loader,
                     model=model,
                     loss_fn=loss_fn,
                     device=device)
print(f"\n<Insert accuracy function values here>\
        Test loss {avg_val_loss:.3f}")

# --- VISUALIZATION ---

# may be splitted into a different file too, if it gets too complicated.
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.show()