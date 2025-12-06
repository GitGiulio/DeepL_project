import numpy as np
import math
import itertools
import torch
import mlflow
import json
import os

# local dependencies
from RNN_training_test import train_validate, test, calculate_metrics

'''
@author Mathijs Tobé & Peter Normann Diekema

No grid search is being done for the transformer, as this model is overall too complex to run various
combinations on. However, for the RNN we really wanted to focus on how to properly do a grid search,
by first doing a general, broader scale search, and later focusing on the 'promising ranges' by also
including the idea of rotation to get unique hyperparameters in each search. 

Initial notes on the lecture / things to keep in mind:
- Random grid search is not the way to go since big gaps can occur
- Rotated grid search sounds really promising, also given the paper "Rotated Grid Search for Hyperparamater Optimization" by Allawala et al.
    -> However, it's complicating to think about rotations in >2D space
- Make sure to search the full range, there will be regions of trainability, which we can search better afterwards
- Models than end low often drop in early epochs. Hence,
    -> Optimize searching algorithm by not training unpromising hyperparameters
- Learning rate scheduler is a good idea in the end if we are plateauing

Hyperparameter specific notes:
- Bigger learning rate is already regularization form
    - Decrease other regularization methods (dropout, batch normalization), works vice versa too!
- Batch size -> If using batch_loss as guidance, set reduction to mean. 
    - Also, store number of time passed when comparing batch sizes, smaller batch sizes usually take less epochs

Containing all the required functions for doing HYPERPARAMETER GRID SEARCH for the RNN model:
- a regular grid search (no randomness, just set combinations from given ranges of hyperparameters)
- a rotated grid search (novel idea, but interesting)
'''

'''
    @author: Peter Normann Diekema - With help of ChatGPT for rotation in N dimensions logic and understanding
    Rotated grid search for RNN specific hyperparameter search 
    
    Shoutout to Kaare for motivating us during the hyperparameter search lecture to create this :D

    ranges: dict {param_name: (low, high)}
    m: points to generate per dimension
    theta_deg: rotation angle
    integer_keys: list of keys to round to integer (like batch_size, hidden layer dimensions)
'''
def return_rotated_combinations(ranges, m=6, theta_deg=20, integer_keys=[]) -> list:
   
    keys = list(ranges.keys())  # get keys to generate points for
    dims = len(keys)  # the number of dimensions

    # first create a grid with the given dimensions, spanning 0-1 with m equal jumps in this range
    # the center will be 0.5, and we use this to rotate the grid around the center, instead of (0,0).
    axis_points = [np.linspace(0, 1, m) for _ in range(dims)]  
    base_grid = np.array(list(itertools.product(*axis_points)))

    # we want to rotate with theta_deg angle (20 degrees)
    theta = math.radians(theta_deg)
    rotated_grid = []

    center = 0.5
    for p in base_grid:
        # first deep copy the p_rot 
        p_rot = p.copy()
        # go over all dimensions, apply rotation to each 2D pair with M dimensions
        # https://matthew-brett.github.io/teaching/rotation_2d.html  (explained how to rotate a vector)
        for i in range(dims):
            for j in range(i + 1, dims):
                x_shift = p_rot[i] - center
                y_shift = p_rot[j] - center
                x_rot = x_shift * np.cos(theta) - y_shift * np.sin(theta)
                y_rot = x_shift * np.sin(theta) + y_shift * np.cos(theta)
                p_rot[i] = x_rot + center
                p_rot[j] = y_rot + center
        rotated_grid.append(p_rot)

    rotated_grid = np.array(rotated_grid)

    # make sure that only hyperparameters WITHIN the specified range (eventually), end up in what you return
    rotated_grid = np.array([p for p in rotated_grid if np.all((p >= 0) & (p <= 1))])

    # get the lower and upper bounds of the needed hyperparameters
    lo_vals = np.array([ranges[k][0] for k in keys])
    hi_vals = np.array([ranges[k][1] for k in keys])
    combos = []
    # this function basically maps all the low and high values and their m 'steps' in between
    # to the rotated grid on scale [0, 1], so we get them in the right range. 
    for p in rotated_grid:
        cfg = {}
        for idx, k in enumerate(keys):
            val = lo_vals[idx] + p[idx] * (hi_vals[idx] - lo_vals[idx])
            if k in integer_keys:
                val = int(round(val))
            cfg[k] = val
        combos.append(cfg)

    return combos

'''
@author Mathijs Tobé

Trivial function to calculate all possible combinations of a dict of hyperparameters.
'''
def return_combinations(dict_hyperparameters) -> list:
    all_parameters = ([list(x) for x in dict_hyperparameters.values()])

    return list(itertools.product(*all_parameters))

'''
@author Giulio

Converts object to JSON for possible eventual parsing locally.

However, this is not really used, since MLFlow proved to be very useful.
'''
def to_serializable(obj):
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
@author Mathijs Tobé
    
Function used for the grid search algorithm
Initialization steps unaffected by the runGridPoint are NOT ran here, and should be provided to the model. 
Runs the whole RNN loop (from creating data loaders to training the model, with specified hyperparamaters)

params: 
grid_search_array : [batch_size, dropout, hidden_layer_dim, lstm_layers, learning_rate, embedded_layer_dim, weight_decay, scheduler_factor]
train/validation/test_data      -> Data sets initialized before
collate_fn                      -> The collate function used for custom batch embeddings and paddings
RNN_model                       -> The custom Recurrent neural network used for the grid search
device                          -> On what device are we running (usually cuda)
len_dir                         -> After the tokenization, how many tokens are there in total
classes_len                     -> How many classes do we need to classify?
epochs : int                    -> For how many epochs should this model maximally run
early_stop_patience : int       -> After how many epochs of no improvement in validation loss should we stop
seed : int                      -> What torch random seed we should use (to get rid of randomness variable in results)
id : int                        -> The ID of this current run (used for logging)
workerID : str                  -> What LUMI instance is currently running this (used for logging)
'''
def run_grid_point(
                grid_search_array : list,
                train_data : torch.utils.data.Subset,
                validation_data : torch.utils.data.Subset,
                test_data : torch.utils.data.Subset,
                collate_fn, 
                RNN_model : torch.nn.Module,
                device : torch.device,
                len_dir : int,
                classes_len : int = 20,
                early_stop_patience : int = 100, 
                epochs : int = 3, 
                seed : int = 123,
                id : int = 0, 
                workerID : str = 'A') -> None:
    
    # Affected by Grid Search
    BATCH_SIZE = int(grid_search_array[0])
    DROPOUT = float(grid_search_array[1])   # Was only used for initial grid search; will be decided upon during final testing.
    HIDDEN_LAYER_DIM = grid_search_array[2]
    LSTM_LAYERS = int(grid_search_array[3])  # Was only used for initial grid search, always 2 now to reduce complexity and prevent overfitting
    LEARNING_RATE = float(grid_search_array[4])   # Was only used for initial grid search; replaced by scheduler
    EMBEDDED_LAYER_DIM = int(grid_search_array[5])
    WEIGHT_DECAY = float(grid_search_array[6])
    SCHEDULER_FACTOR = float(grid_search_array[7])
    
    # Unaffected by grid search
    EPOCHS = epochs

    # EARLY STOPPING
    early_stop_counter = 0  # do not change
    early_stop_best_loss = torch.inf
    early_stop_best_model_state = None
    DELTA = 0.0  # if the loss decreases with maximum this delta, do not reset the counter
    
    # Reproducability
    torch.manual_seed(seed)
    
    r_name = f"RUN {id:04d}"
    with mlflow.start_run(run_name=r_name):
        print(f"Starting new run {id}...")
        
        # Load all the data into dataloaders, we do this here because batch_size is a hyperparameter that can be tweaked.
        TRAIN_DATALOADER = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        VALIDATION_DATALOADER = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        TEST_DATALOADER = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        # MLFlow's magic, log all parameters for this run to the local server.
        mlflow.log_param('batch_size', BATCH_SIZE)
        mlflow.log_param('dropout', DROPOUT)
        mlflow.log_param('hidden_layer_dim', HIDDEN_LAYER_DIM)
        mlflow.log_param('lstm_layers', LSTM_LAYERS)
        mlflow.log_param('learning_rate', LEARNING_RATE)
        mlflow.log_param('embedding_dim', EMBEDDED_LAYER_DIM)
        mlflow.log_param('weight_decay', WEIGHT_DECAY)
        mlflow.log_param('lr_scheduler_factor', SCHEDULER_FACTOR)

        # Build the model with the given hyperparameters, move it to device in use.
        model = RNN_model(
            dir=len_dir,
            dropout=DROPOUT,
            lstm_layers=LSTM_LAYERS,
            dim_embedded=EMBEDDED_LAYER_DIM,
            dim_hidden_layer=HIDDEN_LAYER_DIM,
            dim_out=classes_len).to(device)
        
        # https://arxiv.org/abs/2502.08441
        # Disable weight_decay on parameters: bias, embedding and normalization. 
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if "bias" in name:
                no_decay.append(param)
            elif "embed" in name.lower():
                no_decay.append(param)
            elif "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        # Using AdamW for decoupled weight decay from momentum and variance, recommended for LSTMs that overfit
        # easily, which happened in later stages of training. Weight decay is applied directly during parameter updates, but not to the loss.
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": WEIGHT_DECAY},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=LEARNING_RATE
        )
        
        # Added a scheduler after the initial grid search, because many models got 'stuck' and stopped early when testing them intensively later.
        # read more about this choice in the report.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=2, threshold=0.001)
            
        # --- TRAINING, VALIDATION ---
        print(f"Beginning training... using {device} device")
        for iEpoch in range(EPOCHS):
            t_loss, v_loss, (pltrain, tltrain), (plval, tlval)\
            = train_validate(train_loader=TRAIN_DATALOADER,
                            validation_loader=VALIDATION_DATALOADER,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=device)

            print(f"\nEPOCH {iEpoch}")
            print(f"----- TRAINING -----")
            t_macro_f1, t_weighted_f1, t_balanced_acc = calculate_metrics(False, t_loss, pltrain, tltrain)

            print(f"----- VALIDATION -----")
            v_macro_f1, v_weighted_f1, v_balanced_acc = calculate_metrics(False, v_loss, plval, tlval)

            mlflow.log_metric('train/loss', t_loss, step=iEpoch)
            mlflow.log_metric('train/macro_f1', t_macro_f1, step=iEpoch)
            mlflow.log_metric('train/weighted_f1', t_weighted_f1, step=iEpoch)
            mlflow.log_metric('train/balanced_acc', t_balanced_acc, step=iEpoch)
            mlflow.log_metric('val/loss', v_loss, step=iEpoch)
            mlflow.log_metric('val/macro_f1', v_macro_f1, step=iEpoch)
            mlflow.log_metric('val/weighted_f1', v_weighted_f1, step=iEpoch)
            mlflow.log_metric('val/balanced_acc', v_balanced_acc, step=iEpoch)

            # JSON EXPORT
            metrics_dict = {
                "avg_train_loss": t_loss,
                "avg_val_loss": v_loss,
                "pred_labels_train": pltrain,
                "true_labels_train": tltrain,
                "pred_labels_val": plval,
                "true_labels_val": tlval,
            }
            
            folder_path = f"/metrics/worker{workerID}/run_{id:04d}"
            os.makedirs(folder_path, exist_ok=True)
            
            with open(f"/metrics/worker{workerID}/run_{id:04d}/epoch_{iEpoch}.json", "w") as f:
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
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping...")

                    # Restore the best model which was saved earlier.
                    model.load_state_dict(early_stop_best_model_state)
                    break
        
        # Theoretically, this would've been helpful if the number of parameters trained was enormous, but well performing models
        # can easily be reproduced, so this is commented out.
        # torch.save(model.state_dict(), f"/best_models/RNN_best_model_run{id:04d}_{workerID}.pth")

        # --- TESTING ---
        avg_test_loss, (pred_labels_test, true_labels_test) = test(test_loader=TEST_DATALOADER,
                            model=model,
                            device=device)
        print(f"----- TESTING -----")
        t_macro_f1, t_weighted_f1, t_balanced_acc = calculate_metrics(True, avg_test_loss, pred_labels_test, true_labels_test)
        
        test_metrics_dict = {
            "avg_test_loss": avg_test_loss,
            "pred_labels_test": pred_labels_test,
            "true_labels_test": true_labels_test,
        }

        with open(f"/metrics/worker{workerID}/run_{id:04d}/test_metrics.json", "w") as f:
            json.dump(to_serializable(test_metrics_dict), f, indent=2)

        mlflow.log_metric('test/loss', avg_test_loss)
        mlflow.log_metric('test/macro_f1', t_macro_f1)
        mlflow.log_metric('test/weighted_f1', t_weighted_f1)
        mlflow.log_metric('test/balanced_acc', t_balanced_acc)
 
