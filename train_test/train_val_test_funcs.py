import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

'''
Main train and validation function. Independent of epoch. 

Returns the internal model state with parameters, training loss, and validation loss.
'''
def train_validate(train_loader : DataLoader, 
                   validation_loader : DataLoader, 
                   model : nn.Module, 
                   loss_fn : nn.Module, 
                   optimizer : nn.Module,
                   device : torch.device,
                   scheduler : lr_scheduler = None): 
    estimatedLabels_train = []
    trueLabels_train = []
    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # --- TRAINING ---
    model.train()

    for xbatch, ybatch in train_loader:
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)

        y_pred = model(xbatch)

        loss = loss_fn(y_pred, ybatch)
        batch_losses_train.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO classification
        # for now just an argmax
        yhat = torch.argmax(y_pred, dim=1)        
        estimatedLabels_train.append(yhat.cpu())
        trueLabels_train.append(ybatch.cpu())

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
        for xbatch, ybatch in validation_loader:
            xbatch, ybatch = xbatch.to(device), ybatch.to(device)
            y_pred = model(xbatch)

            estimatedLabels_val_raw_logits.append(y_pred.cpu())

            loss = loss_fn(y_pred, ybatch)

            batch_losses_val.append(loss.item())

            # TODO classification
            # for now just an argmax
            yhat = torch.argmax(y_pred, dim=1) 
            estimatedLabels_val.append(yhat.cpu())
            trueLabels_val.append(ybatch.cpu())
            
            print(y_pred[:5], ybatch[:5])

    # Validation loss will be the total validation loss over all batches divided by the number of batches.
    avg_val_loss = np.mean(batch_losses_val)  # we can do this because reduction =' mean'

    # After validation, do scheduler step
    if scheduler is not None:
        scheduler.step(avg_val_loss)
    
    # all_estimated = torch.cat(estimatedLabels_val, dim=0).numpy().flatten()
    # all_true = torch.cat(trueLabels_val, dim=0).numpy().flatten()
    
    return avg_train_loss, avg_val_loss

'''
Main test function. Independent of epoch. 

Returns the test loss.
'''
def test(test_loader : DataLoader, 
                   model : nn.Module, 
                   loss_fn : nn.Module, 
                   device : torch.device): 
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