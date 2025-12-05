from torch import nn
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, classification_report
import pandas as pd
from tabulate import tabulate

'''
@author Mathijs Tobé

Used by other RNN modules of the Deep Learning project
Defines the loss function for classification as only variable, rest is passed by other functions.
Contains functions for training, validating, testing and calculating the metrics.
'''

# Cross Entropy loss (simple for classification tasks)
# Since in later stages of training, the data was passed on balanced, no specific params need to be set for CEL.
loss_fn = nn.CrossEntropyLoss()

'''
@author Mathijs Tobé

Basic train and validate function.
'''
def train_validate(train_loader: torch.utils.data.DataLoader,
                   validation_loader: torch.utils.data.DataLoader,
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
    
    scheduler.step(avg_val_loss)

    # Format useful lists for calculation of metrics
    pred_labels_val = torch.cat(pred_labels_val, dim=0).cpu().detach().numpy().flatten()
    true_labels_val = torch.cat(true_labels_val, dim=0).cpu().detach().numpy().flatten()

    return avg_train_loss, avg_val_loss, \
        (pred_labels_train, true_labels_train), \
        (pred_labels_val, true_labels_val)

'''
@author Mathijs Tobé

Standard test function
'''
def test(test_loader: torch.utils.data.DataLoader,
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
@author Mathijs Tobé

To calculate different kind of metrics based on:
- Average (train/validation/test) loss, based on batch size, but averaged over size, so different batch sizes are in same range of losses.
- Predicted labels for training/validation/test
- True labels for training/validation/test

Calculates:
- Classification metrics
    - F1 scores (macro, weighted, per-class)
    - Confusion matrix (and prints)
    - Classification report
    - Balanced accuracy score
    
Calculated many things, but since LUMI is limited in what it logs efficiently, we decided to comment some things out that
were not used in later stages of the project anymore.

Also, this function is rewritten for the Transformer part of the project, which also requires regression.
'''
def calculate_metrics(
    showClassificationReport : bool, 
    avg_loss : np.floating, 
    predicted_labels : np.ndarray, 
    true_labels : np.ndarray):
    
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
    # per_class_f1 = f1_score(true_labels, predicted_labels, average=None)

    '''
    A full on confusion matrix of shape NxN.
    '''
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)

    '''
    Balanced accuracy = Each class contributes equally to the accuracy, better than the usual way of calculating accuracy: correct / total
    '''
    bal_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    print(
        f"--- CLASSIFICATION METRICS --- \n"
        f"F1 scores: [Macro={macro_f1:.3f}, Weighted={weighted_f1:.3f}] \n"
        f"Balanced Accuracy = {bal_accuracy:.3f}\n"
        f"Average loss = {avg_loss:.5f}")

    if showClassificationReport:  
        # The classification report contains the 'per class f1 scores', which is relevant if the data is very imbalanced. 
        # In general, it is a nice report to see, but usually only printed once after training during the test phase, to see performance
        # for different players in general.
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        df = pd.DataFrame(report).transpose()
        print(tabulate(df.round(3), headers='keys', tablefmt="pretty"))
        
    # returns these specific ones for logging to MLFlow
    return macro_f1, weighted_f1, bal_accuracy
