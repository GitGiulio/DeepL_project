import json

import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import nn
import numpy as np
from torch.amp import autocast,GradScaler

from transformers import get_scheduler
# '''
# Main train and validation function. Independent of epoch. 

# Returns the internal model state with parameters, training loss, and validation loss.
# '''
# def train_validate(train_loader : DataLoader, 
#                    validation_loader : DataLoader, 
#                    model : nn.Module, 
#                    loss_fn : nn.Module, 
#                    optimizer : nn.Module,
#                    device : torch.device,
#                    scheduler : lr_scheduler = None): 
#     estimatedLabels_train = []
#     trueLabels_train = []
#     batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

#     # --- TRAINING ---
#     model.train()

#     for xbatch, ybatch in train_loader:
#         xbatch, ybatch = xbatch.to(device), ybatch.to(device)

#         y_pred = model(xbatch)

#         loss = loss_fn(y_pred, ybatch)
#         batch_losses_train.append(loss.item())

#         model.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # TODO classification
#         # for now just an argmax
#         yhat = torch.argmax(y_pred, dim=1)        
#         estimatedLabels_train.append(yhat.cpu())
#         trueLabels_train.append(ybatch.cpu())

#     # all_estimated = torch.cat(estimatedLabels_train, dim=0).numpy().flatten()
#     # all_true = torch.cat(trueLabels_train, dim=0).numpy().flatten()
#     avg_train_loss = np.mean(batch_losses_train)

#     # --- VALIDATION ---
#     model.eval()

#     estimatedLabels_val = []
#     # We may need the values in the array below to calculate a PR curve
#     # To optimize F1-scores. But at this point unsure whether we will use F1 scores.
#     estimatedLabels_val_raw_logits = [] 
#     trueLabels_val = []
#     batch_losses_val = []

#     with torch.no_grad():
#         for xbatch, ybatch in validation_loader:
#             xbatch, ybatch = xbatch.to(device), ybatch.to(device)
#             y_pred = model(xbatch)

#             estimatedLabels_val_raw_logits.append(y_pred.cpu())

#             loss = loss_fn(y_pred, ybatch)

#             batch_losses_val.append(loss.item())

#             # TODO classification
#             # for now just an argmax
#             yhat = torch.argmax(y_pred, dim=1) 
#             estimatedLabels_val.append(yhat.cpu())
#             trueLabels_val.append(ybatch.cpu())
            
#             print(y_pred[:5], ybatch[:5])

#     # Validation loss will be the total validation loss over all batches divided by the number of batches.
#     avg_val_loss = np.mean(batch_losses_val)  # we can do this because reduction =' mean'

#     # After validation, do scheduler step
#     if scheduler is not None:
#         scheduler.step(avg_val_loss)
    
#     # all_estimated = torch.cat(estimatedLabels_val, dim=0).numpy().flatten()
#     # all_true = torch.cat(trueLabels_val, dim=0).numpy().flatten()
    
#     return avg_train_loss, avg_val_loss

# '''
# Main test function. Independent of epoch. 

# Returns the test loss.
# '''
# def test(test_loader : DataLoader, 
#                    model : nn.Module, 
#                    loss_fn : nn.Module, 
#                    device : torch.device): 
#     model.eval()

#     estimatedLabels = []
#     trueLabels = []
#     batch_losses_test = []

#     with torch.no_grad():
#         for xbatch, ybatch in test_loader:
#             xbatch, ybatch = xbatch.to(device), ybatch.to(device)
#             y_pred = model(xbatch)

#             loss = loss_fn(y_pred, ybatch)

#             batch_losses_test.append(loss.item())

#             # TODO classification
#             # for now just an argmax
#             yhat = torch.argmax(y_pred, dim=1) 
#             estimatedLabels.append(yhat.cpu())
#             trueLabels.append(ybatch.cpu())

#     # Validation loss will be the total validation loss over all batches divided by the number of batches.
#     avg_test_loss = np.mean(batch_losses_test)
    
#     return avg_test_loss


batch_size = 64

def train_validate(train_loader: DataLoader,
                   validation_loader: DataLoader,
                   model: nn.Module,
                   optimizer,
                   scheduler,
                   device: torch.device,
                   alpha=0.2):

    scaler = GradScaler()

    batch_losses_train = []  # each batch, the loss is stored and later averaged to get an average train loss per epoch

    # used for f1 score and accuracy metrics
    pred_labels_train = []  
    true_labels_train = [] 
    pred_regression_train = []
    true_regression_train = []

    model.train()
    c = 0
    for batch in train_loader:
        c += 1
        if c % 10 == 0:
            print(f"fatti {c * batch_size}")
            break

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        class_labels = batch["class_labels"].to(device)
        reg_labels = batch["regression_labels"].to(device)

        with autocast(device.type):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            class_logits = outputs["classification"]
            class_regr = outputs["regression"]

            ce_loss = torch.nn.functional.cross_entropy(class_logits, class_labels)
            huber_loss = torch.nn.functional.smooth_l1_loss(class_regr, reg_labels)
            loss = ce_loss + alpha * huber_loss  # TODO weight regression appropriately

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        print(loss.item())
        
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

            ce_loss = torch.nn.functional.cross_entropy(class_logits, class_labels)
            huber_loss = torch.nn.functional.smooth_l1_loss(class_regr, reg_labels)
            loss = ce_loss + alpha * huber_loss

        print(loss.item())
        batch_losses_val.append(loss.item())

        preds = outputs["classification"].argmax(dim=1)
        pred_labels_val.append(preds)
        true_labels_val.append(class_labels)
        pred_regression_val.append(class_regr)
        true_regression_val.append(reg_labels)
        
        break
    
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

        ce_loss = torch.nn.functional.cross_entropy(class_logits, class_labels)
        huber_loss = torch.nn.functional.smooth_l1_loss(class_regr, reg_labels)

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

from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

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
def calculateMetrics(avg_loss : np.ndarray, predicted_labels : np.ndarray, true_labels : np.ndarray, 
                     predicted_regression : np.ndarray, true_regression : np.ndarray,set:str,epoch:int):
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
        f"Balanced Accuracy = {bal_accuracy:.3f}"
        f"Average loss = {avg_loss:.5f}")
    
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(tabulate(df.round(3), headers='keys', tablefmt="pretty"))
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=False, cmap="Blues")
    plt.title("Confusion matrix")
    plt.savefig(f"../data/plots_chessGPT/{directory}/confusion_matrix_{set}_{epoch}.png")
    plt.close()
    # --- REGRESSION METRICS ---
    '''
    The basic error functions for regression model outputs
    '''
    mean_abs_error = mean_absolute_error(true_regression, predicted_regression)
    
    mean_squ_error = mean_squared_error(true_regression, predicted_regression)
    
    root_mean_squ_error = mean_squ_error ** 0.5
    
    '''
    R-squared score represents the proportion of the variance in the dependent variable that is predictable from the independent variable (source: https://www.bmc.com/blogs/mean-squared-error-r2-and-variance-in-regression-analysis/)
    Aka the total variance explained by the model / total variance. Low value = no correlation.
    '''
    r2 = r2_score(true_regression, predicted_regression)
    
    print(f"--- REGRESSION METRICS ---\n"
          f"MAE = {mean_abs_error:.3f}, MSE = {mean_squ_error:.3f}, RMSE = {root_mean_squ_error:.3f}\nR2 score = {r2:.3f}")
    
    plt.figure(figsize=(6,6))
    plt.scatter(true_regression, predicted_regression, alpha=0.5)
    plt.plot([true_regression.min(), true_regression.max()], [true_regression.min(), true_regression.max()], 'r--')  # y=x line
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Regression: Predicted vs True")
    plt.savefig(f"../data/plots_chessGPT/{directory}/regression_{set}_{epoch}.png")
    plt.close()

directory = 'metrics4'
mean_elo = 2663.78369140625
std_elo = 110.49032592773438

for i in range(5):
    with open(f"../data/run_metrics_chessGPT/{directory}/metrics_epoch_{i}.json", "r") as f:
        data = json.load(f)

    avg_train_loss = data["avg_train_loss"]
    avg_val_loss = data["avg_val_loss"]
    pred_labels_train = np.array(data["pred_labels_train"])
    true_labels_train = np.array(data["true_labels_train"])
    pred_labels_val = np.array(data["pred_labels_val"])
    true_labels_val = np.array(data["true_labels_val"])
    pred_regression_train = (np.array(data["pred_regression_train"]) * std_elo) + mean_elo
    true_regression_train = (np.array(data["true_regression_train"]) * std_elo) + mean_elo
    pred_regression_val = (np.array(data["pred_regression_val"]) * std_elo) + mean_elo
    true_regression_val = (np.array(data["true_regression_val"]) * std_elo) + mean_elo

    calculateMetrics(avg_train_loss,pred_labels_train,true_labels_train,pred_regression_train,true_regression_train,"train",i)
    print(f"EVAL epoch {i+1}")
    calculateMetrics(avg_val_loss,pred_labels_val,true_labels_val,pred_regression_val,true_regression_val,"val",i)

#print(f"----------TEST--METRICS--------")
#
#with open(f"../data/run_metrics_chessGPT/{directory}/test_metrics.json", "r") as f:
#    data = json.load(f)
#
#avg_test_loss = data["avg_test_loss"]
#pred_labels_test = np.array(data["pred_labels_test"])
#true_labels_test = np.array(data["true_labels_test"])
#pred_regression_test = (np.array(data["pred_regression_test"]) * std_elo) + mean_elo
#true_regression_test = (np.array(data["true_regression_test"]) * std_elo) + mean_elo
#
#calculateMetrics(avg_test_loss,pred_labels_test,true_labels_test,pred_regression_test,true_regression_test,"test",0)
