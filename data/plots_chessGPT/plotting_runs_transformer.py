import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

def calculate_metrics_for_plots(avg_loss: np.ndarray, predicted_labels: np.ndarray, true_labels: np.ndarray):
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
    Balanced accuracy = Each class contributes equally to the accuracy, better than the usual way of calculating accuracy: correct / total
    '''
    bal_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    return macro_f1, weighted_f1, per_class_f1, bal_accuracy

def add_a_model_to_plt(plt,directory,color,n_epochs):
    bal_accuracy_array = []
    epochs = []
    for epoch in range(n_epochs):
        with open(f"../run_metrics_chessGPT/{directory}/metrics_epoch_{epoch}.json", "r") as f:
            data = json.load(f)

        avg_val_loss = data["avg_val_loss"]
        pred_labels_val = np.array(data["pred_labels_val"])
        true_labels_val = np.array(data["true_labels_val"])

        macro_f1, weighted_f1, per_class_f1, bal_accuracy = calculate_metrics_for_plots(avg_val_loss, pred_labels_val,
                                                                                        true_labels_val)
        macro_f1_array.append(macro_f1)
        weighted_f1_array.append(weighted_f1)
        for i, class_f1 in enumerate(per_class_f1):
            if epoch == 0:
                per_class_f1_array.append([])
            per_class_f1_array[i].append(per_class_f1)
        bal_accuracy_array.append(bal_accuracy)
        epochs.append(epoch)

    print(f"----------TEST--METRICS--------")
    if directory != 'Optimus_Prime':
        with open(f"../run_metrics_chessGPT/{directory}/test_metrics.json", "r") as f:
            data = json.load(f)

        avg_test_loss = data["avg_test_loss"]
        pred_labels_test = np.array(data["pred_labels_test"])
        true_labels_test = np.array(data["true_labels_test"])

        macro_f1, weighted_f1, per_class_f1, bal_accuracy = calculate_metrics_for_plots(avg_test_loss, pred_labels_test,
                                                                                        true_labels_test)
        bal_accuracy_array.append(bal_accuracy)
        epochs.append(n_epochs)

    plt.plot(epochs, bal_accuracy_array, color=color, label=directory)


directories = ['Megatron','Bumblebee','Optimus_Prime']
colors = ['red','orange','blue']
#epochs = [9,20,20]

directory = 'Bumblebee'
mean_elo = 2654.766845703125  # 20 players: 2663.914794921875 | 100 players: 2654.766845703125
std_elo = 108.93199920654297  # 20 players: 111.10905456542969 | 100 players: 108.93199920654297

macro_f1_array = []
weighted_f1_array = []
per_class_f1_array = []
bal_accuracy_array = []
epochs = []


"""
plt.figure(figsize=(6, 6))
plt.title("Accuracy per class over time")
for directory,n_epochs,color in zip(directories,epochs,colors):
    print(f"--- {color} ---{directory}---{n_epochs}")
    add_a_model_to_plt(plt,directory,color,n_epochs)
plt.legend()
plt.savefig(f"./comparison_plots/accuracy_model_over_time.png")
plt.close()

"""
for epoch in range(20):
    with open(f"../run_metrics_chessGPT/{directory}/metrics_epoch_{epoch}.json", "r") as f:
        data = json.load(f)

    avg_val_loss = data["avg_val_loss"]
    pred_labels_val = np.array(data["pred_labels_val"])
    true_labels_val = np.array(data["true_labels_val"])

    macro_f1, weighted_f1, per_class_f1, bal_accuracy = calculate_metrics_for_plots(avg_val_loss, pred_labels_val, true_labels_val)
    macro_f1_array.append(macro_f1)
    weighted_f1_array.append(weighted_f1)
    for i,class_f1 in enumerate(per_class_f1):
        if epoch == 0:
            per_class_f1_array.append([])
        per_class_f1_array[i].append(per_class_f1)
    bal_accuracy_array.append(bal_accuracy)
    epochs.append(epoch)

print(f"----------TEST--METRICS--------")

with open(f"../run_metrics_chessGPT/{directory}/test_metrics.json", "r") as f:
    data = json.load(f)

avg_test_loss = data["avg_test_loss"]
pred_labels_test = np.array(data["pred_labels_test"])
true_labels_test = np.array(data["true_labels_test"])

macro_f1, weighted_f1, per_class_f1, bal_accuracy = calculate_metrics_for_plots(avg_test_loss, pred_labels_test, true_labels_test)
macro_f1_array.append(macro_f1)
weighted_f1_array.append(weighted_f1)
for i,class_f1 in enumerate(per_class_f1):
    per_class_f1_array[i].append(per_class_f1)
bal_accuracy_array.append(bal_accuracy)
epochs.append(20)


colors = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    'blue', 'orange', 'green', 'red', 'purple',
    'brown', 'pink', 'gray', 'cyan','olive'
]


plt.figure(figsize=(6, 6))
plt.plot(epochs,macro_f1_array,color="blue", label="Macro F1")
plt.plot(epochs,weighted_f1_array,color="red", label="Weighted F1")
plt.plot(epochs,bal_accuracy_array,color="green", label="Balanced Accuracy")
plt.legend()

for i,class_f1_array in enumerate(per_class_f1_array):
    print(i)
    print(colors[i])
    plt.plot(epochs,class_f1_array,color=colors[i], label=f"Class {i} F1")
plt.title("Accuracy per class over time")
plt.savefig(f"./comparison_plots/accuracy_x_class_over_time_{directory}.png")
plt.close()
