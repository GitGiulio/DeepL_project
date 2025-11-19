from transformers import AutoTokenizer, AutoModelForSequenceClassification
#rom datasets import load_dataset
from transformers import TrainingArguments, Trainer


import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

LIST_OF_PLAYERS = ["Carlsen, Magnus",
                   "Cramling Bellon, Anna",
                   "Caruana, Fabiano",
                   "Nepomniachtchi, Ian",
                   "Firouzja, Alireza",
                   "Giri, Anish",
                   "Niemann, Hans",
                   "Cramling, Pia",
                   "Nakamura, Hikaru",
                   "Botez, Alexandra",
                   "Botez, Andrea",
                   "Belenkaya, Dina",
                   "So, Wesley",]

# ----------------------------
# CONFIGURATION PARAMETERS
# ----------------------------
csv_path = "your_file.csv"       # Path to your CSV
target_column1 = "white_name"          # Column name for labels
target_column2 = "black_name"          # Column name for labels
batch_size = 64                  # Batch size for DataLoader
train_ratio = 0.9                # Train split ratio
val_ratio = 0.05                 # Validation split ratio
test_ratio = 0.05                # Test split ratio
shuffle = True                   # Shuffle data before splitting
num_epochs = 10
# ----------------------------
# STEP 1: LOAD DATA
# ----------------------------
df = pd.read_csv(csv_path)

# Separate features and labels
df = df[(df['white_name'].isin(list_of_players)) | (df['black_name'].isin(list_of_players))]
X = df.drop(columns=[target_column1,target_column2]).values
y = df[target_column1].values

# ----------------------------
# STEP 2: SPLIT DATA
# ----------------------------
# First split into train and temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), shuffle=shuffle)

# Split temp into validation and test
val_size = val_ratio / (val_ratio + test_ratio)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_size), shuffle=shuffle)

# ----------------------------
# STEP 3: CREATE CUSTOM DATASET
# ----------------------------
class CSVDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Change dtype if regression
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# ----------------------------
# STEP 4: CREATE DATALOADERS
# ----------------------------
train_dataset = CSVDataset(X_train, y_train)
val_dataset = CSVDataset(X_val, y_val)
test_dataset = CSVDataset(X_test, y_test)

#---------------------------HERE STARTS THE DIFFERENT CODE---------------------------#

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model_name = "Waterhorse/chessgpt-base-v1"  # Your chess language model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=13)


for param in model.base_model.parameters(): # TODO here I am freezeing all, but some have to be unfreezed
    param.requires_grad = False

# Example: unfreeze last 2 transformer layers
#for layer in model.base_model.encoder.layer[-2:]:
#    for param in layer.parameters():
#        param.requires_grad = True


# model.classifier = MultiTaskHead(model.config.hidden_size) TODO this is how we add a custom classifier

#dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

def tokenize(batch):
    return tokenizer(batch["pgn"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

#------------------- AND HERE WE BACK TO THE FULL CUSTOM CODE-------------#
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()


