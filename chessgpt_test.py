from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

model_name = "Waterhorse/chessgpt-base-v1"  # Your chess language model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define number of classes (e.g., 3 outcomes)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

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
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)