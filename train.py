import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os


# Load the CSV dataset
data_path = "data/codesearchnet_python_train.csv"

# Add quoting=3 to ignore quotes
df = pd.read_csv(data_path)

# Use a subset for faster prototyping (e.g., 10k samples)
df = df.sample(n=10000, random_state=42)

# Prepare the input-output pairs (code -> comment)
# Format: "Code: <code> Comment: <comment>"
df["text"] = "Code: " + df["code"] + " Comment: " + df["comment"]


# # Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text"]])

# # Load tokenizer and model (e.g., CodeBERT or DistilGPT-2)
model_name = "microsoft/codebert-base"  # Or "distilgpt2" for a lighter model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Add padding token if needed (CodeBERT doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset and prepare labels
def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    # Shift input_ids to create labels (for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Split into train and eval sets
train_dataset = tokenized_dataset.shuffle(seed=42).select(range(9000))  # 90%
eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(9000, 10000))  # 10%

# # Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# # Define training arguments
training_args = TrainingArguments(
    output_dir="models/",             
    overwrite_output_dir=True,
    num_train_epochs=3,                # Number of passes over the dataset
    per_device_train_batch_size=4,     # Adjust based on your GPU memory
    per_device_eval_batch_size=4,
    eval_strategy="epoch",             # Evaluate after each epoch
    save_strategy="epoch",             # Save after each epoch
    logging_dir="logs/",               # Where to save logs
    logging_steps=500,
    learning_rate=5e-5,                # Typical learning rate for fine-tuning
    load_best_model_at_end=True,       # Load the best model based on eval loss
)

# # Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# # Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained("models/final_model")
tokenizer.save_pretrained("models/final_model")

print("Training complete! Model saved to models/final_model")