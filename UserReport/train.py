import torch
from transformers import DistilBartForConditionalGeneration, DistilBartTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Define your model and tokenizer
model_name = "facebook/bart-large-cnn"
model = DistilBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = DistilBartTokenizer.from_pretrained(model_name)

# Load your legal documents dataset using Hugging Face datasets
dataset = load_dataset("/train-data")

# Define a function to tokenize and preprocess your dataset
def preprocess_function(examples):
    return tokenizer(examples["document"], truncation=True, padding="max_length", max_length=512)

# Apply the preprocess function to your dataset
dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and validation sets
train_dataset = dataset["train-data"]
eval_dataset = dataset["validation"]

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    save_steps=10_000,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    report_to="tensorboard",
)

# Create a Trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model()

# Optionally, evaluate the model and generate summaries
results = trainer.evaluate()

# Generate summaries for legal documents
def generate_summary(document):
    input_ids = tokenizer.encode(document, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage:
document = "Your input legal document here..."
summary = generate_summary(document)
print("Generated Summary:", summary)
