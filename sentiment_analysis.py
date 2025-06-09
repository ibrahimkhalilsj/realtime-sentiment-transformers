import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
print("Loading cardiffnlp/tweet_eval sentiment dataset...")
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
print(f"Loading tokenizer and model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = ds.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    run_name="sentiment_training_run",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
print("Training model...")
trainer.train()

# Evaluate the model
print("Evaluating model on test set...")
eval_results = trainer.evaluate(encoded_dataset["test"])
print("Test Set Evaluation:", eval_results)

# Define prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_map.get(predicted_class, f"label_{predicted_class}")
    return {"text": text, "sentiment": sentiment}

# Run predictions
print("Running sample predictions...")
sample_texts = [
    "I just got a promotion at work! So excited!",
    "Feeling really down today, nothing is going right.",
    "Wow, I didn't expect that plot twist in the movie!"
]
for text in sample_texts:
    result = predict_sentiment(text)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}\n")