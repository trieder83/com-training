import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

# Initialize the model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Load communication data
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# Convert data to InputExample objects
# Convert data to InputExample objects
def preprocess_data(data):
    examples = []
    for record in data:
        examples.append(InputExample(
            texts=[record["sentence1"], record["sentence2"]],
            label=record["label"]
        ))
    return examples

# Load and preprocess the data
data_file = "communication_data.json"
communication_data = load_data(data_file)
train_examples = preprocess_data(communication_data)

# Sample data: List of InputExample objects
#train_examples = [
#    InputExample(texts=["The cat sat on the mat.", "A cat is sitting on a mat."], label=0.9),
#    InputExample(texts=["The sun is shining brightly.", "It's raining outside."], label=0.1),
#    InputExample(texts=["He loves playing football.", "Soccer is his favorite sport."], label=0.8),
#    InputExample(texts=["The car is fast.", "The vehicle moves quickly."], label=0.85),
#    InputExample(texts=["A beautiful day at the beach.", "I enjoyed a sunny beach day."], label=0.95),
#]

# Create a DataLoader for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune the model
num_epochs = 4
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="fine_tuned_all-MiniLM-L6-v2"
)

print("Model fine-tuned and saved to 'fine_tuned_all-MiniLM-L6-v2'")

