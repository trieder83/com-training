import json
import numpy as np
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

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return [
        InputExample(texts=[item["sentence1"], item["sentence2"]], label=float(item["label"]))
        for item in data
    ]

# Load and preprocess the data
#data_file = "communication_data.json"
data_file = "testdata/m_embeddings_t1.json"
#test_file = "testdata/m_embeddings_t1.json"
test_file= "testdata/m_embeddings_t1_validate.json"

train_examples = load_data(data_file)

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)  # Compute the magnitude
    if magnitude == 0:
        return vector  # Avoid division by zero for zero vectors
    return vector / magnitude

def normalize_vectors(vectors):
    return np.array([normalize_vector(vec) for vec in vectors])

def train_model(train_file):
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
    return model

# Validate the model
def validate_model(test_file, model):
    # Load test data
    test_examples = load_data(test_file)

    # Prepare test sentences and labels
    test_sentences1 = [example.texts[0] for example in test_examples]
    test_sentences2 = [example.texts[1] for example in test_examples]
    true_scores = [example.label for example in test_examples]

    # Generate embeddings for test sentences
    embeddings1 = model.encode(test_sentences1)
    embeddings2 = model.encode(test_sentences2)

    # debug
    #print(f"emb1 {embeddings1}")
    #print(f"emb2 {embeddings2}")

    # Compute cosine similarities
    cosine_similarities = [
        (emb1 @ emb2) / (emb1 @ emb1) ** 0.5 / (emb2 @ emb2) ** 0.5
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ]
    print(f"sim {cosine_similarities}")
    print(f"sim test {true_scores}")

    # Evaluate with mean squared error and Pearson correlation
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    # Evaluate with mean squared error and Pearson/Spearman correlation
    mse = mean_squared_error(true_scores, cosine_similarities)
    pearson_corr, _ = pearsonr(true_scores, cosine_similarities)
    spearman_corr, _ = spearmanr(true_scores, cosine_similarities)

    print(f"Validation Results:\nMSE (lower is better): {mse:.4f}, Pearson(1 to -1, 0 non linar coorelation ) : {pearson_corr:.4f}, Spearman (rank order, -1 to 1: {spearman_corr:.4f}")

trained_model = train_model(data_file)
validate_model(test_file, trained_model)

print("Model fine-tuned and saved to 'fine_tuned_all-MiniLM-L6-v2'")

