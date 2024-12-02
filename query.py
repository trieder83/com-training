from sentence_transformers import SentenceTransformer

# Load the fine-tuned model
model_path = "fine_tuned_all-MiniLM-L6-v2"  # Replace with your model path
model = SentenceTransformer(model_path)

# Function to compute similarity between two sentences
def compute_similarity(sentence1, sentence2):
    # Generate embeddings for the sentences
    embeddings = model.encode([sentence1, sentence2])
    # Compute cosine similarity
    similarity = (embeddings[0] @ embeddings[1]) / (
        (embeddings[0] ** 2).sum() ** 0.5 * (embeddings[1] ** 2).sum() ** 0.5
    )
    return similarity

# Function to get embeddings for a single sentence
def get_embeddings(sentence):
    return model.encode(sentence)

# Example usage
if __name__ == "__main__":
    # Query sentence pairs
    sentence1 = "type: sms timediff: 5 text: The bank is located near the park."
    sentence2 = "type: sms timediff: 5 text: We need to meet near the river bank to finalize the plan."

    # Compute similarity
    similarity = compute_similarity(sentence1, sentence2)
    print(f"Similarity between the sentences: {similarity:.4f}")

    # Query a single sentence for embeddings
    sentence = "The sun is shining brightly today."
    embeddings = get_embeddings(sentence)
    #print(f"Embeddings for the sentence:\n{embeddings}")
