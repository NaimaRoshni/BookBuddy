import time
import numpy as np
from main import query_book_recommendation
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Define test cases
# -------------------------------
test_cases = [
    {"query": "Can you recommend a mystery book?", "expected_keywords": ["The Girl with the Dragon Tattoo", "Gone Girl", "The Da Vinci Code"]},
    {"query": "Tell me about psychological thrillers.", "expected_keywords": ["Gone Girl", "The Silent Patient", "Sharp Objects"]},
    {"query": "Books similar to The Girl with the Dragon Tattoo.", "expected_keywords": ["The Girl with the Dragon Tattoo", "The Silent Patient", "Gone Girl"]},
    {"query": "Suggest me a book by Agatha Christie.", "expected_keywords": ["The Murder of Roger Ackroyd", "The ABC Murders"]},
    {"query": "I am interested in fantasy books like The Name of the Wind.", "expected_keywords": ["The Name of the Wind", "Mistborn", "The Priory of the Orange Tree"]},
    {"query": "Books related to magic and fantasy.", "expected_keywords": ["The Name of the Wind", "The Priory of the Orange Tree", "The Hobbit"]},
    {"query": "What books are about psychological horror?", "expected_keywords": ["The Shining", "Pet Sematary", "The Exorcist"]},
    {"query": "Suggest me something by Stephen King.", "expected_keywords": ["The Shining", "Carrie", "It"]},
    {"query": "Can you recommend a classic detective mystery?", "expected_keywords": ["The Murder of Roger Ackroyd", "The Hound of the Baskervilles"]},
    {"query": "Books like Harry Potter?", "expected_keywords": ["The Name of the Wind", "The Priory of the Orange Tree", "The Hobbit"]},
    {"query": "Can you suggest a historical mystery?", "expected_keywords": ["The Secret History", "The Name of the Rose"]},
    {"query": "What is a good book for learning Python?", "expected_keywords": []},  # Irrelevant query: expected label 0
]

# -------------------------------
# Set a similarity threshold for a positive match
# (Adjust this threshold based on your experiments.)
threshold = 0.5

# -------------------------------
# Initialize the embedder for similarity scoring
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Lists to store ground truth labels, predicted labels, similarity scores, and response times
y_true = []
y_pred = []
similarity_scores = []
response_times = []

print("Starting evaluation of the book recommendation chatbot...\n")
# -------------------------------
# Process each test case
# -------------------------------
for test in test_cases:
    query = test["query"]
    expected_keywords = test["expected_keywords"]
    
    # Ground truth: label 1 if there are expected keywords, else 0
    true_label = 1 if expected_keywords else 0
    y_true.append(true_label)
    
    print(f"Query: {query}")
    
    # Get the chatbot response (using a default model, e.g., "llama3.2:latest")
    start_time = time.time()
    response = query_book_recommendation(query, "llama3.2:latest")
    response_time = time.time() - start_time
    response_times.append(response_time)
    
    print(f"Response: {response}")
    print(f"Response Time: {response_time:.2f} seconds")
    
    # Construct a gold reference text by joining expected keywords (if any)
    gold_reference = " ".join(expected_keywords) if expected_keywords else ""
    
    # Compute embeddings for the response and the gold reference
    response_embedding = embedder.encode(response)
    if gold_reference:
        gold_embedding = embedder.encode(gold_reference)
        # Compute cosine similarity: dot product divided by norms
        cosine_sim = np.dot(response_embedding, gold_embedding) / (np.linalg.norm(response_embedding) * np.linalg.norm(gold_embedding))
    else:
        cosine_sim = 0.0  # For irrelevant queries, treat similarity as 0
    
    similarity_scores.append(cosine_sim)
    print(f"Cosine Similarity: {cosine_sim:.2f}")
    
    # Predicted label: positive (1) if similarity exceeds threshold, otherwise negative (0)
    pred_label = 1 if cosine_sim >= threshold else 0
    y_pred.append(pred_label)
    
    print("-" * 50)

# -------------------------------
# Compute evaluation metrics
# -------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
avg_similarity = np.mean(similarity_scores)
avg_response_time = np.mean(response_times)

print("\nFinal Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Average Cosine Similarity: {avg_similarity:.2f}")
print(f"Average Response Time: {avg_response_time:.2f} seconds")
