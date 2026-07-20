import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

INDEX_FILE = "vector_db/watch_index.faiss"
CHUNKS_FILE = "vector_db/chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
CONFIDENCE_THRESHOLD = 0.30

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Sentence Transformer...")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

# ==========================================================
# Load Vector Database
# ==========================================================

print("Loading FAISS Index...")

index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# ==========================================================
# Search Documents
# ==========================================================

def search_documents(question):

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(embedding)

    scores, indices = index.search(embedding, TOP_K)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"]
        })

    return results


# ==========================================================
# Build Short Answer
# ==========================================================

def build_answer(question):

    results = search_documents(question)

    if len(results) == 0:
        return (
            "❌ Sorry.\n\n"
            "No relevant information found."
        )

    best = results[0]

    if best["score"] < CONFIDENCE_THRESHOLD:
        return (
            "❌ Sorry.\n\n"
            "I couldn't find a confident answer."
        )

    text = best["text"]

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))

    keywords = [
        word.lower()
        for word in re.findall(r"\w+", question)
        if len(word) > 2
    ]

    matched = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        score = 0

        lower = sentence.lower()

        for word in keywords:

            if word in lower:
                score += 1

        if score > 0:
            matched.append((score, sentence))

    matched.sort(reverse=True)

    answer = []

    for score, sentence in matched:

        if sentence not in answer:
            answer.append(sentence)

        if len(answer) == 3:
            break

    # Fallback
    if len(answer) == 0:

        answer = [
            text[:350] + "..."
        ]

    response = f"""
## ✅ Answer

{" ".join(answer)}

---------------------------------------

📄 Source : {best['source']}

⭐ Confidence : {best['score']:.2f}
"""

    return response


# ==========================================================
# Chat Function
# ==========================================================

def chatbot(message, history=None):

    if message is None:
        return "Please enter a question."

    message = message.strip()

    if message == "":
        return "Please enter a question."

    return build_answer(message)
