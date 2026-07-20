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
# Search Function
# ==========================================================

def search_documents(question):

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(embedding)

    scores, indices = index.search(embedding, TOP_K)

    results = []

    seen = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]

        if chunk["text"] in seen:
            continue

        seen.add(chunk["text"])

        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "text": chunk["text"]
            }
        )

    return results

# ==========================================================
# Extract Short Answer
# ==========================================================

def extract_short_answer(text, question):

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))

    question_words = set(
        word.lower()
        for word in re.findall(r"\w+", question)
        if len(word) > 2
    )

    scored = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        words = set(
            word.lower()
            for word in re.findall(r"\w+", sentence)
        )

        score = len(question_words.intersection(words))

        scored.append((score, sentence))

    scored.sort(reverse=True)

    answer = []

    for score, sentence in scored[:3]:

        if sentence not in answer:
            answer.append(sentence)

    if answer:
        return "\n\n".join(answer)

    return text[:350]

# ==========================================================
# Build Answer
# ==========================================================

def build_answer(question):

    results = search_documents(question)

    if not results:

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

    answer = extract_short_answer(
        best["text"],
        question
    )

    response = f"""
## ✅ Answer

{answer}

----------------------------------------

📄 Source : {best['source']}

⭐ Confidence : {best['score']:.2f}
"""

    return response

# ==========================================================
# Chat Function
# ==========================================================

def chatbot(message, history=None):

    if message is None or not message.strip():

        return "Please enter your question."

    return build_answer(message)
