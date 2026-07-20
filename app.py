import re
import pickle
import faiss
import numpy as np
import gradio as gr
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

print("Loading Vector Database...")

index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# ==========================================================
# Search
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
# Build Answer
# ==========================================================

def build_answer(results):

    if len(results) == 0:

        return (
            "❌ Sorry!\n\n"
            "I couldn't find relevant information."
        )

    best = results[0]

    if best["score"] < CONFIDENCE_THRESHOLD:

        return (
            "❌ Sorry!\n\n"
            "I couldn't find a confident answer."
        )

    text = best["text"].replace("\n", " ")

    sentences = re.split(r'(?<=[.!?])\s+', text)

    final_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        if sentence not in final_sentences:
            final_sentences.append(sentence)

        if len(final_sentences) == 5:
            break

    if len(final_sentences) == 0:
        answer = text[:500]
    else:
        answer = "\n\n".join(final_sentences)

    response = f"""
## ✅ Answer

{answer}

---

📄 **Source**

{best["source"]}

⭐ **Confidence**

{best["score"]:.2f}
"""

    return response


# ==========================================================
# Chat Function
# ==========================================================

def chatbot(message, history):

    if not message.strip():

        return "Please enter your question."

    results = search_documents(message)

    return build_answer(results)


# ==========================================================
# Gradio Interface
# ==========================================================

demo = gr.ChatInterface(
    fn=chatbot,

    title="⌚ WatchExpert AI Chatbot",

    description="""
Ask questions about:

• Watch Brands
• Rolex
• Omega
• Warranty
• Returns
• Exchange
• EMI
• Payment Methods
• Shop Timings
• Offers
""",

    examples=[
        "Do you sell Rolex?",
        "Which brands are available?",
        "What is the warranty?",
        "Can I exchange my old watch?",
        "Do you provide EMI?",
        "What are your shop timings?",
        "Which payment methods are accepted?",
        "Tell me about Seiko watches",
        "Do you have Casio?",
        "What is the return policy?"
    ],

    theme=gr.themes.Soft()
)

# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":

    demo.launch(debug=True, share=True)
