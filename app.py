import pickle
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# =====================================================
# Configuration
# =====================================================

INDEX_FILE = "vector_db/watch_index.faiss"
CHUNKS_FILE = "vector_db/chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3
CONFIDENCE_THRESHOLD = 0.25

# =====================================================
# Load Model
# =====================================================

print("=" * 60)
print("Loading Sentence Transformer...")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

# =====================================================
# Load Vector Database
# =====================================================

print("Loading FAISS Index...")

index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# =====================================================
# Search Function
# =====================================================

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

        item = chunks[idx]

        if item["text"] in seen:
            continue

        seen.add(item["text"])

        results.append({
            "score": float(score),
            "source": item["source"],
            "text": item["text"]
        })

    return results


# =====================================================
# Build Answer
# =====================================================

def build_answer(results):

    if len(results) == 0:
        return (
            "❌ Sorry.\n\n"
            "I couldn't find relevant information "
            "in the knowledge base."
        )

    best = results[0]

    if best["score"] < CONFIDENCE_THRESHOLD:

        return (
            "❌ Sorry.\n\n"
            "I couldn't find a confident answer.\n"
            "Please ask your question differently."
        )

    answer = f"""
## ✅ Answer

{best['text']}

---------------------------------------

📄 Source:
{best['source']}

Confidence Score:
{best['score']:.2f}
"""

    return answer


# =====================================================
# Chat Function
# =====================================================

def chatbot(message, history):

    results = search_documents(message)

    answer = build_answer(results)

    return answer


# =====================================================
# Example Questions
# =====================================================

examples = [

    "What brands do you sell?",

    "Do you sell Rolex?",

    "Do you have Seiko automatic watches?",

    "What is the warranty for Tissot?",

    "Do you provide EMI?",

    "Can I exchange my old watch?",

    "What are your shop timings?",

    "What payment methods are accepted?"
]

# =====================================================
# Interface
# =====================================================

demo = gr.ChatInterface(

    fn=chatbot,

    title="⌚ WatchExpert AI Chatbot",

    description="""
Welcome to WatchExpert AI.

Ask questions about:

• Watch Brands

• Watch Models

• Warranty

• Shop Details

• Offers

• Returns

• FAQs
""",

    examples=examples,

    textbox=gr.Textbox(
        placeholder="Type your question here..."
    )
)

# =====================================================
# Launch
# =====================================================

demo.launch(
    share=True
)
