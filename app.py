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

TOP_K = 3
CONFIDENCE_THRESHOLD = 0.25

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
# Search
# ==========================================================

def search_documents(question):

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype(np.float32)

    # Normalize because chatbot.py uses cosine similarity
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


# ==========================================================
# Build Answer
# ==========================================================

def build_answer(results):

    if len(results) == 0:

        return """
❌ Sorry.

I couldn't find relevant information in the knowledge base.
"""

    best = results[0]

    if best["score"] < CONFIDENCE_THRESHOLD:

        return """
❌ Sorry.

I couldn't find a confident answer for your question.

Please try asking in a different way.
"""

    answer = f"""
## ✅ Answer

{best["text"]}

---

📄 **Source:** {best["source"]}

🔍 **Confidence:** {best["score"]:.2f}
"""

    return answer


# ==========================================================
# Chat Function
# ==========================================================

def chatbot(message, history):

    results = search_documents(message)

    answer = build_answer(results)

    return answer


# ==========================================================
# Interface
# ==========================================================

demo = gr.ChatInterface(

    fn=chatbot,

    title="⌚ WatchExpert AI Chatbot",

    description="""
Ask questions about

• Watch Brands

• Watch Models

• Warranty

• Returns

• Offers

• Shop Details

• FAQs
""",

    textbox=gr.Textbox(

        placeholder="Example: What brands do you sell?",

        container=True,

        scale=7
    ),

    chatbot=gr.Chatbot(
        height=550,
        show_copy_button=True
    ),

    theme=gr.themes.Soft(),

    examples=[

        "What brands do you sell?",

        "Do you have Seiko automatic watches?",

        "What is the warranty for Tissot?",

        "Do you provide EMI?",

        "Can I exchange my old watch?",

        "What are your shop timings?",

        "Do you sell Rolex?",

        "What payment methods are accepted?"
    ]
)

# ==========================================================
# Launch
# ==========================================================

demo.launch(
    share=True
)
