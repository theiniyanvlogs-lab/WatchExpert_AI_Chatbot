import pickle
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
INDEX_FILE = "vector_db/watch_index.faiss"
CHUNKS_FILE = "vector_db/chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading Sentence Transformer...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS Index...")
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# -----------------------------
# Search
# -----------------------------
def search(question, top_k=3):
    emb = model.encode([question], convert_to_numpy=True).astype("float32")

    distances, indices = index.search(emb, top_k)

    answer = ""

    for idx in indices[0]:
        if idx == -1:
            continue

        item = chunks[idx]

        answer += f"📄 Source: {item['source']}\n\n"
        answer += item["text"]
        answer += "\n\n"
        answer += "=" * 60
        answer += "\n\n"

    if answer.strip() == "":
        answer = "Sorry, I couldn't find relevant information."

    return answer


# -----------------------------
# Chat Function
# -----------------------------
def chatbot(message, history):
    return search(message)


demo = gr.ChatInterface(
    fn=chatbot,
    title="⌚ WatchExpert AI Chatbot",
    description="Ask questions about watches, warranty, offers, shop details and FAQs.",
    textbox=gr.Textbox(
        placeholder="Example: What brands do you sell?"
    ),
)

demo.launch(share=True)
