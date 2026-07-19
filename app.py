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

# -----------------------------
# Load Model
# -----------------------------
print("Loading Sentence Transformer...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Load FAISS Index
# -----------------------------
print("Loading FAISS Index...")
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# -----------------------------
# Search Function
# -----------------------------
def search_documents(question, top_k=3):

    embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(embedding, top_k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


# -----------------------------
# Chat Function
# -----------------------------
def respond(message, history):

    history = history or []

    results = search_documents(message)

    if not results:

        answer = "❌ Sorry, I couldn't find any relevant information."

    else:

        answer = ""

        for i, item in enumerate(results, start=1):

            answer += f"📄 Source : {item['source']}\n\n"
            answer += item["text"]
            answer += "\n\n"
            answer += "-" * 60
            answer += "\n\n"

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return history, history


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(
    title="WatchExpert AI Chatbot"
) as demo:

    gr.Markdown(
        """
# ⌚ WatchExpert AI Chatbot

### Intelligent PDF-Based Watch Store Assistant

Ask anything about:

- Rolex
- Titan
- Casio
- Fossil
- Seiko
- Citizen
- Tissot
- Warranty
- Offers
- Shop Details
- FAQs
"""
    )

    chatbot = gr.Chatbot(
        type="messages",
        height=550,
        label="WatchExpert AI"
    )

    state = gr.State([])

    msg = gr.Textbox(
        label="Your Question",
        placeholder="Example: What brands do you sell?"
    )

    with gr.Row():

        send = gr.Button("Send", variant="primary")

        clear = gr.Button("Clear Chat")

    send.click(
        respond,
        inputs=[msg, state],
        outputs=[chatbot, state]
    )

    msg.submit(
        respond,
        inputs=[msg, state],
        outputs=[chatbot, state]
    )

    clear.click(
        lambda: ([], []),
        outputs=[chatbot, state]
    )

demo.launch(share=True)
