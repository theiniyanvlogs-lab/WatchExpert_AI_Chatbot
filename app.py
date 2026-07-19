import pickle
import faiss
import gradio as gr
import numpy as np

from sentence_transformers import SentenceTransformer

# -----------------------------------
# Configuration
# -----------------------------------
INDEX_FILE = "vector_db/watch_index.faiss"
CHUNKS_FILE = "vector_db/chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------------
# Load Model
# -----------------------------------
print("Loading Sentence Transformer...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------------
# Load FAISS Index
# -----------------------------------
print("Loading FAISS Index...")
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Vector Database Loaded Successfully!")

# -----------------------------------
# Search Function
# -----------------------------------
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

# -----------------------------------
# Chat Function
# -----------------------------------
def chatbot(message, history):

    results = search_documents(message)

    if len(results) == 0:
        answer = "Sorry, I couldn't find any relevant information."
    else:

        answer = ""

        for i, item in enumerate(results, start=1):

            answer += (
                f"📄 Source: {item['source']}\n\n"
                f"{item['text']}\n\n"
                f"{'-'*60}\n\n"
            )

    history.append((message, answer))

    return history, history

# -----------------------------------
# Gradio Interface
# -----------------------------------
with gr.Blocks(title="WatchExpert AI Chatbot") as demo:

    gr.Markdown(
        """
        # ⌚ WatchExpert AI Chatbot

        Ask questions about:

        ✅ Watch Brands

        ✅ Models

        ✅ Warranty

        ✅ Offers

        ✅ Store Details

        ✅ Returns

        ✅ FAQs
        """
    )

    chatbot_ui = gr.Chatbot(height=500)

    msg = gr.Textbox(
        placeholder="Ask a question...",
        label="Your Question"
    )

    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(
        chatbot,
        inputs=[msg, state],
        outputs=[chatbot_ui, state]
    )

    clear.click(
        lambda: ([], []),
        outputs=[chatbot_ui, state]
    )

# -----------------------------------
# Launch
# -----------------------------------
demo.launch(share=True)
