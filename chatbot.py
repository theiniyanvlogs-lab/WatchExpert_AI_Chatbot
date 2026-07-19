import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
KNOWLEDGE_FOLDER = "knowledge"
VECTOR_DB_FOLDER = "vector_db"

INDEX_FILE = os.path.join(VECTOR_DB_FOLDER, "watch_index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DB_FOLDER, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 100

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading Sentence Transformer Model...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Read PDFs
# -----------------------------
def read_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)

            print(f"Reading: {filename}")

            reader = PdfReader(pdf_path)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents

# -----------------------------
# Split Text into Chunks
# -----------------------------
def split_text(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# -----------------------------
# Build Dataset
# -----------------------------
documents = read_pdfs(KNOWLEDGE_FOLDER)

all_chunks = []

for doc in documents:

    chunks = split_text(doc["text"], CHUNK_SIZE, OVERLAP)

    for chunk in chunks:

        if len(chunk.strip()) > 30:

            all_chunks.append({
                "source": doc["filename"],
                "text": chunk
            })

print(f"\nTotal Chunks: {len(all_chunks)}")

# -----------------------------
# Create Embeddings
# -----------------------------
texts = [c["text"] for c in all_chunks]

print("Generating Embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -----------------------------
# Save Vector Database
# -----------------------------
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

faiss.write_index(index, INDEX_FILE)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("\n===================================")
print("WatchExpert AI Chatbot")
print("Vector Database Created Successfully!")
print("===================================")

print(f"PDF Files Processed : {len(documents)}")
print(f"Text Chunks         : {len(all_chunks)}")
print(f"FAISS Index Saved   : {INDEX_FILE}")
print(f"Chunks Saved        : {CHUNKS_FILE}")
