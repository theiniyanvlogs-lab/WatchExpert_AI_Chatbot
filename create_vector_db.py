import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

KNOWLEDGE_FOLDER = "knowledge"
VECTOR_DB_FOLDER = "vector_db"

INDEX_FILE = os.path.join(VECTOR_DB_FOLDER, "watch_index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DB_FOLDER, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Sentence Transformer Model...")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

# ==========================================================
# Read PDFs
# ==========================================================

def read_pdfs(folder):

    documents = []

    pdf_files = sorted(
        [
            f for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        ]
    )

    for filename in pdf_files:

        print(f"Reading: {filename}")

        pdf_path = os.path.join(folder, filename)

        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents.append(
            {
                "source": filename,
                "text": text
            }
        )

    return documents


# ==========================================================
# Better Text Splitter
# ==========================================================

def split_text(text,
               chunk_size=CHUNK_SIZE,
               overlap=CHUNK_OVERLAP):

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if len(chunk) > 50:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================================
# Read Knowledge Base
# ==========================================================

print("\nLoading PDFs...\n")

documents = read_pdfs(KNOWLEDGE_FOLDER)

all_chunks = []

for doc in documents:

    chunks = split_text(doc["text"])

    for chunk in chunks:

        all_chunks.append(
            {
                "source": doc["source"],
                "text": chunk
            }
        )

print("\n" + "=" * 60)
print("Knowledge Base Summary")
print("=" * 60)

print("PDF Files :", len(documents))
print("Chunks    :", len(all_chunks))

# ==========================================================
# Generate Embeddings
# ==========================================================

texts = [c["text"] for c in all_chunks]

print("\nGenerating Embeddings...\n")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(np.float32)

faiss.normalize_L2(embeddings)

# ==========================================================
# Build FAISS Index
# ==========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# ==========================================================
# Save Database
# ==========================================================

os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

faiss.write_index(index, INDEX_FILE)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)
print("WatchExpert AI Chatbot")
print("Vector Database Created Successfully")
print("=" * 60)

print("PDF Files Processed :", len(documents))
print("Text Chunks         :", len(all_chunks))
print("Embedding Model     :", MODEL_NAME)
print("FAISS Index         :", INDEX_FILE)
print("Chunks File         :", CHUNKS_FILE)

print("\nReady to launch app.py")
