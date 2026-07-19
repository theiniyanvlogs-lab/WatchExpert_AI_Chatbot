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

MIN_CHUNK_LENGTH = 50

# ==========================================================
# Load Sentence Transformer
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

        full_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:

                full_text += text
                full_text += "\n\n"

        documents.append(
            {
                "source": filename,
                "text": full_text
            }
        )

    return documents


# ==========================================================
# Better Text Splitter
# ==========================================================

def split_text(text):

    paragraphs = []

    text = text.replace("\r", "")

    blocks = text.split("\n\n")

    for block in blocks:

        block = block.strip()

        if len(block) < MIN_CHUNK_LENGTH:
            continue

        paragraphs.append(block)

    return paragraphs


# ==========================================================
# Read Knowledge Base
# ==========================================================

print("\nLoading PDFs...\n")

documents = read_pdfs(KNOWLEDGE_FOLDER)

all_chunks = []

seen = set()

for doc in documents:

    chunks = split_text(doc["text"])

    for chunk in chunks:

        key = chunk.strip()

        if key in seen:
            continue

        seen.add(key)

        all_chunks.append(
            {
                "source": doc["source"],
                "text": chunk
            }
        )

print("\n" + "=" * 60)
print("Knowledge Base Summary")
print("=" * 60)

print(f"PDF Files : {len(documents)}")
print(f"Chunks    : {len(all_chunks)}")

# ==========================================================
# Generate Embeddings
# ==========================================================

texts = [item["text"] for item in all_chunks]

print("\nGenerating Embeddings...\n")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(np.float32)

# Normalize embeddings for cosine similarity

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

print(f"PDF Files Processed : {len(documents)}")
print(f"Text Chunks         : {len(all_chunks)}")
print(f"Embedding Model     : {MODEL_NAME}")
print(f"FAISS Index         : {INDEX_FILE}")
print(f"Chunks File         : {CHUNKS_FILE}")

print("\nReady to launch app.py")
