# ⌚ WatchExpert AI Chatbot

## Overview

WatchExpert AI Chatbot is a PDF-based intelligent watch store assistant that answers customer questions using a collection of watch-related PDF documents.

The chatbot uses Sentence Transformers for semantic embeddings and FAISS for fast document retrieval without relying on external LLM APIs.

---

## Features

- PDF-based knowledge retrieval
- Semantic search using FAISS
- Supports multiple watch brands
- Warranty and return information
- Shop details
- Offers and discounts
- Frequently Asked Questions
- Gradio web interface
- Google Colab compatible

---

## Technologies Used

- Python
- Sentence Transformers
- FAISS
- PyPDF
- Gradio
- Google Colab

---

## Project Structure

```
WatchExpert_AI_Chatbot/
│
├── knowledge/
├── vector_db/
├── chatbot.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Create Vector Database

```bash
python chatbot.py
```

---

## Run Chatbot

```bash
python app.py
```

---

## Sample Questions

- What brands do you sell?
- Do you have Seiko automatic watches?
- What is the warranty for Tissot?
- What are your shop timings?
- Do you provide EMI?
- Can I exchange my old watch?

---

## Author

**Sugumar Ranganathan**

GitHub:
https://github.com/theiniyanvlogs-lab
