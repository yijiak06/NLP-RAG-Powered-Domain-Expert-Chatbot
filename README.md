# NLP-RAG-Powered-Domain-Expert-Chatbot

---

# RAG-Powered Financial Announcement Chatbot

## Requirements 1–3

---

## 1. Domain Selection

**Domain:** Saudi Stock Exchange Corporate Announcements

This project focuses on publicly listed company announcements and regulatory disclosures.

**Dataset**

* File: `dataset.csv`
* Total documents: 1839
* Minimum required: 50 (Satisfied)

Each row represents one standalone corporate announcement.

---

## 2. Document Processing

### Text Extraction

```
import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.dropna(subset=["DetailedSummary"])
df = df.drop_duplicates(subset=["DetailedSummary"])
```

---

### Metadata Extraction

Each document is constructed as:

```
content = f"""
Title: {row['Subject']}
Date: {row['Date']}
Content: {row['DetailedSummary']}
Impact: {row['Impact']}
"""
```

Metadata stored:

```
meta = {
    "source_id": f"doc_{i}",
    "title": row["Subject"],
    "date": row["Date"],
    "impact": row["Impact"]
}
```

---

### Chunking Strategy

Fixed-size sliding window

* Chunk size: 800
* Overlap: 100

```
def get_text_chunks(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks
```

---

## 3. Vector Database

Vector Database: ChromaDB (persistent mode)
Embedding Model: all-MiniLM-L6-v2

```
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```

Chunks are indexed in batches for efficiency.

---

## Architecture Overview

```
dataset.csv
    ↓
Load & Clean
    ↓
Metadata Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
ChromaDB (Persistent)
```

---
