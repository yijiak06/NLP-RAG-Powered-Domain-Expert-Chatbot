# -*- coding: utf-8 -*-

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import json
import os


# Configuration

DATA_FILE = "dataset.csv"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"
MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 800
OVERLAP = 100



# Load Dataset

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Basic cleaning
    df = df.dropna(subset=["DetailedSummary"])
    df = df.drop_duplicates(subset=["DetailedSummary"])

    print(f"Data loaded: {len(df)} rows.")
    return df



# Chunking Function

def get_text_chunks(text, size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]

        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        start = end - overlap

    return chunks



# Process Dataset

def process_dataset(df):
    print("Processing documents and extracting metadata...")
    processed_docs = []

    for i, row in tqdm(df.iterrows(), total=len(df)):

        content = (
            f"Title: {row['Subject']}\n"
            f"Date: {row['Date']}\n"
            f"Content: {row['DetailedSummary']}\n"
            f"Impact: {row['Impact']}"
        )

        chunks = get_text_chunks(content, CHUNK_SIZE, OVERLAP)

        metadata = {
            "source_id": f"doc_{i}",
            "title": str(row["Subject"]),
            "date": str(row["Date"]),
            "impact": str(row["Impact"]),
        }

        for j, chunk_text in enumerate(chunks):
            processed_docs.append({
                "id": f"doc_{i}_chunk_{j}",
                "text": chunk_text,
                "metadata": metadata
            })

    # Backup chunks locally
    with open("processed_chunks.json", "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(processed_docs)} chunks.")
    return processed_docs



# Build Chroma DB

def build_db(chunks):
    print(f"Initializing ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    # Reset collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 500
    print("Indexing chunks...")

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch]
        )

    print(f"Database built successfully. Total chunks: {collection.count()}")
    return collection



# Sanity Check

def check_db(collection):
    print("\nRunning test query...")
    results = collection.query(query_texts=["dividend"], n_results=1)

    if results["documents"] and results["documents"][0]:
        print("Match found:")
        print(f"Source: {results['metadatas'][0][0]['title']}")
        print(f"Preview: {results['documents'][0][0][:150]}...")
    else:
        print("No results found.")



# Main Execution

if __name__ == "__main__":
    df = load_data(DATA_FILE)

    if df is not None:
        chunks = process_dataset(df)
        collection = build_db(chunks)
        check_db(collection)