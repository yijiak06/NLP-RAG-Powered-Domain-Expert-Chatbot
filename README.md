# 📊 RAG-Powered Financial Announcement Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** for Saudi Stock Exchange (Tadawul) corporate announcements.

The system enables users to ask financial questions and receive:

* Context-grounded answers
* Source citations
* Multi-turn conversation support
* Fully local execution (no paid API required)

---

# 📌 Project Overview

The chatbot is built in five major stages:

1. Domain Selection
2. Document Processing
3. Vector Database Construction
4. RAG Implementation
5. Conversation Memory

---

# 1️⃣ Domain Selection (Requirement 1)

**Domain:** Corporate Announcements (news)

The dataset contains:

* Dividend announcements
* Earnings reports
* Corporate contracts
* Regulatory disclosures

**Source File:** `dataset.csv`
**Total Documents:** 1,800+

This domain requires accurate retrieval because financial information is structured, numerical, and time-sensitive.

---

# 2️⃣ Document Processing (Requirement 2)

## Data Cleaning

Before indexing:

* Removed rows with missing `DetailedSummary`
* Removed duplicate announcements

This ensures clean and reliable retrieval.

---

## Metadata Extraction

Each chunk stores metadata for citation display:

| Field       | Purpose                      |
| ----------- | ---------------------------- |
| `source_id` | Unique document identifier   |
| `title`     | Announcement subject         |
| `date`      | Announcement date            |
| `impact`    | Financial impact description |

This allows answers to include traceable sources.

---

## Smart Chunking Strategy

We implemented a sliding window approach:

* Chunk Size: 800 characters
* Overlap: 100 characters
* Discard chunks under 50 characters

Why?

Financial summaries often contain important values (company names, dividend amounts, dates). Overlap ensures information is not lost at chunk boundaries.

---

# 3️⃣ Vector Database (Requirement 3)

We use **ChromaDB (Persistent Mode)** to store embeddings.

* Storage Path: `./chroma_db`
* Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
* Similarity Metric: Cosine similarity

The database persists between runs and does not need to be rebuilt each time.

---

## Vector Pipeline

```
dataset.csv
   ↓
Load & Clean
   ↓
Metadata Extraction
   ↓
Chunking
   ↓
Embedding (MiniLM)
   ↓
ChromaDB (Persistent)
```

---

# 4️⃣ RAG Implementation (Requirement 4)

When a user asks a question:

1. The query is embedded.
2. Top-K relevant chunks are retrieved (K = 3).
3. Retrieved context is injected into the prompt.
4. A language model generates an answer.
5. Citations are displayed.

---

## Generation Model

* Model: `google/flan-t5-base`
* Runs locally
* No paid API required
* Max tokens: 256
* Device: CPU

The model is instructed to:

* Use only the provided context
* Avoid hallucination
* Provide citation references
* End with:

```
Sources: [1], [2], ...
```

---

# 5️⃣ Conversation Memory (Requirement 5)

The chatbot supports multi-turn interaction using a rolling memory buffer.

Memory stores:

* Last 5 user messages
* Last 5 assistant responses

Conversation history is injected into the prompt before each new question.

This allows follow-up queries such as:

* “What were their ex-dividend dates?”
* “Tell me more about the first one.”
* “What about that company?”

---

# 🛠️ How to Run the Project

## Step 1 – Install Dependencies

```bash
pip install pandas chromadb sentence-transformers tqdm transformers torch
pip install sentence-transformers
```

---

## Step 2 – Build the Vector Database

```bash
python build_vector_db.py
```

This creates:

* `./chroma_db/`
* `processed_chunks.json`

---

## Step 3 – Run the Chatbot

```bash
python rag_chat.py
```

Type `exit` to quit.

---

# 📁 Project Structure

```
├── dataset.csv
├── build_vector_db.py
├── rag_chat.py
├── chroma_db/
├── processed_chunks.json
└── README.md
```
---

# 🎯 System Capabilities

* Persistent vector database
* Semantic similarity search
* Citation-backed answers
* Multi-turn conversation support
* Fully local execution
* Reproducible pipeline


