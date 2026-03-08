# 📊 RAG-Powered Financial Announcement Chatbot

This project implements an advanced **Retrieval-Augmented Generation (RAG)** chatbot for Saudi Stock Exchange (Tadawul) corporate announcements. 

The system enables users to ask financial questions and receive:
* Context-grounded answers
* Source citations
* Multi-turn conversation support
* Highly accurate retrieval via Cross-Encoder Re-ranking
* An interactive Web UI deployed via Streamlit
* Fully local execution

---

## 📌 Project Overview

The chatbot is built in six major stages:
1. Domain Selection
2. Document Processing
3. Vector Database Construction
4. **Advanced RAG Implementation (Re-ranking)**
5. Conversation Memory
6. **Deployment & Conversational UI**

### 1️⃣ Domain Selection (Requirement 1)
* **Domain:** Corporate Announcements (news)
* **The dataset contains:** Dividend announcements, Earnings reports, Corporate contracts, Regulatory disclosures.
* **Source File:** `dataset.csv` | **Total Documents:** 1,800+
* This domain requires accurate retrieval because financial information is structured, numerical, and time-sensitive.

### 2️⃣ Document Processing (Requirement 2)

**Data Cleaning**
Before indexing:
* Removed rows with missing `DetailedSummary`
* Removed duplicate announcements
* *This ensures clean and reliable retrieval.*

**Metadata Extraction**
Each chunk stores metadata for citation display:
| Field | Purpose |
| :--- | :--- |
| `source_id` | Unique document identifier |
| `title` | Announcement subject |
| `date` | Announcement date |
| `impact` | Financial impact description |

*This allows answers to include traceable sources.*

**Smart Chunking Strategy**
We implemented a sliding window approach:
* **Chunk Size:** 800 characters
* **Overlap:** 100 characters
* **Discard:** Chunks under 50 characters

*Why?* Financial summaries often contain important values (company names, dividend amounts, dates). Overlap ensures information is not lost at chunk boundaries.

### 3️⃣ Vector Database (Requirement 3)
We use **ChromaDB (Persistent Mode)** to store embeddings.
* **Storage Path:** `./chroma_db`
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Similarity Metric:** Cosine similarity
* The database persists between runs and does not need to be rebuilt each time.

**Vector Pipeline**
```text
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

### 4️⃣ Advanced RAG Implementation (Advanced Feature)
To improve retrieval accuracy beyond standard vector similarity, we implemented a **Two-Stage Hybrid Pipeline with Re-ranking**:

1. **Initial Retrieval (Top-20):** The user query is embedded, and the top 20 relevant chunks are quickly retrieved from ChromaDB.
2. **Re-ranking (Top-5):** A Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) evaluates the exact context pairs and scores their semantic relevance, keeping only the most accurate Top 5 chunks.
3. **Generation:** The refined context is injected into the prompt. A language model (`google/flan-t5-base`) generates an answer without hallucination.
4. **Citations** are displayed alongside the generated answer.

### 5️⃣ Conversation Memory (Requirement 5)
The chatbot supports multi-turn interaction using a rolling memory buffer.
* **Memory stores:** Last 5 user messages & Last 5 assistant responses.
* **Context-Aware Search Logic:** The system intelligently prepends the previous user topic to the current query before hitting the database, drastically improving multi-turn retrieval accuracy.

This allows follow-up queries such as:
> “What were their ex-dividend dates?”
> “When was it announced?”

### 6️⃣ Deployment & Web UI (Deployment Feature)
The system has been upgraded from a CLI tool to a modern web application using **Streamlit**.
* **Conversational UI:** Features a chat-like interface with expanding source citations.
* **Input Validation:** Automatically detects and rejects non-alphanumeric inputs to prevent model hallucination.
* **Public URL:** The app is configured to be deployed on Streamlit Community Cloud for public access.

---

## 🛠️ How to Run the Project

**Step 1 – Install Dependencies**
```bash
pip install pandas chromadb sentence-transformers tqdm transformers torch streamlit

```

**Step 2 – Build the Vector Database**

```bash
python build_vector_db.py

```

*This creates the `./chroma_db/` directory and `processed_chunks.json`.*

**Step 3 – Run the Chatbot**

*Option A: Run the Web UI (Recommended)*

```bash
streamlit run app.py

```

*Option B: Run the Terminal CLI*

```bash
python rag_chat.py

```

*(Type `exit` to quit)*

---

## 📁 Project Structure

```text
├── dataset.csv
├── build_vector_db.py
├── rag_chat.py
├── memory.py
├── app.py
├── chroma_db/
├── processed_chunks.json
└── README.md

```

---

## 🎯 System Capabilities

* Persistent vector database
* Two-Stage Semantic Search (Vector DB + Cross-Encoder Re-ranking)
* Citation-backed answers with expandable UI elements
* Smart multi-turn conversation memory
* Fully local execution (No paid APIs required)
* Publicly deployable web interface

```
