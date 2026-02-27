📊 RAG-Powered Financial Announcement Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot for financial announcements from the Saudi Stock Exchange (Tadawul).
It includes:
* Data ingestion and cleaning
* Smart document chunking
* Vector database construction
* Retrieval-Augmented Generation
* Multi-turn conversation memory

1️⃣ Domain Selection (Req 1)
Domain: Saudi Stock Exchange (Tadawul) Corporate Announcements
We use structured financial disclosures including:
* Dividend announcements
* Earnings reports
* Corporate contracts
* Regulatory filings
Dataset: dataset.csv
Total Documents: 1,800+ (well above minimum requirement)
This domain requires precise retrieval because financial data is factual and time-sensitive.

2️⃣ Document Processing (Req 2)
A. Cleaning
Before indexing, we:
* Removed rows missing DetailedSummary
* Removed duplicate announcements
This ensures high-quality retrieval and avoids empty results.

B. Metadata Extraction
Each chunk stores metadata for citation display:
Field	Purpose
source_id	Unique document ID
title	Announcement subject
date	Announcement date
impact	Financial impact tag
This allows answers to include traceable sources.

C. Smart Chunking Strategy
We use a sliding window approach:
* Chunk Size: 800 characters
* Overlap: 100 characters
* Filter: Remove chunks under 50 characters
Why?
Financial announcements can be long. Overlapping chunks prevent important information (like company name + dividend amount) from being split across boundaries.

3️⃣ Vector Database (Req 3)
We use ChromaDB (Persistent Mode) to store embeddings.
* Storage path: ./chroma_db
* Embedding model: sentence-transformers/all-MiniLM-L6-v2
* Similarity metric: Cosine similarity

Why this model?
It provides a strong balance between:
* Speed
* Accuracy
* Lightweight CPU performance
The database persists between runs and does not need rebuilding.

Vector Pipeline
graph LR
    A[dataset.csv] --> B(Clean & Load)
    B --> C(Metadata Extraction)
    C --> D(Chunking)
    D --> E(Embedding Model)
    E --> F[(ChromaDB)]

4️⃣ RAG Implementation (Req 4)
When a user asks a question:
1. The query is embedded 
2. Top-K relevant chunks are retrieved (K = 3) 
3. Retrieved context is injected into the prompt 
4. A language model generates an answer 
5. Citations are displayed 

Generation Model
* Model: google/flan-t5-base 
* Runs locally (no paid API) 
* Max tokens: 256 
* Device: CPU 
The model is instructed to:
* Use only provided context 
* Avoid hallucination 
* Include citation references 
* End answers with: Sources: [1], [2], ...
*  

RAG Flow
graph LR
    A[User Question] --> B[Embed Query]
    B --> C[Chroma Search]
    C --> D[Top-K Chunks]
    D --> E[LLM Prompt]
    E --> F[Answer + Citations]

5️⃣ Conversation Memory (Req 5)
To support follow-up questions, we implemented memory using a rolling buffer.
* Stores last 5 user messages
* Stores last 5 assistant responses
* Injected into prompt before each new question
This allows the chatbot to understand:
* “the first one”
* “their dividend dates”
* “that company”
Memory improves multi-turn reasoning without rebuilding the database.

Memory Flow
graph LR
    A[New Question] --> B[Retrieve Memory]
    B --> C[Retrieve Top-K Chunks]
    C --> D[Prompt = Memory + Context + Question]
    D --> E[LLM Response]
    E --> F[Update Memory]

🛠️ How to Run the Project
Step 1 – Install Dependencies
pip install pandas chromadb sentence-transformers tqdm transformers torch

Step 2 – Build the Vector Database
python build_vector_db.py
This creates:
* ./chroma_db/
* processed_chunks.json

Step 3 – Run the Chatbot
python rag_chat.py
Type exit to quit.

📁 Project Structure
├── dataset.csv
├── build_vector_db.py
├── rag_chat.py
├── memory.py
├── chroma_db/
├── processed_chunks.json
└── README.md

✅ Requirements Summary
Requirement	Status
Req 1 – Domain Selection	✔ Completed
Req 2 – Document Processing	✔ Completed
Req 3 – Vector Database	✔ Completed
Req 4 – RAG Implementation	✔ Completed
Req 5 – Conversation Memory	✔ Completed

🎯 Final System Capabilities
* Persistent vector database
* Semantic search
* Citation-backed answers
* Multi-turn conversation
* Fully local (no paid API)
* Reproducible pipeline
