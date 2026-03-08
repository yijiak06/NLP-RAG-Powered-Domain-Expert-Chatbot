import logging
# Silence ChromaDB warnings to keep the output clean
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder
from memory import ConversationMemory
from chromadb.utils import embedding_functions


# 1. Configuration
DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"

# Re-ranking Configuration
INITIAL_RETRIEVE_K = 20  # Initial retrieval count
TOP_K = 5                # Final number of documents to keep after re-ranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GEN_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 1500

# Initialize Memory (Stores the last 5 turns of conversation)
memory = ConversationMemory(max_turns=5)

# 2. Core Functions

def load_collection():
    """Connect to the Chroma Vector Database."""
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    return client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=sentence_transformer_ef
    )
def retrieve_chunks(collection, question, reranker, initial_k=INITIAL_RETRIEVE_K, final_k=TOP_K):
    """Retrieve and re-rank document chunks based on the question."""
    
    # Stage 1: Initial retrieval from Vector DB
    results = collection.query(
        query_texts=[question],
        n_results=initial_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    # Stage 2: Cross-Encoder Re-ranking
    pairs = [[question, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(scores, documents, metadatas), key=lambda x: x[0], reverse=True)
    
    chunks = []
    for score, doc, meta in scored_docs[:final_k]:
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": score
        })
    
    return chunks

def format_context(chunks):
    """Format retrieved chunks into a string for the LLM prompt."""
    context_text = ""
    citations = []

    for idx, chunk in enumerate(chunks):
        context_text += "[{}] {}\n\n".format(idx + 1, chunk["text"])
        citations.append(
            "[{}] {} ({})".format(
                idx + 1,
                chunk["metadata"]["title"],
                chunk["metadata"]["date"]
            )
        )

    context_text = context_text[:MAX_CONTEXT_CHARS]
    return context_text, citations

def generate_answer(tokenizer, model, question, context_text):
    """Generate an answer using the FLAN-T5 model."""
    
    conversation_history = memory.get_context()

    prompt = """
You are a financial analyst assistant.

Use ONLY the provided context to answer the question.

Conversation History:
{}

Context:
{}

Question:
{}

Detailed Answer:
""".format(conversation_history, context_text, question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def print_sample_questions():
    """Print sample questions to guide the user on testing memory."""
    print("\n💡 You can ask questions like:")
    print("--- Single Turn ---")
    print("- Which companies reported financial results?")
    print("- Tell me about SAL's 2023 performance.")
    print("\n🧠 To test the Memory feature, try a multi-turn conversation:")
    print("  Turn 1: What dividends were announced by ZAIN KSA?")
    print("  Turn 2: When was it announced? (Notice the use of 'it')")
    print("  Turn 3: Did they sign any other contracts? (Notice the use of 'they')\n")

# 3. Main Chat Loop

def chat():
    print("Financial RAG Chatbot Ready (type 'exit' to quit)")
    print_sample_questions()

    print("⏳ Loading Database and Models (This may take a moment)...")
    collection = load_collection()
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    print("✅ System Ready!\n")

    while True:
        try:
            # 1. Get User Input
            question = input("You: ")
            
            # Exit condition
            if question.lower() in ["exit", "quit"]:
                print("Bye!")
                break
            
            # Input validation: Reject inputs that lack alphanumeric characters (e.g., "?", "!!!")
            if not any(char.isalnum() for char in question):
                print("Assistant: Please enter a valid question containing text.\n")
                continue

            # 2. Add current question to memory first
            memory.add_user(question)

            # 3. Smart Search Logic
            search_query = question
            user_msgs = [msg for role, msg in memory.buffer if role == "User"]
            
            if len(user_msgs) >= 2:
                last_topic = user_msgs[-2] 
                search_query = f"{last_topic} {question}"
                print(f"   (DEBUG: Searching with context -> '{search_query[:60]}...')")
            
            # 4. Retrieve Documents
            chunks = retrieve_chunks(collection, search_query, reranker)

            if not chunks:
                print("\nAssistant: This question does not appear related to financial announcements.\n")
                memory.buffer.pop() 
                continue

            # 5. Generate Answer
            context_text, citations = format_context(chunks)
            answer = generate_answer(tokenizer, model, question, context_text)

            # 6. Add assistant answer to memory
            memory.add_assistant(answer)

            print("\nAssistant:\n")
            print(answer)

            print("\nSources:")
            for cite in citations:
                print(cite)
            print("\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
