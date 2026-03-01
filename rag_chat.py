import logging
# Silence ChromaDB warnings to keep the output clean
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from memory import ConversationMemory

# =========================
# 1. Configuration
# =========================
DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"
TOP_K = 3
GEN_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 1500
SIMILARITY_THRESHOLD = 0.5 

# Initialize Memory (Stores the last 5 turns of conversation)
memory = ConversationMemory(max_turns=5)

# =========================
# 2. Core Functions
# =========================

def load_collection():
    """Connect to the Chroma Vector Database."""
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    return client.get_collection(name=COLLECTION_NAME)

def retrieve_chunks(collection, question, k=TOP_K):
    """Retrieve relevant document chunks based on the question."""
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    # Check if results exist
    if results["documents"]:
        for i in range(len(results["documents"][0])):
            # In ChromaDB, simply Append the retrieved documents
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
    
    return chunks

def format_context(chunks):
    """Format retrieved chunks into a string for the LLM prompt."""
    context_text = ""
    citations = []

    for idx, chunk in enumerate(chunks):
        # Append document text
        context_text += "[{}] {}\n\n".format(idx + 1, chunk["text"])
        # Store citation (Title + Date)
        citations.append(
            "[{}] {} ({})".format(
                idx + 1,
                chunk["metadata"]["title"],
                chunk["metadata"]["date"]
            )
        )

    # Truncate context if it is too long to fit in the model
    context_text = context_text[:MAX_CONTEXT_CHARS]
    return context_text, citations

def generate_answer(tokenizer, model, question, context_text):
    """Generate an answer using the FLAN-T5 model."""
    
    # Get conversation history from memory (Step 5)
    conversation_history = memory.get_context()

    # Build the Prompt
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

    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False  # Deterministic output
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def print_sample_questions():
    """Print sample questions to guide the user."""
    print("\nYou can ask questions like:")
    print("- What dividends were announced by ZAIN KSA?")
    print("- Which companies reported financial results?")
    print("- Tell me about SAL's 2023 performance.")
    print("- Which company signed a major contract?\n")

# =========================
# 3. Main Chat Loop
# =========================

def chat():
    print("Financial RAG Chatbot Ready (type 'exit' to quit)")
    print_sample_questions()

    print("Loading Database and Model...")
    collection = load_collection()
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    print("System Ready!\n")

    while True:
        try:
            # 1. Get User Input
            question = input("You: ")
            
            # Exit condition
            if question.lower() in ["exit", "quit"]:
                print("Bye!")
                break
            
            # Skip empty input
            if not question.strip():
                continue

            # 2. Add current question to memory first
            memory.add_user(question)

            # 3. Smart Search Logic
            # Combine the PREVIOUS user question with the CURRENT one.
            # This helps resolve pronouns like "it" (e.g., "When was it announced?")
            # without including the entire history which might confuse the retrieval.
            
            search_query = question
            
            # memory.buffer contains tuples: [("User", "msg1"), ("Assistant", "msg2"), ...]
            user_msgs = [msg for role, msg in memory.buffer if role == "User"]
            
            # If there are at least 2 user messages, grab the previous one
            if len(user_msgs) >= 2:
                last_topic = user_msgs[-2] 
                search_query = f"{last_topic} {question}"
                # Debug print to show context-aware retrieval is working
                print(f"   (DEBUG: Searching with context -> '{search_query[:60]}...')")
            
            # 4. Retrieve Documents
            chunks = retrieve_chunks(collection, search_query)

            if not chunks:
                print("\nAssistant: This question does not appear related to financial announcements.\n")
                # Remove the unanswered question from memory to prevent confusion
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
