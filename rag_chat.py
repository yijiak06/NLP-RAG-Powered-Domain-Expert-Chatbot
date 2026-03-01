import logging
# Silence ChromaDB warnings to keep the output clean
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import deque

# =========================
# 1. Configuration
# =========================
DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"
TOP_K = 3
GEN_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 1500
SIMILARITY_THRESHOLD = 0.5 

# =========================
# 2. Memory Class (Integrated)
# =========================
class ConversationMemory:
    def __init__(self, max_turns=5):
        self.buffer = deque(maxlen=max_turns * 2)

    def add_user(self, text):
        self.buffer.append(("User", text))

    def add_assistant(self, text):
        self.buffer.append(("Assistant", text))

    def get_context(self):
        return "\n".join(
            [f"{role}: {message}" for role, message in self.buffer]
        )

    def clear(self):
        self.buffer.clear()

# Initialize Memory
memory = ConversationMemory(max_turns=5)

# =========================
# 3. Core Functions
# =========================

def load_collection():
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    return client.get_collection(name=COLLECTION_NAME)

def retrieve_chunks(collection, question, k=TOP_K):
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    if results["documents"]:
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

    return chunks

def format_context(chunks):
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
    print("\nYou can ask questions like:")
    print("- What dividends were announced by ZAIN KSA?")
    print("- Which companies reported financial results?")
    print("- Tell me about SAL's 2023 performance.")
    print("- Which company signed a major contract?\n")

# =========================
# 4. Main Chat Loop
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
            question = input("You: ")

            if question.lower() in ["exit", "quit"]:
                print("Bye!")
                break

            if not question.strip():
                continue

            memory.add_user(question)

            search_query = question
            user_msgs = [msg for role, msg in memory.buffer if role == "User"]

            if len(user_msgs) >= 2:
                last_topic = user_msgs[-2]
                search_query = f"{last_topic} {question}"
                print(f"   (DEBUG: Searching with context -> '{search_query[:60]}...')")

            chunks = retrieve_chunks(collection, search_query)

            if not chunks:
                print("\nAssistant: This question does not appear related to financial announcements.\n")
                if memory.buffer and memory.buffer[-1][0] == "User":
                    memory.buffer.pop()
                continue

            context_text, citations = format_context(chunks)
            answer = generate_answer(tokenizer, model, question, context_text)

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