import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from memory import ConversationMemory

# Configuration
DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"
TOP_K = 3
GEN_MODEL = "google/flan-t5-base"
MAX_CONTEXT_CHARS = 1500
SIMILARITY_THRESHOLD = 0.5

memory = ConversationMemory(max_turns=5)

# Load Chroma Collection
def load_collection():
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    return client.get_collection(name=COLLECTION_NAME)

# Retrieve Relevant Chunks
def retrieve_chunks(collection, question, k=TOP_K):
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        if distance < SIMILARITY_THRESHOLD:
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

    return chunks

# Format Context for Prompt
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

# Generate Answer
def generate_answer(tokenizer, model, question, context_text):
    conversation_history = memory.get_context()

    prompt = """
You are a financial analyst assistant.

Use ONLY the provided context to answer the question.

When answering:
- Include specific numbers (amounts, percentages, dates)
- Mention the company name clearly
- Provide full details if available
- Do not give one-word answers
- If the answer is not in the context, say:
"I could not find relevant information in the database."

Conversation History:
{}

Context:
{}

Question:
{}

Detailed Answer:
""".format(conversation_history, context_text, question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print Sample Questions
def print_sample_questions():
    print("\nYou can ask questions like:")
    print("- What dividends were announced by ZAIN KSA?")
    print("- Which companies reported financial results?")
    print("- Tell me about SAL's 2023 performance.")
    print("- Which company signed a major contract?\n")

# Main Chat Loop
def chat():
    print("Financial RAG Chatbot Ready (type 'exit' to quit)")
    print_sample_questions()

    collection = load_collection()

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            break

        memory.add_user(question)

        chunks = retrieve_chunks(collection, question)

        if not chunks:
            print("\nAssistant:\n")
            print("This question does not appear related to financial announcements in the database.\n")
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

if __name__ == "__main__":
    chat()