import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from transformers import pipeline
from memory import ConversationMemory

DB_DIR = "./chroma_db"
COLLECTION_NAME = "financial_rag"
TOP_K = 3
GEN_MODEL = "google/flan-t5-base"

memory = ConversationMemory(max_turns=5)

def load_collection():
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=DB_DIR, settings=settings)
    return client.get_collection(name=COLLECTION_NAME)

def retrieve_chunks(collection, question, k=TOP_K):
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas"]
    )

    chunks = []
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
        context_text += f"[{idx+1}] {chunk['text']}\n\n"
        citations.append(f"[{idx+1}] {chunk['metadata']['title']} ({chunk['metadata']['date']})")

    return context_text, citations

def generate_answer(generator, question, context_text):
    conversation_history = memory.get_context()

    prompt = f"""
You are a financial domain expert assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Conversation History:
{conversation_history}

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
End your answer with:
Sources: [1], [2], ...
"""

    result = generator(
        prompt,
        max_new_tokens=256,
        truncation=True
    )

    return result[0]["generated_text"]

def chat():
    print("Financial RAG Chatbot Ready (type 'exit' to quit)")
    collection = load_collection()
    generator = pipeline("text2text-generation", model=GEN_MODEL)

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        memory.add_user(question)

        chunks = retrieve_chunks(collection, question)
        context_text, citations = format_context(chunks)

        answer = generate_answer(generator, question, context_text)

        memory.add_assistant(answer)

        print("\nAssistant:\n")
        print(answer)

        print("\nCitations:")
        for cite in citations:
            print(cite)

if __name__ == "__main__":
    chat()