from AI.chunker import Chunker
from AI.openaiclient import AIClient
from config.config import BASE_DIR, RESPONSE_POLICIES_COLLECTION
from database.pgdb import Database

POLICY_FILE = BASE_DIR / "policies" / "communication_procedures.md"


if __name__ == "__main__":
    ai = AIClient()
    db = Database()

    # 1. Chunking
    text = POLICY_FILE.read_text(encoding="utf-8")
    chunks = Chunker().chunk_text(text)
    print(f"[chunker] {len(text)} znaków -> {len(chunks)} chunków")

    # 2. Embeddingi
    embeddings = ai.embed_many(chunks)
    print(f"[openai] {len(embeddings)} embeddingów, wymiar {len(embeddings[0])}")

    # 3. Kolekcja
    collection = db.get_create_collection(RESPONSE_POLICIES_COLLECTION)
    print(f"[chroma] kolekcja '{collection.name}', przed: {collection.count()} rekordów")

    # 4. Upload - upsert ze stałymi id, żeby ponowne odpalenie nie dublowało danych
    collection.upsert(
        ids=[f"{POLICY_FILE.stem}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": POLICY_FILE.name, "chunk_index": i, "scope": "universal"}
            for i in range(len(chunks))
        ],
    )
    print(f"[chroma] po: {collection.count()} rekordów")

    # 5. Similarity search - query_embeddings, bo kolekcja nie zna modelu OpenAI
    question = "Klient pyta o status swojego zamówienia, co mam zrobić?"
    result = collection.query(
        query_embeddings=[ai.embed_text(question)],
        n_results=3,
    )
    print(f"\n[query] {question}")
    for doc, dist in zip(result["documents"][0], result["distances"][0]):
        print(f"  {dist:.3f} | {doc[:100]!r}")
