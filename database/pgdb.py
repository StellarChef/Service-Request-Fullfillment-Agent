from config_db import client


class Database:
    def __init__(self):
        self.client = client

    def get_create_collection(self, collection_name):
        db = self.client

        collection = db.get_or_create_collection(collection_name)
        if collection in db.list_collections():
            return collection

    def add_document(self, col_name, document, doc_id):
        collection = self.get_create_collection(col_name)
        chunks = self.chunker.chunk_text(document)

        collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
        )

    def query_collection(
        self, col_name: str, query_embedding: list[float], n_results=3
    ):
        results = self.client.query(
            collection_name=col_name,
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results
