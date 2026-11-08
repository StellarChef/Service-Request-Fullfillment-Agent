from .connection import client
from AI.chunker import Chunker
from AI.openaiclient import AIClient
from models.schemas import ChunkMetadata, DocumentMetadata


class Database:
    def __init__(self):
        self.client = client
        self.chunker = Chunker()
        self.ai = AIClient()

    def get_create_collection(self, collection_name):
        db = self.client

        collection = db.get_or_create_collection(collection_name)
        return collection

    def add_document(self, col_name: str, document: str, meta: DocumentMetadata):
        chunks = self.chunker.chunk_text(document)
        if not chunks:
            raise ValueError(f"Document '{meta.doc_id}' is empty")

        # Chroma rejects None values, so unset optional fields are dropped
        metadatas = [
            ChunkMetadata(**meta.model_dump(), chunk_index=i).model_dump(exclude_none=True)
            for i in range(len(chunks))
        ]

        collection = self.get_create_collection(col_name)
        collection.upsert(
            ids=[f"{meta.doc_id}#c{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=self.ai.embed_many(chunks),
            metadatas=metadatas,
        )

    def query_collection(
        self, col_name: str, query_embedding: list[float], n_results=3
    ):
        collection = self.client.get_collection(col_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results
    
    def collection_list(self, coll_name):
        db = self.client
        collection = db.get_collection(coll_name)
        return collection.get()

