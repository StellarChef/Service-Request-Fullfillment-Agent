from .connection import client
from AI.chunker import Chunker, to_chroma_metadata
from AI.openaiclient import AIClient
from models.schemas import DocumentMetadata
from AI.documentnormalizer import DocumentNormalizer


class Database:
    def __init__(self):
        self.client = client
        self.chunker = Chunker()
        self.ai = AIClient()
        self.dn = DocumentNormalizer(self.ai)  # the normalizer needs a client with parse()

    def get_create_collection(self, collection_name):
        db = self.client

        collection = db.get_or_create_collection(collection_name)
        return collection

    def add_document(self, col_name: str, document: str, meta: DocumentMetadata):
        """normalize -> chunk -> upsert."""
        canonical = self.dn.normalize(document, meta)
        chunks = self.chunker.chunk_document(canonical)
        if not chunks:
            raise ValueError(f"Document '{meta.doc_id}' produced no chunks")

        texts = [text for text, _ in chunks]
        collection = self.get_create_collection(col_name)
        collection.upsert(
            ids=[f"{meta.doc_id}#c{m.chunk_index}" for _, m in chunks],
            documents=texts,
            embeddings=self.ai.embed_many(texts),
            metadatas=[to_chroma_metadata(m) for _, m in chunks],
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

