from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config.config import CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY, RESPONSE_POLICIES_COLLECTION


@lru_cache(maxsize=1)
def get_client() -> ClientAPI:
    # one client per process, created lazily on first use
    return chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )


class DatabaseClient:

    def __init__(self):
        self.client = get_client()

    def get_collection(self, name: str = RESPONSE_POLICIES_COLLECTION) -> Collection:
        # embedding_function=None because embeddings are computed by AIClient
        return self.client.get_or_create_collection(name=name, embedding_function=None)
