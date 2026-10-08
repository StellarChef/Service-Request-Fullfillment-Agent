from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from config.config import CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY, POLICIES_COLLECTION

POLICIES_COLLECTION = POLICIES_COLLECTION

@lru_cache(maxsize=1)
def get_client() -> ClientAPI:
    # one client per process, created lazily on first use
    return chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )


def get_collection(name: str = POLICIES_COLLECTION) -> Collection:
    # embedding_function=None because embeddings are computed by AIClient
    return get_client().get_or_create_collection(name=name, embedding_function=None)
