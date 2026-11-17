"""Step 3 of the pipeline: matches policies to a classified client request.

Bridge between the vector DB (Database) and the business logic (AIService):
embeds the request, searches the policy collections and returns the fragments
that apply to it.
"""
import logging

from chromadb.errors import NotFoundError

from AI.openaiclient import AIClient
from config.config import (
    COMPANY_POLICIES_COLLECTION,
    RESPONSE_POLICIES_COLLECTION,
    VERIFICATION_POLICIES_COLLECTION,
)
from database.pgdb import Database
from models.schemas import InputClassificationModel, Policychunk


logger = logging.getLogger(__name__)


class PolicyMatcher:
    # result key -> Chroma collection searched by retrieve_all()
    COLLECTIONS = {
        "procedures": RESPONSE_POLICIES_COLLECTION,
        "policies": COMPANY_POLICIES_COLLECTION,
        "verification": VERIFICATION_POLICIES_COLLECTION,
    }

    def __init__(self, client: AIClient, db: Database):
        self.client = client
        self.db = db

    def retrieve_all(
        self,
        text: str,
        classification: InputClassificationModel,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict[str, list[Policychunk]]:
        """Embeds the request once and returns the nearest chunks from every policy
        collection, closest first:
            {"procedures": [...], "policies": [...], "verification": [...]}
        Each list is the input of reasoning()."""
        embedding = self.client.embed_text(text)
        found = {}
        for key, collection in self.COLLECTIONS.items():
            try:
                found[key] = self.retrieve(embedding, classification, collection, n_results, where)
            except NotFoundError:
                logger.warning("Policy collection '%s' does not exist, skipping", collection)
                found[key] = []
        return found

    def retrieve(
        self,
        query: str | list[float],
        classification: InputClassificationModel,
        collection_name: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[Policychunk]:
        """Returns the n nearest chunks from one collection, closest first.

        query: request text (embedded here) or a ready embedding, so one embedding
        can be reused across collections. Output is the input of reasoning()."""
        embedding = self.client.embed_text(query) if isinstance(query, str) else query
        results = self.db.query_collection(
            collection_name, embedding, n_results=n_results, where=where
        )

        # Chroma returns one list per query embedding; we send a single query
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            Policychunk(
                id=chunk_id,
                text=document,
                similarity=self._similarity(distance),
                category=classification.category,
                collection=collection_name,
                metadata=metadata or {},
            )
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    @staticmethod
    def _similarity(distance: float) -> float:
        """Converts a Chroma distance to cosine similarity.

        Collections use Chroma's default squared L2 distance. OpenAI embeddings are
        unit-length, so ||a - b||^2 = 2 - 2*cos and cos = 1 - d/2.
        If a collection is ever created with "cosine" space, this must become 1 - d.
        """
        return 1 - distance / 2

    def reasoning(
        self, text: str, classification: InputClassificationModel, chunks: list[Policychunk]
    ) -> list[Policychunk]:
        """Completes the reasoning for chunks found in the DB: sets reason_percentage
        and drops fragments that do not apply to the request.

        Placeholder: passes the DB results through unchanged.
        """
        return chunks

    def to_rules(self, procedures: list[Policychunk], policies: list[Policychunk]) -> dict:
        """Turns reasoned chunks into the Jinja context for the reply prompt.

        procedures: fragments from the response procedures (how to talk to the client),
        policies: fragments from the company policies (what the client is entitled to).
        Expects the output of reasoning(), e.g.:
            to_rules(self.reasoning(text, cls, procedures), self.reasoning(text, cls, policies))
        """
        return {"procedures": procedures, "policies": policies}
