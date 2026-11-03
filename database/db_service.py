import traceback

from config.config import BASE_DIR, RESPONSE_POLICIES_COLLECTION
from database.pgdb import Database
from database.connection import client

POLICY_FILE = BASE_DIR / "policies" / "communication_procedures.md"


class DBService:
    def __init__(self):
        self.db = Database()


def upload_policy(self, policy: str, collection_name="policies") -> bool:
    """Uploads a policy via Database.add_document. True = data is in the collection."""
    try:
        self.db.add_document(
            collection_name, policy.read_text(encoding="utf-8"), policy.stem
        )
        return self.db.get_create_collection(collection_name).count() > 0
    except Exception:
        traceback.print_exc()
        return False


def get_policy(self, collection_name) -> str:
    """Retrieves a policy via Database.query_collection. Returns the policy as a string."""
    try:
        collection = self.db.get_create_collection(collection_name)
        if collection.count() == 0:
            return "No policy found."
        else:
            # Get the all document in the collection
            results = collection.get(include=["documents"])
            # Concatenate the documents into a single string
            return "\n".join(results["documents"])
    except Exception:
        traceback.print_exc()
        return "Error retrieving policy."


def delete_policy(self, collection_name) -> bool:
    """Deletes a policy via Database.delete_collection. Returns True if the collection was deleted."""
    try:
        self.db.delete_collection(collection_name)
        return True
    except Exception:
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(upload_policy())
