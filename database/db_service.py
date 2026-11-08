import traceback
from pathlib import Path

from config.config import BASE_DIR, RESPONSE_POLICIES_COLLECTION
from database.pgdb import Database
from models.schemas import DocumentMetadata

POLICY_FILE = BASE_DIR / "policies" / "communication_procedures.md"

class ServiceDatabase():
    def __init__(self):
        self.db = Database()
    def upload_policy(self, policy: Path, meta: DocumentMetadata, collection_name: str) -> bool:
        """Uploads a policy via Database.add_document. True = data is in the collection."""
        try:
            self.db.add_document(collection_name, policy.read_text(encoding="utf-8"), meta)
            return self.db.get_create_collection(collection_name).count() > 0
        except Exception:
            traceback.print_exc()
            return False

    def get_policies(self, col_name):
            return self.db.collection_list(col_name)

    def remove_policy(self, col_name, doc_id) -> bool:
        self.db.get_create_collection(col_name).delete(where={"doc_id": doc_id})

service = ServiceDatabase()
if __name__ == "__main__":
    print(service.get_policies("policies"))
