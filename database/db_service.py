import traceback

from config.config import BASE_DIR, RESPONSE_POLICIES_COLLECTION
from database.pgdb import Database

POLICY_FILE = BASE_DIR / "policies" / "communication_procedures.md"


def upload_policy(policy:str, collection_name="policies") -> bool:
    """Uploads a policy via Database.add_document. True = data is in the collection."""
    try:
        db = Database()
        db.add_document(collection_name, policy.read_text(encoding="utf-8"), policy.stem)
        return db.get_create_collection(collection_name).count() > 0
    except Exception:
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(upload_policy())

