import os
from pathlib import Path
from dotenv import load_dotenv

#This is place for collection of all the configuration variables for the project.

env = load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "AI" / "prompts"

CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")

COMPANY_POLICIES_COLLECTION = "company-policies"
RESPONSE_POLICIES_COLLECTION = "response-policies"
VERIFICATION_POLICIES_COLLECTION = "verification-policies"



