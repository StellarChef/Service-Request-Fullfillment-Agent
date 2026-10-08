from pathlib import Path
from dotenv import load_dotenv

#This is place for collection of all the configuration variables for the project.

env = load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "AI" / "prompts"



