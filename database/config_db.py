from config.config import CHROMA_TENANT,CHROMA_DATABASE,CHROMA_API_KEY
from .chroma import DBClient

client = DBClient(CHROMA_TENANT,CHROMA_DATABASE,CHROMA_API_KEY).get_client()