import chromadb
from config.config import vdb_

chroma_client = chromadb.HttpClient(host='localhost', port=8000) 