# Database handle file for the Service Request Agent

# In this project we reach out to 2 databases:
# - Vector DB (Chroma Cloud): stores our policies, used to analyse the problem the client reaches out with (RAG)
# - Postgres: stores application data

# DatabaseService responsibilities (MVP: add, search, replace):
# - add_document: sanitize (Gate 4: Document Guard) -> chunk -> embed -> upsert
# - search: embed the client question -> return top-k matching chunks (Gate 3: RAG)
# - get_document: fetch all chunks of one document
# - delete_document: remove all chunks of one document (needs doc_id in chunk metadata)
# - replace_document: delete old chunks, then add again (upsert alone leaves stale chunks when the chunk count shrinks)
#
# Backlog:
# - tenant isolation: one collection per tenant + one shared collection for universal procedures
