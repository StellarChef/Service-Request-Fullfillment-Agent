from typing import Literal

from pydantic import BaseModel, Field


class ClientInput(BaseModel):
    textclient: str


class Client(BaseModel):
    name: str
    email: str
    phone: str | None


class InputClassificationModel(BaseModel):
    category: Literal["COMPLAINT", "INQUIRY", "SPAM"]
    confidence: float
    language: str
    flagged: bool

class DocumentMetadata(BaseModel):
    """Document-level description required for every document added to the vector DB.
    Copied into the metadata of each chunk, so chunks can be filtered and removed by doc_id."""
    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")  # e.g. "regulamin-pracy-2026"
    doc_title: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)  # e.g. "regulamin", "polityka_odpowiedzi"
    version: str = Field(min_length=1)
    is_latest: bool
    language: str = Field(pattern=r"^[a-z]{2}$")  # ISO 639-1
    valid_from: int = Field(ge=0)  # unix timestamp (seconds)
    access_groups: list[str] = Field(min_length=1)
    source_url: str | None = None


class ChunkMetadata(DocumentMetadata):
    """Metadata stored with a single chunk. Structural fields are optional because
    the current Chunker splits by characters and does not know chapters or pages."""
    chunk_index: int = Field(ge=0)
    chapter_number: int | None = None
    chapter_title: str | None = None
    paragraph: int | None = None
    section_path: str | None = None  # e.g. "Rozdział 1 > § 2"
    page: int | None = None


class Policychunk(BaseModel):
    id: str
    text: str
    similarity: float
    category: Literal["COMPLAINT", "INQUIRY", "SPAM"]

class ServiceResponse(BaseModel):
  classification: InputClassificationModel
  matched_policies: list[Policychunk]
  reply: str
  flagged: bool

