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


# Structural unit kinds. "preamble" is assigned by code only (text before the first heading).
HeadingKind = Literal["part", "chapter", "section", "article", "point", "annex", "other"]
SectionKind = Literal["preamble", "part", "chapter", "section", "article", "point", "annex", "other"]


class ChunkMetadata(DocumentMetadata):
    """Metadata stored with a single chunk. Structural fields stay empty when a document
    is chunked by size only (Chunker.chunk_text); Chunker.chunk_document fills them."""
    chunk_index: int = Field(ge=0)
    chapter_number: int | None = None
    chapter_title: str | None = None
    paragraph: int | None = None
    section_path: str | None = None  # e.g. "Rozdział 1 > § 2"
    section_kind: SectionKind | None = None
    section_number: str | None = None  # number as written, e.g. "IV", "2.3", "§ 5"
    page: int | None = None


class DocumentHeading(BaseModel):
    """One heading marked by the LLM. No defaults or constraints here, so the model
    stays compatible with strict structured output; the normalizer validates it."""
    line: int  # 1-based line number in the whole document
    kind: HeadingKind
    level: int  # 1 = top-level unit of this document
    number: str | None  # as written, e.g. "IV", "2.3", "§ 5", "A"
    title: str | None  # title without the number


class DocumentSegmentation(BaseModel):
    """LLM response: headings found in one window of the document."""
    headings: list[DocumentHeading]


class CanonicalSection(BaseModel):
    index: int
    kind: SectionKind
    level: int  # nesting depth after normalization; 0 = preamble
    number: str | None = None
    number_int: int | None = None
    title: str | None = None
    label: str  # heading line as written, used in paths
    path: str  # e.g. "Rozdział 1 Postanowienia ogólne > § 2"; "" for preamble
    parent: int | None = None  # index of the parent section
    heading_line: int | None = None  # None for preamble and the no-heading fallback
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str  # exact slice of the original document, heading line included

    def body(self) -> str:
        """Section text without its heading line."""
        if self.heading_line is None:
            return self.text
        parts = self.text.split("\n", 1)
        return parts[1] if len(parts) > 1 else ""


class CanonicalDocument(BaseModel):
    """Every source document is reduced to this template before chunking."""
    metadata: DocumentMetadata
    sections: list[CanonicalSection]

    def render_markdown(self) -> str:
        """Readable preview of the detected structure (e.g. for an admin before saving)."""
        out = [f"# {self.metadata.doc_title}\n"]
        for s in self.sections:
            if s.heading_line is not None:
                out.append(f"{'#' * min(s.level + 1, 6)} {s.label}  `[{s.kind}]`\n")
            body = s.body().strip()
            if body:
                out.append(body + "\n")
        return "\n".join(out)


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

