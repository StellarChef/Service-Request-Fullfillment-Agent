from models.schemas import CanonicalDocument, CanonicalSection, ChunkMetadata

# Kinds whose number is stored as ChunkMetadata.paragraph
_PARAGRAPH_KINDS = {"section", "article", "point"}


def to_chroma_metadata(meta: ChunkMetadata) -> dict:
    """Chroma rejects None values, so unset optional fields are dropped.
    access_groups stays a list: chromadb 1.5.x stores lists and supports $contains on them."""
    return meta.model_dump(exclude_none=True)


class Chunker:
    def __init__(self, chunk_size=500,overlap=25):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_document(self, doc: CanonicalDocument, max_chars: int = 2000) -> list[tuple[str, ChunkMetadata]]:
        """One section = one chunk; longer sections are split by paragraphs, then by size.
        Returns (text to embed, metadata) pairs; the text starts with the section path."""
        result: list[tuple[str, ChunkMetadata]] = []
        title = doc.metadata.doc_title
        for section in doc.sections:
            body = section.body().strip()
            if not body:
                # e.g. a chapter heading followed directly by its first paragraph
                continue
            header = f"{title} > {section.path}" if section.path else title
            chapter = self._chapter_of(doc, section)
            for piece in self._split(body, max_chars):
                meta = ChunkMetadata(
                    **doc.metadata.model_dump(),
                    chunk_index=len(result),
                    section_path=section.path or None,
                    section_kind=section.kind,
                    section_number=section.number,
                    chapter_number=chapter.number_int if chapter else None,
                    chapter_title=(chapter.title or chapter.label) if chapter else None,
                    paragraph=section.number_int if section.kind in _PARAGRAPH_KINDS else None,
                )
                result.append((f"{header}\n{piece}", meta))
        return result

    @staticmethod
    def _chapter_of(doc: CanonicalDocument, section: CanonicalSection) -> CanonicalSection | None:
        """The section itself or its nearest ancestor of kind "chapter"."""
        current: CanonicalSection | None = section
        while current is not None:
            if current.kind == "chapter":
                return current
            current = doc.sections[current.parent] if current.parent is not None else None
        return None

    def _split(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        pieces: list[str] = []
        buffer = ""
        for paragraph in (p.strip() for p in text.split("\n\n")):
            if not paragraph:
                continue
            if len(paragraph) > max_chars:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                sized = Chunker(chunk_size=max_chars, overlap=min(self.overlap, max_chars - 1))
                pieces.extend(sized.chunk_text(paragraph))
                continue
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) > max_chars:
                pieces.append(buffer)
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            pieces.append(buffer)
        return pieces
