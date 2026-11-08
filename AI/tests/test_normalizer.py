"""Offline tests for DocumentNormalizer and Chunker.chunk_document (no API key needed).

Run from the repo root:
    python -m pytest AI/tests/test_normalizer.py
    python -m AI.tests.test_normalizer
"""
from AI.chunker import Chunker, to_chroma_metadata
from AI.documentnormalizer import DocumentNormalizer, number_to_int
from models.schemas import DocumentHeading, DocumentMetadata, DocumentSegmentation


class FakeAIClient:
    """Returns prepared segmentations, one per parse() call, and records the prompts."""

    def __init__(self, *responses: list[DocumentHeading]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def parse(self, messages, schema):
        assert schema is DocumentSegmentation
        self.calls.append(messages)
        headings = self.responses.pop(0) if self.responses else []
        return DocumentSegmentation(headings=headings)


def H(line, kind, level, number=None, title=None):
    return DocumentHeading(line=line, kind=kind, level=level, number=number, title=title)


def meta(doc_id="regulamin-pracy-2026", title="Regulamin pracy", language="pl"):
    return DocumentMetadata(
        doc_id=doc_id, doc_title=title, doc_type="regulamin", version="2026",
        is_latest=True, language=language, valid_from=1767225600,
        access_groups=["wszyscy_pracownicy"],
    )


LONG_PARAGRAPHS = "\n\n".join(f"Akapit {i}. " + "x" * 280 for i in range(3))
HUGE_PARAGRAPH = "y" * 1200

PL_DOC = (
    "Regulamin pracy\n"                                  # 1
    "\n"                                                 # 2
    "Niniejszy regulamin określa organizację pracy.\n"   # 3
    "\n"                                                 # 4
    "Rozdział I Postanowienia ogólne\n"                  # 5
    "§ 1\n"                                              # 6
    "Regulamin obowiązuje wszystkich pracowników.\n"     # 7
    "§ 2\n"                                              # 8
    "Pracodawca zapewnia warunki pracy.\n"               # 9
    "Rozdział II Obowiązki\n"                            # 10
    "§ 3\n"                                              # 11
    f"{LONG_PARAGRAPHS}\n\n{HUGE_PARAGRAPH}\n"           # 12+
)
PL_HEADINGS = [
    H(5, "chapter", 1, "I", "Postanowienia ogólne"),
    H(6, "section", 2, "§ 1"),
    H(8, "section", 2, "§ 2"),
    H(10, "chapter", 1, "II", "Obowiązki"),
    H(11, "section", 2, "§ 3"),
]

EN_DOC = (
    "Chapter 1 Scope\n"           # 1
    "Section 1.1 Purpose\n"       # 2
    "This policy defines rules.\n"  # 3
    "Section 1.2 Definitions\n"   # 4
    "Customer means a person.\n"  # 5
    "Annex A Templates\n"         # 6
    "Template text.\n"            # 7
)
# sections are marked with level 3 under a level-1 chapter: the normalizer flattens them to 2
EN_HEADINGS = [
    H(1, "chapter", 1, "1", "Scope"),
    H(2, "section", 3, "1.1", "Purpose"),
    H(4, "section", 3, "1.2", "Definitions"),
    H(6, "annex", 1, "A", "Templates"),
]


def test_number_to_int():
    assert number_to_int("IV", "chapter") == 4
    assert number_to_int("2.3", "section") == 2
    assert number_to_int("§ 5", "section") == 5
    assert number_to_int("a", "point") is None
    assert number_to_int("i", "point") is None
    assert number_to_int("i", "chapter") == 1
    assert number_to_int("XIV", "article") == 14
    assert number_to_int("A", "annex") is None
    assert number_to_int(None, "chapter") is None


def test_polish_document_sections():
    doc = DocumentNormalizer(FakeAIClient(PL_HEADINGS)).normalize(PL_DOC, meta())
    s = doc.sections

    assert "".join(x.text for x in s) == PL_DOC  # exact slices covering the whole document
    assert s[0].kind == "preamble" and s[0].end_line == 4 and s[0].path == ""
    assert [x.path for x in s[1:]] == [
        "Rozdział I Postanowienia ogólne",
        "Rozdział I Postanowienia ogólne > § 1",
        "Rozdział I Postanowienia ogólne > § 2",
        "Rozdział II Obowiązki",
        "Rozdział II Obowiązki > § 3",
    ]
    assert [x.number_int for x in s[1:]] == [1, 1, 2, 2, 3]
    assert s[3].text == "§ 2\nPracodawca zapewnia warunki pracy.\n"
    assert s[3].parent == 1 and s[3].level == 2


def test_english_document_levels_are_flattened():
    doc = DocumentNormalizer(FakeAIClient(EN_HEADINGS)).normalize(EN_DOC, meta("policy-en", "Policy", "en"))
    s = doc.sections

    assert "".join(x.text for x in s) == EN_DOC
    assert s[0].kind == "chapter"  # no preamble when the document starts with a heading
    assert [x.level for x in s] == [1, 2, 2, 1]
    assert s[2].path == "Chapter 1 Scope > Section 1.2 Definitions"
    assert s[3].kind == "annex" and s[3].number_int is None and s[3].parent is None


def test_fallback_without_headings():
    text = "Just text.\n\nNo headings here.\n"
    doc = DocumentNormalizer(FakeAIClient([])).normalize(text, meta())
    assert len(doc.sections) == 1
    assert doc.sections[0].kind == "other" and doc.sections[0].text == text

    chunks = Chunker().chunk_document(doc)
    assert len(chunks) == 1 and chunks[0][0] == "Regulamin pracy\nJust text.\n\nNo headings here."


def test_chunk_document_metadata_and_split():
    doc = DocumentNormalizer(FakeAIClient(PL_HEADINGS)).normalize(PL_DOC, meta())
    chunks = Chunker().chunk_document(doc, max_chars=500)

    # chapter headings have no own body, so they produce no chunk
    paths = [m.section_path for _, m in chunks]
    assert "Rozdział I Postanowienia ogólne" not in paths

    preamble_text, preamble_meta = chunks[0]
    assert preamble_text.startswith("Regulamin pracy\n") and preamble_meta.section_kind == "preamble"

    text, m = next(c for c in chunks if c[1].section_number == "§ 2")
    assert text == "Regulamin pracy > Rozdział I Postanowienia ogólne > § 2\nPracodawca zapewnia warunki pracy."
    assert (m.chapter_number, m.chapter_title, m.paragraph) == (1, "Postanowienia ogólne", 2)
    assert m.doc_id == "regulamin-pracy-2026"

    long_chunks = [c for c in chunks if c[1].section_number == "§ 3"]
    assert len(long_chunks) > 3  # 3 paragraphs + a 1200-char paragraph split by chunk_text
    header = "Regulamin pracy > Rozdział II Obowiązki > § 3\n"
    assert all(t.startswith(header) and len(t) - len(header) <= 500 for t, _ in long_chunks)
    assert all(m.chapter_number == 2 and m.paragraph == 3 for _, m in long_chunks)

    assert [m.chunk_index for _, m in chunks] == list(range(len(chunks)))
    stored = to_chroma_metadata(chunks[0][1])
    assert None not in stored.values() and stored["access_groups"] == ["wszyscy_pracownicy"]


def test_windows_reject_bad_headings():
    text = "".join(f"Line {n}\n" if n not in (2, 8) else "\n" for n in range(1, 13))
    client = FakeAIClient(
        [H(1, "chapter", 1, "1"), H(2, "section", 2), H(7, "section", 2)],  # 2 empty, 7 outside 1-5
        [H(6, "section", 2), H(6, "section", 2), H(8, "section", 2)],       # duplicate, 8 empty
        [H(11, "section", 2)],
    )
    doc = DocumentNormalizer(client, window_lines=5).normalize(text, meta())

    assert len(client.calls) == 3
    assert [x.heading_line for x in doc.sections] == [1, 6, 11]
    assert "".join(x.text for x in doc.sections) == text
    # the chapter from window 1 is passed as an open heading to window 2
    assert "level 1, chapter: Line 1" in client.calls[1][0]["content"]
    assert "6 to 10 of a longer document" in client.calls[1][0]["content"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
