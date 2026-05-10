from src.parsers.ppt_parser import SlideChunk
from src.parsers.doc_parser import DocChunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    문자 기준 슬라이딩 윈도우 청킹.
    문장 경계(\n)를 최대한 존중하며 분할.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # 청크 끝에서 가장 가까운 줄바꿈 위치 탐색 (최대 100자 이내)
        boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end  # 줄바꿈 없으면 그냥 자름

        chunks.append(text[start:boundary].strip())
        start = boundary - overlap

    return [c for c in chunks if c.strip()]


def chunk_slide(slide: SlideChunk, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[SlideChunk]:
    """슬라이드 하나를 필요시 여러 청크로 분할."""
    parts = _split_text(slide.text, chunk_size, overlap)
    if len(parts) == 1:
        return [slide]

    result = []
    for i, part in enumerate(parts):
        result.append(SlideChunk(
            text=part,
            source=slide.source,
            slide_index=slide.slide_index,
            slide_title=slide.slide_title,
            doc_type=slide.doc_type,
            metadata={**slide.metadata, "chunk_part": i},
        ))
    return result


def chunk_doc(doc: DocChunk, chunk_size: int = CHUNK_SIZE,
              overlap: int = CHUNK_OVERLAP) -> list[DocChunk]:
    """문서 청크 하나를 필요시 여러 청크로 분할."""
    parts = _split_text(doc.text, chunk_size, overlap)
    if len(parts) == 1:
        return [doc]

    result = []
    for i, part in enumerate(parts):
        result.append(DocChunk(
            text=part,
            source=doc.source,
            page_index=doc.page_index,
            doc_type=doc.doc_type,
            metadata={**doc.metadata, "chunk_part": i},
        ))
    return result


def chunk_all(items: list[SlideChunk | DocChunk],
              chunk_size: int = CHUNK_SIZE,
              overlap: int = CHUNK_OVERLAP) -> list[SlideChunk | DocChunk]:
    """SlideChunk / DocChunk 리스트 일괄 청킹."""
    result = []
    for item in items:
        if isinstance(item, SlideChunk):
            result.extend(chunk_slide(item, chunk_size, overlap))
        else:
            result.extend(chunk_doc(item, chunk_size, overlap))
    return result
