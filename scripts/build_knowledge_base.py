"""
Phase 2 — 지식베이스 구축 스크립트

사용법:
  # data/input/ 전체 인덱싱
  python scripts/build_knowledge_base.py

  # 특정 디렉토리만
  python scripts/build_knowledge_base.py --ppt-dir ./data/input/ppt --rfp-dir ./data/input/rfp

  # 기존 컬렉션 초기화 후 재구축
  python scripts/build_knowledge_base.py --reset
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from tqdm import tqdm

from src.config import INPUT_DIR
from src.parsers.ppt_parser import parse_pptx_dir, SlideChunk
from src.parsers.doc_parser import parse_rfp_dir, DocChunk
from src.parsers.chunker import chunk_all
from src.embeddings.embedder import OllamaEmbedder
from src.vectorstore.milvus_store import MilvusStore


def build(ppt_dir: Path, rfp_dir: Path, reset: bool = False):
    logger.info("=" * 60)
    logger.info("AutoReport 지식베이스 구축 시작")

    # ── 1. 파싱 ────────────────────────────────────────────────
    logger.info(f"[1] 문서 파싱")
    ppt_chunks = parse_pptx_dir(ppt_dir)
    rfp_chunks = parse_rfp_dir(rfp_dir, doc_type="rfp")

    all_raw = ppt_chunks + rfp_chunks
    if not all_raw:
        logger.error("인덱싱할 문서가 없습니다. data/input/ 에 파일을 넣어주세요.")
        logger.info("  PPT:     data/input/ppt/*.pptx")
        logger.info("  RFP:     data/input/rfp/*.pdf 또는 *.docx")
        return

    logger.info(f"  총 파싱: PPT {len(ppt_chunks)}개 + RFP {len(rfp_chunks)}개")

    # ── 2. 청킹 ────────────────────────────────────────────────
    logger.info(f"[2] 텍스트 청킹")
    chunks = chunk_all(all_raw)
    logger.info(f"  청킹 후: {len(all_raw)}개 → {len(chunks)}개")

    # ── 3. 임베딩 ──────────────────────────────────────────────
    logger.info(f"[3] 임베딩 생성")
    embedder = OllamaEmbedder()
    if not embedder.health_check():
        logger.error("Ollama 임베딩 서버 연결 실패. ollama serve 확인.")
        return

    texts = [c.text for c in chunks]
    vectors = []
    for text in tqdm(texts, desc="임베딩", unit="chunk"):
        vectors.append(embedder.embed(text))

    # ── 4. Milvus 저장 ─────────────────────────────────────────
    logger.info(f"[4] Milvus 저장")
    store = MilvusStore()

    if reset:
        logger.warning("  컬렉션 초기화 (--reset)")
        try:
            store.drop()
        except Exception:
            pass
        store = MilvusStore()

    sources = [c.source for c in chunks]
    doc_types = [c.doc_type for c in chunks]
    slide_indices = [c.slide_index for c in chunks]

    inserted = store.insert(texts, vectors, sources, doc_types, slide_indices)
    logger.success(f"  저장 완료: {inserted}개 벡터 (총 컬렉션: {store.count()}개)")

    # ── 5. 요약 ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.success("지식베이스 구축 완료!")
    logger.info(f"  컬렉션: {store.collection_name}")
    logger.info(f"  총 벡터: {store.count()}")
    _print_source_summary(chunks)


def _print_source_summary(chunks):
    from collections import Counter
    sources = Counter(Path(c.source).name for c in chunks)
    logger.info("  파일별 청크 수:")
    for name, cnt in sorted(sources.items()):
        logger.info(f"    {name}: {cnt}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoReport 지식베이스 구축")
    parser.add_argument("--ppt-dir", type=Path, default=INPUT_DIR / "ppt",
                        help="전략 PPT 디렉토리 (기본: data/input/ppt)")
    parser.add_argument("--rfp-dir", type=Path, default=INPUT_DIR / "rfp",
                        help="RFP 문서 디렉토리 (기본: data/input/rfp)")
    parser.add_argument("--reset", action="store_true",
                        help="기존 컬렉션 삭제 후 재구축")
    args = parser.parse_args()

    args.ppt_dir.mkdir(parents=True, exist_ok=True)
    args.rfp_dir.mkdir(parents=True, exist_ok=True)

    build(args.ppt_dir, args.rfp_dir, reset=args.reset)
