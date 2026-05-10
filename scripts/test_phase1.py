"""Phase 1 검증 스크립트 — Ollama + Milvus 연동 확인"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL, MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, EMBED_DIM


def check_ollama():
    logger.info("=" * 50)
    logger.info("[1] Ollama 연결 및 모델 확인")
    from src.llm.ollama_client import OllamaClient
    client = OllamaClient()

    ok = client.health_check()
    assert ok, "Ollama 서버에 연결할 수 없습니다. `ollama serve` 실행 여부를 확인하세요."

    models = client.list_models()
    logger.info(f"  설치된 모델: {models}")

    missing = [m for m in [OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL] if m not in models]
    if missing:
        logger.warning(f"  미설치 모델: {missing} — 설치 필요: ollama pull <model>")
    else:
        logger.success(f"  LLM({OLLAMA_LLM_MODEL}), Embed({OLLAMA_EMBED_MODEL}) 모두 준비됨")

    return models


def check_embedding(models: list[str]):
    logger.info("=" * 50)
    logger.info("[2] 임베딩 테스트")

    if OLLAMA_EMBED_MODEL not in models:
        logger.warning(f"  {OLLAMA_EMBED_MODEL} 미설치 — 임베딩 테스트 건너뜀")
        return None

    from src.embeddings.embedder import OllamaEmbedder
    embedder = OllamaEmbedder()

    sample_texts = [
        "정보시스템 감리는 발주기관의 이익을 보호하기 위해 수행됩니다.",
        "클라우드 전환 사업에서는 데이터 이행 검증이 핵심 감리 포인트입니다.",
        "RAG 기반 검색은 기존 유사 제안서에서 관련 콘텐츠를 자동 추출합니다.",
    ]

    vectors = embedder.embed_batch(sample_texts)
    dim = len(vectors[0])
    assert dim == EMBED_DIM, f"임베딩 차원 불일치: expected {EMBED_DIM}, got {dim}"
    logger.success(f"  임베딩 성공: {len(vectors)}개 벡터, 차원={dim}")
    return list(zip(sample_texts, vectors))


def check_milvus(embedded_data):
    logger.info("=" * 50)
    logger.info("[3] Milvus 연결 및 CRUD 테스트")

    from src.vectorstore.milvus_store import MilvusStore
    store = MilvusStore()

    assert store.health_check(), "Milvus 연결 실패"
    logger.info(f"  컬렉션 '{MILVUS_COLLECTION}' 엔티티 수: {store.count()}")

    if embedded_data is None:
        logger.warning("  임베딩 데이터 없음 — 삽입/검색 테스트 건너뜀")
        return

    texts = [d[0] for d in embedded_data]
    vectors = [d[1] for d in embedded_data]
    sources = ["test_doc.txt"] * len(texts)
    doc_types = ["rfp"] * len(texts)
    slide_indices = list(range(len(texts)))

    inserted = store.insert(texts, vectors, sources, doc_types, slide_indices)
    assert inserted == len(texts), f"삽입 수 불일치: {inserted} != {len(texts)}"
    logger.success(f"  삽입 성공: {inserted}개")

    results = store.search(vectors[0], top_k=3)
    assert len(results) > 0, "검색 결과 없음"
    logger.success(f"  유사도 검색 성공: top-{len(results)} 결과")
    for r in results:
        logger.info(f"    score={r['score']:.4f} | {r['text'][:60]}...")

    store.drop()
    logger.info(f"  테스트 컬렉션 정리 완료")


def check_llm_generate(models: list[str]):
    logger.info("=" * 50)
    logger.info("[4] LLM 생성 테스트")

    if OLLAMA_LLM_MODEL not in models:
        logger.warning(f"  {OLLAMA_LLM_MODEL} 미설치 — LLM 생성 테스트 건너뜀")
        return

    from src.llm.ollama_client import OllamaClient
    client = OllamaClient()

    system = "당신은 정보시스템 감리 전문가입니다. 간결하게 한국어로 답변하세요."
    prompt = "클라우드 전환 사업에서 가장 중요한 감리 포인트를 한 문장으로 설명하세요."

    logger.info("  LLM 응답 생성 중...")
    response = client.generate(prompt, system=system)
    logger.success(f"  LLM 응답:\n  {response}")


def main():
    logger.info("AutoReport Phase 1 검증 시작")
    logger.info(f"  Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    logger.info(f"  LLM 모델: {OLLAMA_LLM_MODEL}")
    logger.info(f"  임베딩 모델: {OLLAMA_EMBED_MODEL}")

    models = check_ollama()
    embedded = check_embedding(models)
    check_milvus(embedded)
    check_llm_generate(models)

    logger.info("=" * 50)
    logger.success("Phase 1 검증 완료!")


if __name__ == "__main__":
    main()
