"""
Phase 3 — RAG 파이프라인

흐름:
  사용자 입력 (키워드/기술특성/업무특성)
    → 쿼리 임베딩
    → Milvus 유사 청크 검색
    → 섹션별 프롬프트 조합
    → Ollama LLM 호출
    → 감리제안서 섹션 초안 반환
"""

from dataclasses import dataclass, field
from loguru import logger

from src.embeddings.embedder import OllamaEmbedder
from src.vectorstore.milvus_store import MilvusStore
from src.llm.ollama_client import OllamaClient


# ── 감리제안서 섹션 정의 ──────────────────────────────────────────────────────

SECTIONS = {
    "사업_이해": {
        "label": "사업 이해 및 배경",
        "instruction": "본 사업의 목적, 범위, 주요 특성을 분석하여 감리 관점에서 사업 이해 내용을 작성하세요. 200~300자 내외, 공문서체.",
        "rfp_weight": 0.7,   # RFP 청크 비중
    },
    "감리_전략": {
        "label": "감리 수행 전략",
        "instruction": "유사 사업 사례를 참고하여 본 사업의 감리 수행 전략을 작성하세요. 기술적 위험요소 2개 이상 포함, 300자 내외, 공문서체.",
        "rfp_weight": 0.4,
    },
    "위험_및_대응": {
        "label": "예상 위험 및 대응방안",
        "instruction": "본 사업의 주요 위험요소와 감리 대응방안을 항목별로 작성하세요. 위험요소 3개 이상, 각 50자 이내 대응방안 포함.",
        "rfp_weight": 0.5,
    },
    "감리_체계": {
        "label": "감리 수행 체계",
        "instruction": "감리 조직 구성, 역할, 수행 절차를 기술하세요. 설계·중간·완료 감리 단계 구분 포함, 250자 내외.",
        "rfp_weight": 0.3,
    },
    "기대_효과": {
        "label": "기대 효과",
        "instruction": "본 감리 수행을 통해 발주기관이 얻을 수 있는 기대 효과를 구체적으로 기술하세요. 200자 내외.",
        "rfp_weight": 0.4,
    },
}

SYSTEM_PROMPT = (
    "당신은 15년 경력의 정보시스템 감리 전문가입니다. "
    "제공된 참고 자료와 RFP 요약을 바탕으로 감리제안서 섹션을 작성합니다. "
    "반드시 한국어 공문서체로 작성하고, 지정된 분량을 준수하세요. "
    "참고 자료에 없는 사실을 임의로 추가하지 마세요."
)


# ── 입력/출력 데이터클래스 ────────────────────────────────────────────────────

@dataclass
class QueryInput:
    project_name: str                  # 사업명
    keywords: list[str]                # 키워드 (예: ["QR코드", "방사선안전"])
    tech_tags: list[str] = field(default_factory=list)   # 기술특성
    domain_tags: list[str] = field(default_factory=list) # 업무특성
    audit_type: str = "종합"           # 감리 유형
    top_k: int = 5                     # 검색 결과 수

    @property
    def query_text(self) -> str:
        parts = [self.project_name] + self.keywords + self.tech_tags + self.domain_tags
        return " ".join(parts)


@dataclass
class SectionResult:
    section_key: str
    section_label: str
    content: str
    references: list[dict] = field(default_factory=list)  # 근거 슬라이드 목록
    similarity_warning: bool = False  # 유사도 낮은 경우 경고


@dataclass
class ProposalDraft:
    project_name: str
    audit_type: str
    sections: list[SectionResult] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"# {self.project_name} 감리제안서 초안", f"감리 유형: {self.audit_type}", ""]
        for s in self.sections:
            lines.append(f"## {s.section_label}")
            lines.append(s.content)
            if s.similarity_warning:
                lines.append("⚠️ [유사 참고자료 부족 — 검토 필요]")
            lines.append("")
        return "\n".join(lines)


# ── RAG 파이프라인 ────────────────────────────────────────────────────────────

class RAGPipeline:
    LOW_SIMILARITY_THRESHOLD = 0.5

    def __init__(self):
        self.embedder = OllamaEmbedder()
        self.store = MilvusStore()
        self.llm = OllamaClient()

    def health_check(self) -> dict:
        return {
            "ollama": self.embedder.health_check(),
            "milvus": self.store.health_check(),
        }

    def generate_draft(self, query: QueryInput,
                       sections: list[str] | None = None) -> ProposalDraft:
        """
        감리제안서 초안 생성.
        sections: 생성할 섹션 키 목록 (None이면 전체)
        """
        target_sections = sections or list(SECTIONS.keys())
        logger.info(f"초안 생성 시작: {query.project_name} / 섹션 {len(target_sections)}개")

        # 1. 쿼리 임베딩
        query_vector = self.embedder.embed(query.query_text)

        draft = ProposalDraft(project_name=query.project_name, audit_type=query.audit_type)

        for sec_key in target_sections:
            sec_def = SECTIONS.get(sec_key)
            if not sec_def:
                logger.warning(f"알 수 없는 섹션: {sec_key} — 건너뜀")
                continue

            logger.info(f"  섹션 생성: [{sec_def['label']}]")
            result = self._generate_section(query, query_vector, sec_key, sec_def)
            draft.sections.append(result)

        logger.success(f"초안 생성 완료: {len(draft.sections)}개 섹션")
        return draft

    def _generate_section(self, query: QueryInput, query_vector: list[float],
                          sec_key: str, sec_def: dict) -> SectionResult:
        rfp_weight = sec_def.get("rfp_weight", 0.5)
        n_ppt = max(1, round(query.top_k * (1 - rfp_weight)))
        n_rfp = max(1, round(query.top_k * rfp_weight))

        # 2. 유사 청크 검색 (PPT + RFP 분리)
        ppt_hits = self.store.search(query_vector, top_k=n_ppt, doc_type_filter="ppt")
        rfp_hits = self.store.search(query_vector, top_k=n_rfp, doc_type_filter="rfp")
        all_hits = ppt_hits + rfp_hits

        # 유사도 경고 체크
        low_sim = all_hits and max(h["score"] for h in all_hits) < self.LOW_SIMILARITY_THRESHOLD

        # 3. 프롬프트 조합
        prompt = self._build_prompt(query, sec_def, ppt_hits, rfp_hits)

        # 4. LLM 호출
        content = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        return SectionResult(
            section_key=sec_key,
            section_label=sec_def["label"],
            content=content,
            references=all_hits,
            similarity_warning=low_sim,
        )

    def _build_prompt(self, query: QueryInput, sec_def: dict,
                      ppt_hits: list[dict], rfp_hits: list[dict]) -> str:
        lines = []

        # 컨텍스트: 유사 PPT 슬라이드
        if ppt_hits:
            lines.append("[참고 자료 — 유사 사업 전략]")
            for i, h in enumerate(ppt_hits, 1):
                src = h["source"].split("/")[-1]
                lines.append(f"  [{i}] ({src}, 슬라이드 {h['slide_index']+1})")
                lines.append(f"  {h['text'][:300]}")
            lines.append("")

        # 컨텍스트: RFP 내용
        if rfp_hits:
            lines.append("[RFP 참고 내용]")
            for i, h in enumerate(rfp_hits, 1):
                src = h["source"].split("/")[-1]
                lines.append(f"  [{i}] ({src}, p.{h['slide_index']+1})")
                lines.append(f"  {h['text'][:300]}")
            lines.append("")

        # 사업 정보
        lines.append("[현재 사업 정보]")
        lines.append(f"  사업명: {query.project_name}")
        if query.keywords:
            lines.append(f"  키워드: {', '.join(query.keywords)}")
        if query.tech_tags:
            lines.append(f"  기술특성: {', '.join(query.tech_tags)}")
        if query.domain_tags:
            lines.append(f"  업무특성: {', '.join(query.domain_tags)}")
        lines.append(f"  감리 유형: {query.audit_type}")
        lines.append("")

        # 생성 지시
        lines.append(f"[작성 지시 — {sec_def['label']}]")
        lines.append(sec_def["instruction"])

        return "\n".join(lines)

    def search_similar(self, query_text: str, top_k: int = 5,
                       doc_type: str | None = None) -> list[dict]:
        """단순 유사 문서 검색 (섹션 생성 없이 검색만)."""
        vector = self.embedder.embed(query_text)
        return self.store.search(vector, top_k=top_k, doc_type_filter=doc_type)
