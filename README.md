# AutoReport — 감리제안서 자동작성 시스템

Local LLM + RAG 기반 완전 오프라인 감리제안서 초안 생성 시스템

## 기술 스택

| 구분 | 기술 |
|------|------|
| LLM | Ollama (EXAONE-3.5:7.8b / qwen2.5:14b) |
| 임베딩 | nomic-embed-text (Ollama) |
| 벡터 DB | Milvus Standalone (Docker) |
| RAG | LangChain |
| UI | Gradio |
| 출력 | python-docx |

## 빠른 시작

### 1. 사전 요구사항

- Python 3.9+
- Docker (Colima 또는 Docker Desktop)
- Ollama

### 2. 환경 구성

```bash
# 가상환경 생성 및 패키지 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
```

### 3. Milvus 실행

```bash
docker-compose up -d
```

### 4. Ollama 모델 다운로드

```bash
ollama serve          # 서버 시작 (별도 터미널)
ollama pull nomic-embed-text
ollama pull EXAONE-3.5:7.8b   # 또는 qwen2.5:14b
```

### 5. Phase 1 검증

```bash
source .venv/bin/activate
python scripts/test_phase1.py
```

## 프로젝트 구조

```
AutoReport/
├── src/
│   ├── config.py            # 설정값
│   ├── embeddings/          # nomic-embed-text 임베딩
│   ├── vectorstore/         # Milvus 연동
│   ├── llm/                 # Ollama LLM 클라이언트
│   ├── parsers/             # PPT/PDF/DOCX 파서 (Phase 2)
│   └── pipeline/            # RAG 파이프라인 (Phase 3)
├── scripts/
│   └── test_phase1.py       # Phase 1 검증
├── data/
│   ├── input/               # 입력 문서 (RFP, PPT)
│   └── output/              # 생성된 제안서
├── docker-compose.yml       # Milvus Standalone
└── requirements.txt
```

## 개발 단계

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | 환경 구성 (Ollama + Milvus 연동) | ✅ |
| 2 | 지식베이스 구축 (PPT/RFP 파서) | 🔜 |
| 3 | RAG 생성 파이프라인 | 🔜 |
| 4 | UI + DOCX 출력 | 🔜 |
| 5 | 검증 및 튜닝 | 🔜 |
