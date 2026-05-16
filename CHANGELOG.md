# Changelog

AgenticRAG 프로젝트의 변경 이력입니다.

## [2.0.0] - 2025-05-16

### Added
- **arXiv 검색 도구** (`tools/arxiv_search.py`): arXiv API 기반 프리프린트 논문 검색 (API 키 불필요)
- **OQMD 검색 도구** (`tools/oqmd_search.py`): Open Quantum Materials Database REST API 기반 DFT 데이터 검색 (API 키 불필요)
- **웹검색 LLM 요약**: 검색 결과를 Agent에 전달하기 전 LLM으로 요약 (`WEB_SEARCH_SUMMARIZE` 설정으로 on/off)
- **멀티턴 대화 메모리**: `ConversationBufferWindowMemory`로 최근 N턴 대화 유지 (`MEMORY_WINDOW_SIZE` 설정)
- `requirements.txt`에 `arxiv>=2.1.0`, `requests>=2.25.0` 추가

### Changed
- API 키 관리를 `config.py`로 중앙화 (각 파일에서 직접 `os.getenv()` 호출하지 않음)
- ReAct 프롬프트에 `{chat_history}` 변수 추가 및 도구 설명 업데이트 (`prompts.py`)
- Agent 도구 목록 4개 → 6개 확장 (`agent.py`)
- `vectordb.py` import 경로를 최신 LangChain 패키지로 업데이트

### Fixed
- 비대화형 환경(Streamlit, Jupyter)에서 `input()` 호출 시 에러 → `sys.stdin.isatty()` 가드 추가
- Pydantic v1/v2 호환성 문제 → `try/except` import 분기 (`prompts.py`)
- TensorFlow, LangChain 등 불필요한 경고 메시지 억제 (`config.py`)
- Embedding 모델을 HuggingFace에서 Ollama 기반으로 통일

## [1.0.0] - 2025-05-15

### Added
- 초기 프로젝트 구조
- VectorDB 검색 도구 (`tools/vectordb_search.py`)
- Materials Project 검색 도구 (`tools/materials_project.py`)
- Crossref 논문 검색 도구 (`tools/crossref.py`)
- 웹 검색 도구 (`tools/web_search.py`) — Brave Search API
- ReAct Agent (`agent.py`)
- Streamlit UI (`app.py`)
- PDF → C-P-P 추출 → VectorDB 구축 파이프라인 (`vectordb.py`)
