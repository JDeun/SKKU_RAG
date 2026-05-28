"""
AgenticRAG 설정 파일
===================
LLM, Embedding 모델, API 키, 하이퍼파라미터 등 모든 설정을 관리합니다.
"""

import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv

# ==================== 로깅 및 경고 억제 ====================
# 불필요한 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tf_keras")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")

# TensorFlow/Abseil 로깅 억제
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# LangChain 관련 경고 억제
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)

# 환경변수 로드 (.env 파일이 있으면 자동 로드)
load_dotenv()


# ==================== 경로 설정 ====================
# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent
# VectorDB 저장 경로
VECTOR_DB_PATH = PROJECT_ROOT / "chroma_db"
# PDF 파일 경로 (기본값, 사용자가 변경 가능)
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "pdfs"


# ==================== API 키 설정 ====================
# Google Gemini API 키
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다. .env 파일에 추가하세요.")

# Materials Project API 키
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")
if not MATERIALS_PROJECT_API_KEY:
    logging.warning("MATERIALS_PROJECT_API_KEY 환경변수가 설정되지 않았습니다. https://next-gen.materialsproject.org/api 에서 발급받으세요.")

# Groq API 키 (선택 - Gemini 사용량 초과·오류 시 fallback으로 자동 사용)
# https://console.groq.com/ 에서 발급 (무료)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Crossref는 API 키 불필요 (이메일 권장)
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO") or None  # 실제 이메일 설정 시 Crossref polite pool 사용 가능

# Brave Search API 키 (선택, 없으면 DuckDuckGo로 자동 전환)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


# ==================== LLM 설정 ====================
# Gemini 모델 설정
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0  # 0에 가까울수록 결정론적, 1에 가까울수록 창의적
LLM_MAX_OUTPUT_TOKENS = 2048  # 최대 출력 토큰 수
LLM_STREAMING = False  # ReAct agent에서 streaming은 파싱 오류 유발 — False 유지

# Groq fallback 모델 설정 (Gemini 실패 시 자동 사용)
# 무료 티어: llama-3.3-70b-versatile 권장
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")


# ==================== Embedding 모델 설정 ====================
# Ollama 임베딩 모델 (로컬 실행, API 키 불필요)
# 사전 준비: ollama pull qwen3-embedding:latest
EMBEDDING_MODEL_NAME = "qwen3-embedding:latest"  # 변경 시 chroma_db 재생성 필요
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ==================== VectorDB 설정 ====================
# 텍스트 분할 (Chunking) 파라미터
CHUNK_SIZE = 800  # 청크 크기 (토큰 단위)
CHUNK_OVERLAP = 100  # 청크 간 오버랩 크기
# VectorDB 검색 파라미터
RETRIEVAL_TOP_K = 10  # 검색 시 반환할 상위 문서 수


# ==================== Agent 설정 ====================
# ReAct Agent의 최대 반복 횟수 (무한 루프 방지)
AGENT_MAX_ITERATIONS = 10
# Agent 타임아웃 (초 단위)
AGENT_TIMEOUT = 120


# ==================== Tool 설정 ====================
# Materials Project API 타임아웃
MP_API_TIMEOUT = 30  # 초
# Crossref API 타임아웃
CROSSREF_API_TIMEOUT = 30  # 초


# ==================== 로깅 설정 ====================
# 로그 레벨: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"
# 로그 출력 여부
VERBOSE = True  # True 시 상세 로그 출력


# ==================== 설정 확인 함수 ====================
def validate_config():
    """
    필수 설정값들이 올바르게 설정되었는지 확인합니다.
    """
    errors = []
    
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY가 설정되지 않았습니다.")
    
    if not VECTOR_DB_PATH.parent.exists():
        errors.append(f"VectorDB 상위 디렉토리가 존재하지 않습니다: {VECTOR_DB_PATH.parent}")
    
    if errors:
        for error in errors:
            logging.error("설정 오류: %s", error)
        return False

    logging.info("설정 검증 완료")
    return True


def print_config():
    """
    현재 설정값을 출력합니다.
    """
    print("\n" + "="*50)
    print("AgenticRAG 설정")
    print("="*50)
    print(f"📁 프로젝트 루트: {PROJECT_ROOT}")
    print(f"📁 VectorDB 경로: {VECTOR_DB_PATH}")
    print(f"🤖 LLM 모델: {LLM_MODEL_NAME}")
    print(f"🌡️  Temperature: {LLM_TEMPERATURE}")
    print(f"🔢 Embedding 모델: {EMBEDDING_MODEL_NAME}")
    print(f"🖥️  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"📏 청크 크기: {CHUNK_SIZE}")
    print(f"📊 검색 Top-K: {RETRIEVAL_TOP_K}")
    print(f"🔑 Materials Project API: {'설정됨' if MATERIALS_PROJECT_API_KEY else '미설정'}")
    print(f"🔑 Groq API (fallback): {'설정됨 → ' + GROQ_MODEL_NAME if GROQ_API_KEY else '미설정 (Gemini만 사용)'}")
    print(f"📧 Crossref mailto: {CROSSREF_MAILTO}")
    print("="*50 + "\n")


# 모듈 임포트 시 자동으로 설정 검증
if __name__ == "__main__":
    print_config()
    validate_config()
