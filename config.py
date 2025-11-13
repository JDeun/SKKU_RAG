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

# sentence-transformers 경고 억제 (모델 생성 시)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# LangChain 관련 경고 억제
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)

# 기타 라이브러리 로깅 억제
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

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
    print("⚠️  경고: GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    print("   .env 파일에 추가하거나 직접 입력하세요.")
    GOOGLE_API_KEY = input("Google API 키를 입력하세요: ").strip()

# Materials Project API 키
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")
if not MATERIALS_PROJECT_API_KEY:
    print("⚠️  경고: MATERIALS_PROJECT_API_KEY 환경변수가 설정되지 않았습니다.")
    print("   https://next-gen.materialsproject.org/api 에서 발급받으세요.")
    MATERIALS_PROJECT_API_KEY = input("Materials Project API 키를 입력하세요 (선택, 엔터 시 스킵): ").strip() or None

# Crossref는 API 키 불필요 (이메일 권장)
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "your.email@example.com")


# ==================== LLM 설정 ====================
# Gemini 모델 설정
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.0  # 0에 가까울수록 결정론적, 1에 가까울수록 창의적
LLM_MAX_OUTPUT_TOKENS = 2048  # 최대 출력 토큰 수
LLM_STREAMING = True  # 스트리밍 출력 여부


# ==================== Embedding 모델 설정 ====================
# HuggingFace Embedding 모델 (로컬 실행, API 키 불필요)
# sentence-transformers 호환 모델 사용
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DEVICE = "cpu"  # "cuda" 사용 시 GPU 가속


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
        print("\n❌ 설정 오류:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ 설정 검증 완료")
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
    print(f"🖥️  디바이스: {EMBEDDING_DEVICE}")
    print(f"📏 청크 크기: {CHUNK_SIZE}")
    print(f"📊 검색 Top-K: {RETRIEVAL_TOP_K}")
    print(f"🔑 Materials Project API: {'설정됨' if MATERIALS_PROJECT_API_KEY else '미설정'}")
    print(f"📧 Crossref mailto: {CROSSREF_MAILTO}")
    print("="*50 + "\n")


# 모듈 임포트 시 자동으로 설정 검증
if __name__ == "__main__":
    print_config()
    validate_config()
