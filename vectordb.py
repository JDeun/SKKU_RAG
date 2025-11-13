"""
VectorDB 생성 모듈
=================
PDF 문서를 로드하고, C-P-P(Composition-Process-Property)를 추출하여
메타데이터와 함께 VectorDB에 저장합니다.
"""

import os
import warnings

# 로그 억제 설정 (imports 전에 실행)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 메시지 억제

from pathlib import Path
from typing import List, Optional
import tiktoken
from tqdm import tqdm

# LangChain 관련 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# PDF 처리 라이브러리 (pymupdf 사용)
import fitz  # PyMuPDF

# 설정 및 프롬프트 임포트
import config
import prompts


# ==================== 토큰 계산 유틸리티 ====================
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text: str) -> int:
    """
    텍스트의 토큰 수를 계산합니다.
    
    Args:
        text: 토큰 수를 계산할 텍스트
        
    Returns:
        토큰 개수
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


# ==================== PDF 로드 ====================
def load_single_pdf(filepath: str, filename: str) -> List[Document]:
    """
    단일 PDF 파일을 로드하고 페이지별로 분할합니다.
    PyMuPDF(fitz)를 사용하여 더 빠르고 정확하게 텍스트를 추출합니다.
    
    Args:
        filepath: PDF 파일의 전체 경로
        filename: PDF 파일명 (메타데이터용)
        
    Returns:
        Document 객체 리스트 (각 페이지별로)
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(filepath):
            print(f"⚠️  파일이 존재하지 않습니다: {filepath}")
            return []
        
        # PyMuPDF로 PDF 열기
        doc = fitz.open(filepath)
        
        # 암호화 확인
        if doc.is_encrypted:
            print(f"🔒 암호화된 PDF 건너뜀: {filename}")
            doc.close()
            return []
        
        total_pages = len(doc)
        pages_with_metadata = []
        
        # 페이지별로 텍스트 추출
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")  # 텍스트 추출
            
            # 빈 페이지 스킵
            if not text.strip():
                continue
            
            pages_with_metadata.append(
                Document(
                    page_content=text,
                    metadata={
                        'source': filename,
                        'page': page_num + 1,  # 1부터 시작
                        'total_pages': total_pages
                    }
                )
            )
        
        doc.close()
        print(f"  ✓ {filename}: {len(pages_with_metadata)} 페이지 로드")
        return pages_with_metadata
        
    except Exception as e:
        print(f"❌ PDF 처리 오류: {filename} - {e}")
        return []


def load_pdfs(pdf_path: str) -> List[Document]:
    """
    PDF 파일 또는 폴더를 로드합니다.
    
    Args:
        pdf_path: PDF 파일 경로 또는 폴더 경로
        
    Returns:
        모든 페이지의 Document 리스트
    """
    pdf_path = Path(pdf_path)
    all_pages = []
    
    if not pdf_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {pdf_path}")
        return []
    
    # 폴더인 경우
    if pdf_path.is_dir():
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print("⚠️  PDF 파일이 없습니다.")
            return []
        
        print(f"📂 {len(pdf_files)}개의 PDF 파일 발견")
        for pdf_file in tqdm(pdf_files, desc="PDF 로드 중"):
            pages = load_single_pdf(str(pdf_file), pdf_file.name)
            all_pages.extend(pages)
    
    # 단일 파일인 경우
    elif pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
        print(f"📄 단일 PDF 파일 로드: {pdf_path.name}")
        pages = load_single_pdf(str(pdf_path), pdf_path.name)
        all_pages.extend(pages)
    
    else:
        print(f"❌ 유효한 PDF 파일 또는 폴더가 아닙니다: {pdf_path}")
    
    print(f"✅ 총 {len(all_pages)} 페이지 로드 완료\n")
    return all_pages


# ==================== 텍스트 분할 ====================
def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> List[Document]:
    """
    문서를 청크 단위로 분할합니다.
    
    Args:
        documents: 분할할 Document 리스트
        chunk_size: 청크 크기 (토큰 수)
        chunk_overlap: 청크 간 오버랩 크기
        
    Returns:
        분할된 청크 리스트
    """
    if not documents:
        print("⚠️  분할할 문서가 없습니다.")
        return []
    
    print(f"📄 문서 분할 중 (청크 크기: {chunk_size}, 오버랩: {chunk_overlap})...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # 중복 제거 (동일 내용 + 동일 출처 + 동일 페이지)
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        key = (
            chunk.page_content.strip(),
            chunk.metadata.get("source", ""),
            chunk.metadata.get("page", None)
        )
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    print(f"✅ {len(chunks)}개 청크 생성 → 중복 제거 후 {len(unique_chunks)}개\n")
    return unique_chunks


def add_cpp_to_chunks(
    chunks: List[Document],
    batch_size: int = 5
) -> List[Document]:
    """
    모든 청크에 C-P-P 메타데이터를 추가합니다.
    JSON Output Parser를 사용하여 안정적으로 데이터를 추출합니다.
    
    Args:
        chunks: 청크 리스트
        batch_size: 배치 크기 (한 번에 처리할 청크 수)
        
    Returns:
        C-P-P 메타데이터가 추가된 청크 리스트
    """
    if not chunks:
        return []
    
    print(f"🔬 C-P-P 추출 중 (총 {len(chunks)}개 청크)...")
    
    # LLM 및 추출 체인 초기화
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY
    )
    extraction_chain = prompts.CPP_EXTRACTION_PROMPT | llm | prompts.json_parser
    
    processed_chunks = []
    for i in tqdm(range(0, len(chunks)), desc="C-P-P 추출"):
        chunk = chunks[i]
        try:
            # 체인 실행
            cpp = extraction_chain.invoke({"text": chunk.page_content})
        except Exception as e:
            # 파싱 오류 발생 시 기본값 사용
            print(f"⚠️  C-P-P 추출/파싱 오류 (청크 {i}): {e}")
            cpp = {"composition": "N/A", "process": "N/A", "property": "N/A"}
            
        # 메타데이터에 C-P-P 추가
        chunk.metadata.update(cpp)
        processed_chunks.append(chunk)
    
    print(f"✅ C-P-P 추출 완료\n")
    return processed_chunks


# ==================== VectorDB 생성/로드 ====================
def create_or_load_vectordb(
    chunks: Optional[List[Document]] = None,
    persist_directory: str = str(config.VECTOR_DB_PATH),
    force_recreate: bool = False
) -> Chroma:
    """
    VectorDB를 생성하거나 기존 DB를 로드합니다.
    
    Args:
        chunks: 저장할 청크 리스트 (None이면 기존 DB 로드)
        persist_directory: DB 저장 경로
        force_recreate: True이면 기존 DB 삭제 후 재생성
        
    Returns:
        Chroma VectorDB 인스턴스
    """
    # Embedding 모델 초기화
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.EMBEDDING_DEVICE}
    )
    
    # 기존 DB가 있고 재생성이 아닌 경우
    if os.path.exists(persist_directory) and not force_recreate:
        print(f"📂 기존 VectorDB 로드: {persist_directory}")
        try:
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print(f"✅ VectorDB 로드 완료 (문서 수: {db._collection.count()})\n")
            return db
        except Exception as e:
            print(f"⚠️  기존 DB 로드 실패: {e}")
            print("   새로운 DB를 생성합니다.\n")
    
    # 새 DB 생성
    if chunks is None or len(chunks) == 0:
        print("❌ 저장할 청크가 없습니다.")
        return None
    
    print(f"💾 VectorDB 생성 중 ({len(chunks)}개 청크)...")
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        db.persist()
        print(f"✅ VectorDB 생성 완료: {persist_directory}\n")
        return db
    except Exception as e:
        print(f"❌ VectorDB 생성 실패: {e}")
        return None


# ==================== 전체 파이프라인 ====================
def build_vectordb_pipeline(
    pdf_path: str,
    extract_cpp: bool = True,
    force_recreate: bool = False
) -> Chroma:
    """
    PDF → 청크 → C-P-P 추출 → VectorDB 생성의 전체 파이프라인
    
    Args:
        pdf_path: PDF 파일 또는 폴더 경로
        extract_cpp: C-P-P를 추출할지 여부
        force_recreate: 기존 DB를 삭제하고 재생성할지
        
    Returns:
        VectorDB 인스턴스
    """
    print("="*60)
    print("VectorDB 구축 시작")
    print("="*60 + "\n")
    
    # 1. PDF 로드
    documents = load_pdfs(pdf_path)
    if not documents:
        print("❌ 로드된 문서가 없습니다.")
        return None
    
    # 2. 문서 분할
    chunks = split_documents(documents)
    if not chunks:
        print("❌ 생성된 청크가 없습니다.")
        return None
    
    # 3. C-P-P 추출 (옵션)
    if extract_cpp:
        chunks = add_cpp_to_chunks(chunks)
    
    # 4. VectorDB 생성
    db = create_or_load_vectordb(
        chunks=chunks,
        force_recreate=force_recreate
    )
    
    print("="*60)
    print("VectorDB 구축 완료")
    print("="*60 + "\n")
    
    return db


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    # 설정 출력
    config.print_config()
    
    # PDF 경로 입력
    pdf_path = input("PDF 파일 또는 폴더 경로를 입력하세요: ").strip().strip('"\'')
    
    # VectorDB 구축
    db = build_vectordb_pipeline(
        pdf_path=pdf_path,
        extract_cpp=True,
        force_recreate=False
    )
    
    if db:
        print("\n✅ VectorDB가 성공적으로 구축되었습니다!")
        
        # 테스트 검색
        test_query = "Cu alloy properties"
        print(f"\n🔍 테스트 검색: '{test_query}'")
        results = db.similarity_search(test_query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- 결과 {i} ---")
            print(f"출처: {doc.metadata.get('source')} (p.{doc.metadata.get('page')})")
            print(f"Composition: {doc.metadata.get('composition', 'N/A')}")
            print(f"Process: {doc.metadata.get('process', 'N/A')[:100]}...")
            print(f"Property: {doc.metadata.get('property', 'N/A')[:100]}...")
