"""
VectorDB Search Tool
====================
VectorDB에서 C-P-P 메타데이터를 포함한 문서를 검색하는 도구입니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import config
from vectordb import create_or_load_vectordb


# 전역 VectorDB 인스턴스 (lazy loading)
_vectordb = None


def get_vectordb():
    """
    VectorDB 인스턴스를 가져옵니다 (싱글톤 패턴).
    """
    global _vectordb
    if _vectordb is None:
        _vectordb = create_or_load_vectordb()
        if _vectordb is None:
            raise RuntimeError("VectorDB를 로드할 수 없습니다. vectordb.py로 먼저 DB를 생성하세요.")
    return _vectordb


def search_vectordb(
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K
) -> List[Dict[str, Any]]:
    """
    VectorDB에서 쿼리와 유사한 문서를 검색합니다.
    
    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수
        
    Returns:
        문서 리스트 (C-P-P 메타데이터 포함)
        [
            {
                "content": str,
                "source": str,
                "page": int,
                "composition": str,
                "process": str,
                "property": str
            }
        ]
    """
    try:
        db = get_vectordb()
        
        # 유사도 검색
        results = db.similarity_search(query, k=top_k)
        
        if not results:
            return []
        
        # 결과 포맷팅
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "composition": doc.metadata.get("composition", "N/A"),
                "process": doc.metadata.get("process", "N/A"),
                "property": doc.metadata.get("property", "N/A")
            })
        
        return formatted_results
        
    except Exception as e:
        return [{
            "error": f"VectorDB 검색 오류: {str(e)}",
            "query": query
        }]


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

vectordb_search_tool = Tool(
    name="vectordb_search",
    description="""
    Searches C-P-P (Composition-Process-Property) data from research papers stored in VectorDB.

    Input: search query (e.g., "Cu-Mg alloy resistivity", "electromigration properties")
    Output: relevant document chunks with C-P-P metadata

    Use for: experimental data, manufacturing processes, material properties from papers
    """,
    func=lambda query: _format_results(search_vectordb(query))
)


def _format_results(results: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 읽기 쉬운 형식으로 포맷팅합니다.
    
    Args:
        results: 문서 리스트
        
    Returns:
        포맷팅된 결과 문자열
    """
    if not results:
        return "검색 결과가 없습니다."
    
    # 에러 처리
    if "error" in results[0]:
        return f"오류: {results[0]['error']}\n검색어: {results[0].get('query', 'N/A')}"
    
    # 결과 포맷팅
    output = [f"=== {len(results)}개의 관련 문서 검색 ===\n"]
    
    for i, doc in enumerate(results, 1):
        output.append(f"[{i}] {doc['source']} (p.{doc['page']})")
        output.append(f"  📌 Composition: {doc['composition']}")
        output.append(f"  🔧 Process: {doc['process'][:200]}..." if len(doc['process']) > 200 else f"  🔧 Process: {doc['process']}")
        output.append(f"  📊 Property: {doc['property'][:200]}..." if len(doc['property']) > 200 else f"  📊 Property: {doc['property']}")
        output.append(f"  📄 Content: {doc['content']}")
        output.append("")  # 빈 줄
    
    return "\n".join(output)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("VectorDB Search Tool 테스트\n")
    
    # VectorDB 존재 확인
    try:
        db = get_vectordb()
        print(f"✅ VectorDB 로드 성공 (문서 수: {db._collection.count()})\n")
    except Exception as e:
        print(f"❌ VectorDB 로드 실패: {e}")
        print("vectordb.py를 먼저 실행하여 DB를 생성하세요.\n")
        exit(1)
    
    # 테스트 검색
    print("1. 'Cu-Mg alloy resistivity' 검색:")
    print(vectordb_search_tool.run("Cu-Mg alloy resistivity"))
    
    print("\n" + "="*60 + "\n")
    
    print("2. 'electromigration' 검색:")
    print(vectordb_search_tool.run("electromigration"))
