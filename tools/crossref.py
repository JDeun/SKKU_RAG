"""
Crossref Tool
=============
Crossref API를 사용하여 학술 논문 메타데이터를 검색하는 도구입니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, List, Dict, Any
from habanero import Crossref
import config


def search_crossref(
    query: str,
    rows: int = 5,
    sort: str = "relevance",
    order: str = "desc"
) -> List[Dict[str, Any]]:
    """
    Crossref에서 학술 논문을 검색합니다.
    
    Args:
        query: 검색 키워드
        rows: 반환할 결과 수 (기본 5개)
        sort: 정렬 기준 (published, relevance, etc.)
        order: 정렬 순서 (desc, asc)
        
    Returns:
        논문 메타데이터 리스트
        [
            {
                "title": str,
                "authors": str,
                "doi": str,
                "year": int,
                "journal": str,
                "abstract": str (선택)
            }
        ]
    """
    try:
        # Crossref 클라이언트 초기화
        cr = Crossref(mailto=config.CROSSREF_MAILTO)
        
        # 검색 실행
        results = cr.works(
            query=query,
            limit=rows,
            sort=sort,
            order=order
        )
        
        if not results or "message" not in results:
            return []
        
        items = results["message"].get("items", [])
        
        # 결과 파싱
        papers = []
        for item in items:
            # 제목 추출
            title = item.get("title", ["No title"])[0] if item.get("title") else "No title"
            
            # 저자 추출
            authors = []
            if "author" in item:
                for author in item["author"][:3]:  # 최대 3명
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
            
            if len(item.get("author", [])) > 3:
                authors.append("et al.")
            
            authors_str = ", ".join(authors) if authors else "Unknown authors"
            
            # DOI
            doi = item.get("DOI", "No DOI")
            
            # 발행 연도
            published = item.get("published", {})
            if "date-parts" in published and published["date-parts"]:
                year = published["date-parts"][0][0]
            else:
                year = "Unknown year"
            
            # 저널명
            journal = item.get("container-title", ["Unknown journal"])
            journal_str = journal[0] if isinstance(journal, list) and journal else "Unknown journal"
            
            # 초록 (있는 경우)
            abstract = item.get("abstract", "No abstract available")
            
            papers.append({
                "title": title,
                "authors": authors_str,
                "doi": doi,
                "year": year,
                "journal": journal_str,
                "abstract": abstract[:300] + "..." if len(abstract) > 300 else abstract
            })
        
        return papers
        
    except Exception as e:
        return [{
            "error": f"Crossref API 오류: {str(e)}",
            "query": query
        }]


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

crossref_tool = Tool(
    name="crossref_search",
    description="""
    Searches academic papers from Crossref database.
    
    IMPORTANT: This is an English-language scholarly database. Non-English queries may not return relevant results.
    
    Input: Search keywords (e.g., "copper alloy electromigration", "titanium aerospace materials 2024")
    Output: Paper title, authors, DOI, year, journal name, abstract
    
    Use for: Recent publications, literature review, citation needs
    """,
    func=lambda query: _format_results(search_crossref(query))
)


def _format_results(papers: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 읽기 쉬운 형식으로 포맷팅합니다.
    
    Args:
        papers: 논문 메타데이터 리스트
        
    Returns:
        포맷팅된 결과 문자열
    """
    if not papers:
        return "검색 결과가 없습니다."
    
    # 에러 처리
    if "error" in papers[0]:
        return f"오류: {papers[0]['error']}\n검색어: {papers[0].get('query', 'N/A')}"
    
    # 결과 포맷팅
    output = [f"=== {len(papers)}개의 논문 검색 결과 ===\n"]
    
    for i, paper in enumerate(papers, 1):
        output.append(f"{i}. {paper['title']}")
        output.append(f"   저자: {paper['authors']}")
        output.append(f"   저널: {paper['journal']} ({paper['year']})")
        output.append(f"   DOI: {paper['doi']}")
        
        # 초록이 있고 "No abstract"가 아닌 경우만 출력
        if paper['abstract'] and not paper['abstract'].startswith("No abstract"):
            output.append(f"   초록: {paper['abstract']}")
        
        output.append("")  # 빈 줄
    
    return "\n".join(output)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("Crossref Tool 테스트\n")
    
    # 테스트 1: 기본 검색
    print("1. 'copper electromigration' 검색:")
    print(crossref_tool.run("copper electromigration"))
    
    print("\n" + "="*60 + "\n")
    
    # 테스트 2: 최근 연구
    print("2. 'semiconductor interconnect 2024' 검색:")
    print(crossref_tool.run("semiconductor interconnect 2024"))
