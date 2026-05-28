"""
arXiv Search Tool
==================
arXiv API를 사용하여 학술 프리프린트 논문을 검색하는 도구입니다.
API 키가 필요하지 않습니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import arxiv

import config


def search_arxiv(
    query: str,
    max_results: int = None,
    category_filter: str = None
) -> List[Dict[str, Any]]:
    """
    arXiv에서 논문을 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본값: config.ARXIV_MAX_RESULTS)
        category_filter: arXiv 카테고리 필터 (예: "cond-mat.mtrl-sci")

    Returns:
        검색 결과 리스트
    """
    if max_results is None:
        max_results = config.ARXIV_MAX_RESULTS

    try:
        # 카테고리 필터 적용
        search_query = query
        if category_filter:
            search_query = f"cat:{category_filter} AND all:{query}"

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": ", ".join([a.name for a in paper.authors[:5]]),
                "abstract": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "arxiv_id": paper.entry_id.split("/")[-1],
                "pdf_url": paper.pdf_url,
                "categories": paper.categories
            })

        return results

    except Exception as e:
        return [{"error": f"arXiv 검색 오류: {str(e)}", "query": query}]


def _format_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과를 읽기 쉬운 형식으로 포맷팅합니다."""
    if not results:
        return "arXiv 검색 결과가 없습니다."

    if "error" in results[0]:
        return f"오류: {results[0]['error']}\n검색어: {results[0].get('query', 'N/A')}"

    output = [f"=== arXiv 검색 결과 ({len(results)}건) ===\n"]

    for i, paper in enumerate(results, 1):
        output.append(f"{i}. {paper['title']}")
        output.append(f"   저자: {paper['authors']}")
        output.append(f"   발행일: {paper['published']}")
        output.append(f"   arXiv ID: {paper['arxiv_id']}")
        output.append(f"   PDF: {paper['pdf_url']}")
        output.append(f"   카테고리: {', '.join(paper.get('categories', []))}")
        abstract = paper['abstract'][:300] + "..." if len(paper['abstract']) > 300 else paper['abstract']
        output.append(f"   초록: {abstract}")
        output.append("")

    return "\n".join(output)


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

def _parse_arxiv_input(input_str: str):
    """
    Agent 입력을 파싱합니다.
    - "query" → query만 사용
    - "query | category" → query + category_filter 사용
    """
    if "|" in input_str:
        parts = input_str.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return input_str.strip(), None


arxiv_search_tool = Tool(
    name="arxiv_search",
    description="""
    Search preprint papers from arXiv. No API key required.
    Useful for finding latest research trends, theoretical studies, and simulation results.

    Input: search keywords (e.g., "copper interconnect electromigration")
           Optionally add category filter with pipe: "query | cond-mat.mtrl-sci"
    Output: paper title, authors, abstract, published date, PDF link

    Use for: latest preprints, research trends, theoretical background
    """,
    func=lambda query: _format_results(
        search_arxiv(*_parse_arxiv_input(query))
    )
)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("arXiv Search Tool 테스트\n")

    print("1. 'copper interconnect electromigration' 검색:")
    print(arxiv_search_tool.run("copper interconnect electromigration"))

    print("\n" + "=" * 60 + "\n")

    print("2. 재료과학 카테고리 필터 검색:")
    results = search_arxiv("band gap prediction", category_filter="cond-mat.mtrl-sci")
    print(_format_results(results))
