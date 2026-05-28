"""
Web Search Tool
================
Brave Search API를 우선 사용하고, API 키 미설정·오류·Rate Limit 발생 시
DuckDuckGo로 자동 전환(fallback)합니다.
"""

import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
from langchain_core.tools import Tool
import requests
import config


# ==================== Brave Search ====================
def _brave_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Brave Search API로 웹 검색을 시도합니다.

    Returns:
        검색 결과 리스트. API 키 없음·오류 시 빈 리스트를 반환하여 fallback을 유도합니다.
    """
    api_key = config.BRAVE_API_KEY
    if not api_key:
        return []

    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
        params = {
            "q": query,
            "count": max_results,
            "country": "US",
            "search_lang": "en"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"][:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "_engine": "Brave"
                })
        return results

    except (requests.exceptions.RequestException, ValueError):
        logging.debug("Brave Search 오류 → DuckDuckGo로 전환", exc_info=True)
        return []
    except Exception:
        logging.exception("Brave search error")
        return []


# ==================== DuckDuckGo Fallback ====================
def _duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    DuckDuckGo로 웹 검색을 수행합니다 (Brave 실패 시 fallback).

    Returns:
        검색 결과 리스트. 패키지 미설치 또는 오류 시 에러 딕셔너리를 담은 리스트 반환.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [{
            "error": "duckduckgo-search 패키지가 설치되지 않았습니다.",
            "suggestion": "pip install duckduckgo-search 를 실행하세요."
        }]

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", ""),
                "_engine": "DuckDuckGo"
            }
            for r in raw
        ]
    except Exception:
        logging.exception("DuckDuckGo search error")
        return [{
            "error": "DuckDuckGo 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "query": query
        }]


# ==================== 공개 인터페이스 ====================
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    웹 검색을 수행합니다.

    1순위: Brave Search API (BRAVE_API_KEY 환경변수 필요)
    2순위: DuckDuckGo (API 키 불필요, 자동 fallback)

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5개)

    Returns:
        검색 결과 리스트
    """
    results = _brave_search(query, max_results)
    if results:
        return results

    # Brave 실패(키 없음·오류·빈 결과) → DuckDuckGo fallback
    return _duckduckgo_search(query, max_results)


# ==================== LangChain Tool 래퍼 ====================
web_search_tool = Tool(
    name="web_search",
    description="""
    Brave Search(또는 DuckDuckGo fallback)로 일반 웹에서 최신 정보를 검색합니다.

    Input: 검색 키워드 (예: "copper alloy market trends 2024")
    Output: 웹 검색 결과 (제목, 링크, 요약)

    Use for: 최신 뉴스, 트렌드, 시장 동향, 일반 정보 검색
    """,
    func=lambda query: _format_results(web_search(query))
)


def _format_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과를 읽기 쉬운 형식으로 포맷팅합니다."""
    if not results:
        return "검색 결과가 없습니다."

    if "error" in results[0]:
        error = results[0]["error"]
        suggestion = results[0].get("suggestion", "")
        query = results[0].get("query", "")
        msg = f"오류: {error}"
        if suggestion:
            msg += f"\n{suggestion}"
        if query:
            msg += f"\n검색어: {query}"
        return msg

    engine = results[0].get("_engine", "Web")
    output = [f"=== {engine} 웹 검색 결과 ({len(results)}개) ===\n"]

    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result.get('title', '제목 없음')}")
        if result.get("link"):
            output.append(f"   링크: {result['link']}")
        if result.get("snippet"):
            snippet = result["snippet"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            output.append(f"   요약: {snippet}")
        output.append("")

    return "\n".join(output)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("Web Search Tool 테스트 (Brave → DuckDuckGo fallback)\n")

    print("1. 'copper alloy trends 2024' 검색:")
    print(web_search_tool.run("copper alloy trends 2024"))

    print("\n" + "="*60 + "\n")

    print("2. 'materials science news' 검색:")
    print(web_search_tool.run("materials science news"))
