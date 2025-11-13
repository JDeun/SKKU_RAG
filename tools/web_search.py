"""
Web Search Tool
================
Brave Search API를 사용하여 일반 웹에서 정보를 검색하는 도구입니다.
DuckDuckGo보다 더 안정적이고 rate limit이 관대합니다.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import requests


def web_search(
    query: str,
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Brave Search API를 사용하여 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5개)

    Returns:
        검색 결과 리스트
    """
    # API 키 확인
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return [{
            "error": "BRAVE_API_KEY 환경변수가 설정되지 않았습니다.",
            "suggestion": "https://api.search.brave.com/ 에서 API 키를 발급받으세요."
        }]

    try:
        # Brave Search API 호출
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

        # 결과 파싱
        results = []
        if "web" in data and "results" in data["web"]:
            for result in data["web"]["results"][:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "link": result.get("url", ""),
                    "snippet": result.get("description", "")
                })

        return results

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "429" in error_msg:
            return [{
                "error": "Brave Search API rate limit 초과",
                "query": query,
                "suggestion": "잠시 후 다시 시도하거나 API 플랜을 확인하세요."
            }]
        elif "401" in error_msg:
            return [{
                "error": "Brave Search API 키가 유효하지 않습니다.",
                "suggestion": "API 키를 다시 확인하세요."
            }]
        else:
            return [{
                "error": f"Brave Search API 오류: {error_msg}",
                "query": query
            }]
    except Exception as e:
        return [{
            "error": f"웹 검색 오류: {str(e)}",
            "query": query
        }]


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

web_search_tool = Tool(
    name="web_search",
    description="""
    Brave Search API를 사용하여 일반 웹에서 최신 정보를 검색합니다.
    DuckDuckGo보다 더 안정적이고 rate limit이 관대합니다.

    Input: 검색 키워드 (예: "copper alloy market trends 2024")
    Output: 웹 검색 결과 (제목, 링크, 요약)

    Use for: 최신 뉴스, 트렌드, 시장 동향, 일반 정보 검색
    """,
    func=lambda query: _format_results(web_search(query))
)


def _format_results(results: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 읽기 쉬운 형식으로 포맷팅합니다.
    """
    if not results:
        return "검색 결과가 없습니다."

    # 에러 처리
    if "error" in results[0]:
        return f"오류: {results[0]['error']}\n검색어: {results[0].get('query', 'N/A')}"

    # 결과 포맷팅
    output = [f"=== Brave 웹 검색 결과 ({len(results)}개) ===\n"]

    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result.get('title', '제목 없음')}")
        if 'link' in result:
            output.append(f"   링크: {result['link']}")
        if 'snippet' in result:
            snippet = result['snippet'][:200] + "..." if len(result['snippet']) > 200 else result['snippet']
            output.append(f"   요약: {snippet}")
        output.append("")  # 빈 줄

    return "\n".join(output)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("Brave Web Search Tool 테스트\n")

    # API 키 확인
    if not os.getenv("BRAVE_API_KEY"):
        print("⚠️  BRAVE_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 BRAVE_API_KEY=your_key_here 추가하세요.")
        exit(1)

    # 테스트 검색
    print("1. 'copper alloy trends 2024' 검색:")
    print(web_search_tool.run("copper alloy trends 2024"))

    print("\n" + "="*60 + "\n")

    print("2. 'materials science news' 검색:")
    print(web_search_tool.run("materials science news"))
