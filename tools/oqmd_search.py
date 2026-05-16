"""
OQMD Search Tool
==================
Open Quantum Materials Database (OQMD) REST API를 사용하여
DFT 계산 기반 재료 데이터를 검색하는 도구입니다.
API 키가 필요하지 않습니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import requests

import config


def search_oqmd(
    query: str,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    OQMD에서 재료 데이터를 검색합니다.

    Args:
        query: 화학식 또는 원소 조합 (예: "Cu2O", "Cu-Mg")
        limit: 최대 결과 수

    Returns:
        검색 결과 리스트
    """
    if limit is None:
        limit = config.OQMD_MAX_RESULTS

    try:
        # 화학식 기반 검색
        # OQMD API는 composition 또는 generic 파라미터 지원
        base_url = config.OQMD_API_BASE_URL
        url = f"{base_url}/formationenergy"

        # 하이픈이 있으면 generic (원소 조합), 없으면 composition (정확한 화학식)
        if "-" in query:
            params = {
                "generic": query,
                "limit": limit,
                "format": "json"
            }
        else:
            params = {
                "composition": query,
                "limit": limit,
                "format": "json"
            }

        response = requests.get(
            url,
            params=params,
            timeout=config.OQMD_API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        results = []
        entries = data.get("data", [])

        for entry in entries[:limit]:
            result = {
                "composition": entry.get("composition", "N/A"),
                "formula": entry.get("name", "N/A"),
                "spacegroup": entry.get("spacegroup", "N/A"),
                "formation_energy": entry.get("delta_e", "N/A"),
                "stability": entry.get("stability", "N/A"),
                "band_gap": entry.get("band_gap", "N/A"),
                "volume": entry.get("volume", "N/A"),
                "ntypes": entry.get("ntypes", "N/A"),
                "natoms": entry.get("natoms", "N/A"),
                "entry_id": entry.get("entry_id", "N/A"),
            }
            results.append(result)

        return results

    except requests.exceptions.Timeout:
        return [{"error": "OQMD API 타임아웃", "query": query}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"OQMD API 오류: {str(e)}", "query": query}]
    except Exception as e:
        return [{"error": f"OQMD 검색 오류: {str(e)}", "query": query}]


def _format_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과를 읽기 쉬운 형식으로 포맷팅합니다."""
    if not results:
        return "OQMD 검색 결과가 없습니다."

    if "error" in results[0]:
        return f"오류: {results[0]['error']}\n검색어: {results[0].get('query', 'N/A')}"

    output = [f"=== OQMD 검색 결과 ({len(results)}건) ===\n"]

    for i, entry in enumerate(results, 1):
        output.append(f"{i}. {entry['composition']} ({entry['formula']})")
        output.append(f"   공간군: {entry['spacegroup']}")
        output.append(f"   Formation Energy: {entry['formation_energy']} eV/atom")
        output.append(f"   Stability: {entry['stability']} eV/atom")
        if entry['band_gap'] and entry['band_gap'] != "N/A":
            output.append(f"   Band Gap: {entry['band_gap']} eV")
        output.append(f"   Volume: {entry['volume']} A^3")
        output.append(f"   원자 수: {entry['natoms']}, 원소 수: {entry['ntypes']}")
        output.append(f"   Entry ID: {entry['entry_id']}")
        output.append("")

    return "\n".join(output)


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

oqmd_search_tool = Tool(
    name="oqmd_search",
    description="""
    OQMD(Open Quantum Materials Database)에서 DFT 계산 기반 재료 데이터를 검색합니다.
    API 키가 필요하지 않습니다. Materials Project의 보완 데이터소스로 활용됩니다.

    Input: 화학식 또는 원소 조합 (예: "Cu2O", "Fe-Ni", "CuMg")
    Output: Formation energy, band gap, stability, 공간군 등 DFT 계산 결과

    Use for: DFT 계산 데이터 cross-reference, 안정성 비교, 밴드갭 확인
    """,
    func=lambda query: _format_results(search_oqmd(query))
)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("OQMD Search Tool 테스트\n")

    print("1. 'Cu2O' 검색 (정확한 화학식):")
    print(oqmd_search_tool.run("Cu2O"))

    print("\n" + "=" * 60 + "\n")

    print("2. 'Cu-Mg' 검색 (원소 조합):")
    print(oqmd_search_tool.run("Cu-Mg"))
