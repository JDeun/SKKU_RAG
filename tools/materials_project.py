"""
Materials Project Tool
=====================
Materials Project API를 사용하여 계산 재료 데이터를 검색하는 도구입니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any
from mp_api.client import MPRester
import config


def search_materials_project(
    formula: str,
    property_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Materials Project에서 재료 데이터를 검색합니다.
    
    Args:
        formula: 화학식 (예: "Cu2O", "CuMg", "Si")
        property_name: 검색할 속성 (선택, 예: "band_gap", "formation_energy")
        
    Returns:
        재료 데이터 딕셔너리
        {
            "material_id": str,
            "formula": str,
            "band_gap": float,
            "formation_energy_per_atom": float,
            "crystal_system": str,
            "space_group": str,
            "density": float,
            "volume": float,
            "nsites": int,
            "elements": list
        }
    """
    if not config.MATERIALS_PROJECT_API_KEY:
        return {
            "error": "Materials Project API 키가 설정되지 않았습니다.",
            "info": "config.py에서 MATERIALS_PROJECT_API_KEY를 설정하세요."
        }
    
    try:
        with MPRester(config.MATERIALS_PROJECT_API_KEY) as mpr:
            # formula로 검색
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_per_atom",
                    "symmetry",
                    "density",
                    "volume",
                    "nsites",
                    "elements",
                    "is_stable"
                ]
            )
            
            if not docs:
                return {
                    "error": f"'{formula}'에 대한 데이터를 찾을 수 없습니다.",
                    "suggestion": "화학식을 확인하거나 다른 조성을 시도하세요."
                }
            
            # 가장 안정한 구조 선택 (is_stable=True, formation_energy_per_atom이 가장 낮은 순)
            stable_docs = sorted(
                docs,
                key=lambda d: (not d.is_stable, d.formation_energy_per_atom or float('inf'))
            )
            doc = stable_docs[0]
            
            # 결과 포맷팅
            result = {
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "energy_per_atom": doc.energy_per_atom,
                "is_stable": doc.is_stable,
                "crystal_system": doc.symmetry.crystal_system.value if doc.symmetry else "N/A",
                "space_group": doc.symmetry.symbol if doc.symmetry else "N/A",
                "density": doc.density,
                "volume": doc.volume,
                "nsites": doc.nsites,
                "elements": [str(e) for e in doc.elements]
            }
            
            # 특정 속성만 요청한 경우
            if property_name:
                if property_name in result:
                    return {
                        "formula": result["formula"],
                        property_name: result[property_name],
                        "material_id": result["material_id"]
                    }
                else:
                    result["warning"] = f"'{property_name}' 속성을 찾을 수 없습니다."
            
            return result
            
    except Exception as e:
        return {
            "error": f"Materials Project API 오류: {str(e)}",
            "formula": formula
        }


# ==================== LangChain Tool 래퍼 ====================
from langchain.tools import Tool

materials_project_tool = Tool(
    name="materials_project",
    description="""
    Searches DFT calculation data from Materials Project database.
    
    Input: Chemical formula (e.g., "Cu2O", "TiAl", "Si") or "Formula property:property_name" (e.g., "CuMg property:band_gap")
    
    Output: Band gap, formation energy, crystal structure, density, stability, space group
    
    Use for: Theoretical properties, computational data, stability analysis, electronic structure
    """,
    func=lambda query: _parse_and_search(query)
)


def _parse_and_search(query: str) -> str:
    """
    쿼리를 파싱하여 Materials Project 검색을 수행합니다.
    
    Args:
        query: "Cu2O" 또는 "Cu2O property:band_gap"
        
    Returns:
        검색 결과 문자열
    """
    parts = query.strip().split()
    formula = parts[0]
    
    # property: 구문 파싱
    property_name = None
    for part in parts[1:]:
        if part.startswith("property:"):
            property_name = part.split(":", 1)[1]
            break
    
    # 검색 실행
    result = search_materials_project(formula, property_name)
    
    # 에러 처리
    if "error" in result:
        return f"오류: {result['error']}\n{result.get('info', '')}{result.get('suggestion', '')}"
    
    # 결과 포맷팅
    if property_name and property_name in result:
        return (
            f"Material: {result['formula']} (ID: {result['material_id']})\n"
            f"{property_name}: {result[property_name]}"
        )
    
    # 전체 정보 반환
    output = [
        f"=== {result['formula']} (ID: {result['material_id']}) ===",
        f"Band Gap: {result['band_gap']} eV",
        f"Formation Energy: {result['formation_energy_per_atom']} eV/atom",
        f"Stable: {'Yes' if result['is_stable'] else 'No'}",
        f"Crystal System: {result['crystal_system']}",
        f"Space Group: {result['space_group']}",
        f"Density: {result['density']:.2f} g/cm³",
        f"Elements: {', '.join(result['elements'])}"
    ]
    
    if "warning" in result:
        output.append(f"\n경고: {result['warning']}")
    
    return "\n".join(output)


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    print("Materials Project Tool 테스트\n")
    
    # 테스트 1: 기본 검색
    print("1. Cu2O 검색:")
    print(materials_project_tool.run("Cu2O"))
    
    print("\n" + "="*60 + "\n")
    
    # 테스트 2: 특정 속성 검색
    print("2. Si의 band_gap 검색:")
    print(materials_project_tool.run("Si property:band_gap"))
    
    print("\n" + "="*60 + "\n")
    
    # 테스트 3: CuMg 검색
    print("3. CuMg 검색:")
    print(materials_project_tool.run("CuMg"))
