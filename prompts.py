"""
프롬프트 관리 모듈
=================
C-P-P 추출, RAG, Agent에 사용되는 모든 프롬프트를 관리합니다.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# ==================== Few-shot 예제 (참고용) ====================
FEW_SHOT_EXAMPLES = [
    {
        "composition": "Cu",
        "process": "Fabrication of Cu damascene interconnects followed by electromigration (EM) testing; variation in line width, thickness, and grain structure (bamboo, polycrystalline, etc.)",
        "property": "EM lifetime: bamboo > polycrystalline; bamboo structures have fewer grain boundaries, reducing atomic diffusion paths"
    },
    {
        "composition": "Cu, Ti",
        "process": "Use of Cu(2.5%Ti) alloy as a seed layer; annealing at 100–250℃ before CMP to promote bamboo grain structure",
        "property": "EM lifetime: more than 5× improvement with Ti doping; Ti stabilizes grain boundaries; resistivity increases by ~17%"
    },
    {
        "composition": "Cu, H₂ plasma-treated interface",
        "process": "H₂-based plasma pre-treatment on Cu surface before dielectric deposition",
        "property": "Over 10× increase in EM lifetime; silicide-like barrier forms at the interface; resistivity roughly doubles"
    },
    {
        "composition": "Cu, TaN/Ta liner, a-SiCxNyHz cap, W via",
        "process": "Standard damascene process with embedded via structure; increases contact area and improves interface reliability",
        "property": "Improved EM reliability; embedded vias maintain current flow even with voids; better than flush via configuration"
    },
    {
        "composition": "Cu (various linewidths)",
        "process": "Fabrication of interconnects with 0.09–1.9 μm width; comparison of samples with and without pre-CMP annealing",
        "property": "EM lifetime improves with annealing due to bamboo grain formation; bamboo structure is harder to form in narrow lines"
    },
    {
        "composition": "Cu, Al",
        "process": "Deposition using PVD; subsequent high temperature processes run at 350 to 400℃",
        "property": "Resistivity : Cu 2.5μΩ·cm, CuAl 4.5μΩ·cm",
    },
    {
        "composition": "Cu, Sn",
        "process": "deposition using RF magnetron sputtering on TaN film; Sn films: Rf sputter-deposition at 30W at a deposition rate of 62nm/min, Cu: at 100W at 57nm/min; Before sputtering, TaN substrate cleaning for 1min in a 100℃ solution of 28% ammonia water:30% H2O2 (1:1); annealing at 550℃ for 60min in H2(400Pa) ambient ",
        "property": "agglomeration: CuSn alloys higher than pure Cu; melting point: CuSn alloys lower than pure Cu; Bond strength: Sn-O bond: 531.8kJ/mol, Cu-O bond: 269.0kJ/mol ",
    },
    {
        "composition": "Cu, Co",
        "process": "Deposition using PVD; 400℃/1hr anneal after PVD-Co; using Co/tCoSFB-Cu composite metal system wth TaN<1 ",
        "property": "Line resistance: Co/tCoSFB-Cu composite less than Co/Cu Composite",
    },
    {
        "composition": "Cu, Al",
        "process": "PVD Cu(2at.%Al) seed → ECD Cu → 400℃ annealing",
        "property": "EM activation energy: 1.15±0.1eV (vs pure Cu 0.85eV) / Interface diffusivity reduction",
    },
    {
        "composition": "Cu, Mn",
        "process": "PVD Cu(0.5at.%Mn) seed → Post-CMP annealing at 400℃",
        "property": "Q_GB: 0.77±0.05eV / Z*_GB: -0.4 / Bamboo grain blocking effect",
    },
    {
        "composition": "Co, Cr",
        "process": "DC magnetron sputtering → 450℃ 2hr annealing in N₂",
        "property": "Breakdown voltage: 31.2V (200% ↑ vs pure Co) / 1.2nm Cr₂O₃ barrier formation",
    },
    {
        "composition": "Co, Zn",
        "process": "Chip-on-target sputtering → 450℃ interfacial reaction",
        "property": "Zn₂SiO₄ barrier layer / Breakdown field: 6.2MV/cm",
    },
    {
        "composition": "Cu, Mn, Co",
        "process": "Multi-step annealing (100℃ pre-CMP + 400℃ post)",
        "property": "EM lifetime enhancement >20× / Critical length effect",
    },
    {
        "composition": "Cu, Al, Mn",
        "process": "TaN/Ta liner integration → 3-level damascene architecture",
        "property": "Current density tolerance: 35mA/μm² / Void growth rate control",
    },
    {
        "composition": "Nb",
        "process": "Dilute doping (~1at%) using PVD",
        "property": "Electrical resistivity decreased compared to pure Cu under dilute doping condition",
    },
    {
        "composition": "Fe",
        "process": "Dilute doping (~1at%) using PVD",
        "property": "Electrical resistivity decreased compared to pure Cu under dilute doping condition",
    },
    {
        "composition": "Ag",
        "process": "Submonolayer, monolayer doping (10–90% grain boundary coverage) onto Cu grain boundaries",
        "property": "Specific resistivity (γR) similar to pure Cu",
    },
    {
        "composition": "Zn, Cd, Be, Mg",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage), targeting Σ13a and Σ17 grain boundaries",
        "property": "γR decreased at Σ13a and Σ17 boundaries",
    },
    {
        "composition": "Pd, Ni, Co, Ti",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage)",
        "property": "γR increased for all grain boundaries with doping",
    },
    {
        "composition": "Al, In",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage), targeting Σ13a and Σ17",
        "property": "γR decreased at Σ13a and partially decreased at Σ17",
    },
    {
        "composition": "B",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage), targeting Σ13a",
        "property": "γR partially decreased at Σ13a boundaries",
    },
    {
        "composition": "Ga",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage), targeting Σ13a and Σ17",
        "property": "γR decreased at Σ13a and partially decreased at Σ17",
    },
    {
        "composition": "Si, Sn, Ge",
        "process": "Submonolayer to monolayer doping (10–90% grain boundary coverage), targeting Σ13 boundaries",
        "property": "γR partially decreased at Σ13 boundaries",
    },
    {
        "composition": "Ru, Ta",
        "process": "Deposition using PVD in Ar and N₂ ambient: RuTa → RuTa(N) → Cu; 10% Ta used; nitrogen doping applied",
        "property": "Resistivity: RuTa < RuTa(N) < Ta / Barrier performance: RuTa(N) ≈ RuTa > Ta / Wettability: RuTa/RuTa(N) ≈ Ta / Electromigration lifetime: RuTa > Ta (better wettability reduces sidewall agglomeration and gives narrower lifetime distribution)",
    },
    {
        "composition": "Cu, Mg",
        "process": "DC magnetron sputtering; Cu or Cu-5%Mg alloy used; deposition power: 150W/250W; annealing at 200–500℃ for 5–60 min",
        "property": "Resistivity (after 350℃ annealing): Cu 1.8μΩ·cm, Cu(Mg) 2.0μΩ·cm / Grain size (400℃, 30min): Cu 24–34nm, Cu(Mg) 16–26nm / Debonding energy (SiO₂): Cu 8.76, Cu(Mg) 20.1 J/m² / Lifetime: Cu 2.7h, Cu(Mg) 29.1h / β (at 180℃): Cu 1.54, Cu(Mg) 4.18",
    },
    {
        "composition": "Ag, Cu",
        "process": "Single crystal grown using Czochralski method; alloy prepared by furnace cooling from melt",
        "property": "Single Crystal Ag: 1.49μΩ·cm / Single Crystal Ag-3%Cu: 1.35μΩ·cm / Alloy Ag-3%Cu: 1.76μΩ·cm",
    }
]

# Few-shot 예제를 문자열로 변환
FEW_SHOT_EXAMPLES_STR = "\n\n".join([
    f"composition : {ex['composition']}\n"
    f"process : {ex['process']}\n"
    f"property : {ex['property']}"
    for ex in FEW_SHOT_EXAMPLES
])


# ==================== C-P-P 데이터 구조 정의 (Pydantic) ====================
class CPPData(BaseModel):
    """C-P-P(Composition-Process-Property) 데이터 구조"""
    composition: str = Field(description="합금의 구성 요소 (예: 'Cu, Mg')")
    process: str = Field(description="합금의 제조 및 실험 공정 (예: 'DC magnetron sputtering; annealing at 200–500℃')")
    property: str = Field(description="합금의 주요 특성 (예: 'Resistivity: 2.0μΩ·cm, Debonding energy: 20.1 J/m²')")

# JSON 파서 생성
json_parser = JsonOutputParser(pydantic_object=CPPData)


# ==================== C-P-P 추출용 JSON 프롬프트 ====================
CPP_EXTRACTION_PROMPT = PromptTemplate(
    template="""
You are an AI assistant specializing in materials science. Your task is to extract Composition-Process-Property (C-P-P) data from the given text.
The material should be a metallic alloy suitable for semiconductor interconnects.

Based on the text below, extract the C-P-P data and provide it in JSON format.

**GUIDELINES:**
- **Composition:** Identify the metallic elements and their proportions.
- **Process:** Briefly describe the manufacturing or experimental process.
- **Property:** List the key properties of the alloy.
- If any information is not available, use "N/A".

**FEW-SHOT EXAMPLES:**
---
Text: "We investigated Cu-Mg alloys... DC magnetron sputtering was used... annealing at 350℃... The resistivity of Cu(Mg) was 2.0μΩ·cm, and the debonding energy with SiO₂ was 20.1 J/m²."
JSON Output:
{{
    "composition": "Cu, Mg",
    "process": "DC magnetron sputtering; annealing at 350℃",
    "property": "Resistivity: 2.0μΩ·cm, Debonding energy (SiO₂): 20.1 J/m²"
}}
---
Text: "PVD Cu(2at.%Al) seed was deposited, followed by ECD Cu and 400℃ annealing. This process resulted in an EM activation energy of 1.15±0.1eV."
JSON Output:
{{
    "composition": "Cu, Al (2 at.%)",
    "process": "PVD Cu(Al) seed → ECD Cu → 400℃ annealing",
    "property": "EM activation energy: 1.15±0.1eV"
}}
---

**TEXT TO ANALYZE:**
{text}

**JSON OUTPUT FORMAT:**
{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)


# ==================== Agent용 ReAct 프롬프트 ====================
REACT_SYSTEM_PROMPT = """You are a materials science research agent using the ReAct framework.

You have access to the following tools:
{tools}

=== STRICT OUTPUT FORMAT ===

You MUST follow this format exactly. No blank lines between Thought/Action/Observation.

Thought: [analyze the query and decide what to do next]
Action: [one of: {tool_names}]
Action Input: [input for the tool]
Observation: [tool result — filled in automatically, do NOT write this yourself]
Thought: [analyze the observation and decide what to do next]
Action: [tool name]
Action Input: [input]
Observation: [result]
Thought: [I now have all the information I need to answer.]
Final Answer: [comprehensive answer with citations]

=== CRITICAL RULES ===

1. After every "Thought:", write EXACTLY ONE of:
   (a) "Action: [tool_name]" followed immediately by "Action Input: [input]"  — to call a tool
   (b) "Final Answer: [answer]"  — when you have enough information, NO Action needed
2. NEVER write "Action Input:" without "Action:" on the line immediately before it.
3. NEVER write "Final Answer:" without a "Thought:" on the line immediately before it.
4. NEVER add blank lines between Thought, Action, Action Input, or Final Answer.
5. Do NOT write "Observation:" yourself — it is filled in by the system.

=== TOOL SELECTION RULES ===

- **vectordb_search**: Experimental data from research papers. Use FIRST for material properties, processes, compositions.
- **materials_project**: DFT calculation data only. Input must be exact chemical formula (e.g., "Cu2O", "CuMg"). Use for theoretical properties.
- **crossref_search**: Latest academic papers (English database). Translate Korean queries to English.
- **web_search**: General web info, news, industry trends. Use as last resort.

=== EXAMPLE ===

Question: "Resistivity of Cu-Mg alloys?"
Thought: I need experimental resistivity data from papers. I will use vectordb_search first.
Action: vectordb_search
Action Input: Cu-Mg alloy resistivity
Observation: [experimental data found]
Thought: I have experimental data. I will also check theoretical properties from Materials Project.
Action: materials_project
Action Input: CuMg
Observation: [DFT calculation data]
Thought: I now have sufficient information from both sources to give a comprehensive answer.
Final Answer: [synthesis with citations from both sources]

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
