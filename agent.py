"""
AgenticRAG Agent
================
ReAct 프레임워크를 사용하여 4개의 도구를 통합한 에이전트입니다.
"""

import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from typing import Optional, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import config
import prompts

from tools.vectordb_search import vectordb_search_tool
from tools.materials_project import materials_project_tool
from tools.crossref import crossref_tool
from tools.web_search import web_search_tool


# ==================== LLM 빌더 (Gemini → Groq fallback) ====================
def _build_llm(temperature: float):
    """
    Gemini를 기본 LLM으로 생성합니다.
    GROQ_API_KEY가 설정되어 있으면 Gemini 실패 시 Groq로 자동 전환합니다.

    LangChain의 with_fallbacks()를 사용하므로 호출 코드 변경 없이 동작합니다.
    Gemini가 사용량 한도 초과·인증 오류·타임아웃 등으로 예외를 던지면
    즉시 Groq로 재시도합니다.
    """
    gemini = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        temperature=temperature,
        streaming=config.LLM_STREAMING,
        max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        google_api_key=config.GOOGLE_API_KEY
    )

    if not config.GROQ_API_KEY:
        return gemini

    from langchain_groq import ChatGroq
    groq_llm = ChatGroq(
        model=config.GROQ_MODEL_NAME,
        temperature=temperature,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        api_key=config.GROQ_API_KEY
    )
    return gemini.with_fallbacks([groq_llm])


# ==================== Agent 초기화 ====================
def create_agent(
    verbose: bool = config.VERBOSE,
    temperature: float = config.LLM_TEMPERATURE
) -> AgentExecutor:
    """
    ReAct 에이전트를 생성합니다.

    Args:
        verbose: 상세 로그 출력 여부
        temperature: LLM temperature

    Returns:
        AgentExecutor 인스턴스
    """
    llm = _build_llm(temperature)

    tools = [
        vectordb_search_tool,
        materials_project_tool,
        crossref_tool,
        web_search_tool
    ]

    react_prompt = PromptTemplate.from_template(prompts.REACT_SYSTEM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=config.AGENT_MAX_ITERATIONS,
        max_execution_time=config.AGENT_TIMEOUT,
        handle_parsing_errors=(
            "Output format error. Rewrite your response following this exact format:\n"
            "Thought: [your reasoning]\n"
            "Action: [tool name]\n"
            "Action Input: [query]\n"
            "OR, if you already have enough information:\n"
            "Thought: [your reasoning]\n"
            "Final Answer: [your answer]\n"
            "Do NOT add blank lines between these lines."
        ),
        return_intermediate_steps=True
    )

    return agent_executor


# ==================== Agent 실행 ====================
def run_agent(
    query: str,
    agent: Optional[AgentExecutor] = None,
    return_steps: bool = False
) -> Dict[str, Any]:
    """
    에이전트를 실행하여 쿼리에 답변합니다.

    Args:
        query: 사용자 질문
        agent: AgentExecutor 인스턴스 (None이면 새로 생성)
        return_steps: 중간 단계 반환 여부

    Returns:
        {
            "output": str,
            "intermediate_steps": list  # return_steps=True인 경우
        }
    """
    if agent is None:
        agent = create_agent()

    try:
        result = agent.invoke({"input": query})

        response: Dict[str, Any] = {
            "output": result.get("output", "답변을 생성할 수 없습니다.")
        }

        if return_steps:
            response["intermediate_steps"] = result.get("intermediate_steps", [])

        return response

    except Exception:
        logging.exception("Agent execution failed")
        return {
            "output": "에이전트 실행 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "error": "agent_error"
        }


# ==================== 대화형 인터페이스 ====================
def interactive_chat():
    """
    CLI 기반 대화형 인터페이스
    """
    print("="*60)
    print("AgenticRAG 챗봇")
    print("="*60)
    print("재료과학 연구를 위한 AI 에이전트입니다.")
    print("- VectorDB: 논문의 C-P-P 데이터 검색")
    print("- Materials Project: DFT 계산 데이터")
    print("- Crossref: 최신 논문 검색")
    print("- Web Search: 일반 웹 정보 검색")
    print("\n종료하려면 'exit', 'quit', 또는 'q'를 입력하세요.\n")
    print("="*60 + "\n")

    print("🤖 에이전트 초기화 중...")
    try:
        agent = create_agent(verbose=True)
        print("✅ 준비 완료!\n")
    except Exception:
        logging.exception("Agent initialization failed")
        print("❌ 에이전트 초기화 실패. 설정을 확인하세요.")
        return

    while True:
        try:
            user_input = input("💬 질문: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 AgenticRAG를 종료합니다.")
                break

            if not user_input:
                continue

            print("\n🔍 검색 중...\n")
            result = run_agent(user_input, agent=agent)

            print("\n" + "="*60)
            print("📝 답변:")
            print("="*60)
            print(result["output"])
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 AgenticRAG를 종료합니다.")
            break
        except Exception:
            logging.exception("Interactive chat error")
            print("\n❌ 오류 발생: 잠시 후 다시 시도해 주세요.\n")


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgenticRAG Agent")
    parser.add_argument(
        "--query",
        type=str,
        help="단일 쿼리 실행 (대화형 모드 대신)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    config.print_config()

    if args.query:
        print(f"질문: {args.query}\n")
        agent = create_agent(verbose=args.verbose)
        result = run_agent(args.query, agent=agent, return_steps=True)

        print("="*60)
        print("답변:")
        print("="*60)
        print(result["output"])

        if args.verbose and "intermediate_steps" in result:
            print("\n" + "="*60)
            print("중간 단계:")
            print("="*60)
            for step in result["intermediate_steps"]:
                print(f"\nAction: {step[0].tool}")
                print(f"Input: {step[0].tool_input}")
                out = str(step[1])
                print(f"Output: {out[:200]}{'...' if len(out) > 200 else ''}")

    else:
        interactive_chat()
