"""
AgenticRAG Agent
================
ReAct 프레임워크를 사용하여 4개의 도구를 통합한 에이전트입니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from typing import Optional, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# 설정 및 프롬프트
import config
import prompts

# 도구 import
from tools.vectordb_search import vectordb_search_tool
from tools.materials_project import materials_project_tool
from tools.crossref import crossref_tool
from tools.web_search import web_search_tool
from tools.arxiv_search import arxiv_search_tool
from tools.oqmd_search import oqmd_search_tool


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
    # LLM 초기화
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        temperature=temperature,
        streaming=config.LLM_STREAMING,
        google_api_key=config.GOOGLE_API_KEY
    )

    # 도구 리스트
    tools = [
        vectordb_search_tool,
        materials_project_tool,
        crossref_tool,
        web_search_tool,
        arxiv_search_tool,
        oqmd_search_tool,
    ]

    # 멀티턴 대화 메모리
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=config.MEMORY_WINDOW_SIZE,
        return_messages=False
    )

    # ReAct 프롬프트 구성
    react_prompt = PromptTemplate.from_template(prompts.REACT_SYSTEM_PROMPT)

    # ReAct Agent 생성
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # AgentExecutor로 래핑
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        max_iterations=config.AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
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
            "output": str,  # 최종 답변
            "intermediate_steps": list  # 중간 단계 (return_steps=True인 경우)
        }
    """
    if agent is None:
        agent = create_agent()

    try:
        result = agent.invoke({"input": query})

        response = {
            "output": result.get("output", "답변을 생성할 수 없습니다.")
        }

        if return_steps:
            response["intermediate_steps"] = result.get("intermediate_steps", [])

        return response

    except Exception as e:
        return {
            "output": f"에이전트 실행 중 오류 발생: {str(e)}",
            "error": str(e)
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
    print("- Web Search: 일반 웹 정보 검색 (LLM 요약)")
    print("- arXiv Search: 프리프린트 논문 검색 (API 키 불필요)")
    print("- OQMD Search: DFT 계산 데이터 (API 키 불필요)")
    print("\n종료하려면 'exit', 'quit', 또는 'q'를 입력하세요.\n")
    print("="*60 + "\n")

    # Agent 초기화
    print("🤖 에이전트 초기화 중...")
    try:
        agent = create_agent(verbose=True)  # 추론 과정 표시
        print("✅ 준비 완료!\n")
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")
        return

    # 대화 루프
    while True:
        try:
            # 사용자 입력
            user_input = input("💬 질문: ").strip()

            # 종료 명령
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 AgenticRAG를 종료합니다.")
                break

            # 빈 입력 스킵
            if not user_input:
                continue

            # 에이전트 실행
            print("\n🔍 검색 중...\n")
            result = run_agent(user_input, agent=agent)

            # 결과 출력
            print("\n" + "="*60)
            print("📝 답변:")
            print("="*60)
            print(result["output"])
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 AgenticRAG를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


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

    # 설정 출력
    config.print_config()

    # 단일 쿼리 모드
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
                print(f"Output: {step[1][:200]}...")

    # 대화형 모드
    else:
        interactive_chat()
