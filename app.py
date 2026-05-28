"""
AgenticRAG Streamlit App
=========================
재료과학 연구를 위한 AgenticRAG 웹 인터페이스
"""

import logging
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import config
from agent import create_agent, run_agent


# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="AgenticRAG - Materials Science Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 세션 상태 초기화 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "agent_temp" not in st.session_state:
    st.session_state.agent_temp = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ==================== 사이드바 ====================
with st.sidebar:
    st.title("🔬 AgenticRAG")
    st.markdown("재료과학 연구 AI 에이전트")
    
    st.markdown("---")
    
    # 도구 정보
    st.subheader("📚 사용 가능한 도구")

    st.markdown("**1. VectorDB Search**")
    st.caption("논문에서 C-P-P 데이터 검색")

    st.markdown("**2. Materials Project**")
    st.caption("DFT 계산 데이터 조회")
    mp_status = "✅" if config.MATERIALS_PROJECT_API_KEY else "⚠️"
    st.caption(f"상태: {mp_status}")

    st.markdown("**3. Crossref**")
    st.caption("최신 논문 검색")
    st.caption("✅")

    st.markdown("**4. Web Search**")
    st.caption("일반 웹 정보 검색 (LLM 요약)")
    st.caption("✅")

    st.markdown("**5. arXiv Search**")
    st.caption("프리프린트 논문 검색")
    st.caption("✅ API 키 불필요")

    st.markdown("**6. OQMD Search**")
    st.caption("DFT 계산 데이터 (cross-reference)")
    st.caption("✅ API 키 불필요")

    st.markdown("---")
    
    # 설정
    st.subheader("⚙️ 설정")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config.LLM_TEMPERATURE,
        step=0.1,
        key="temperature_slider",
        help="값이 낮을수록 일관된 답변, 높을수록 창의적인 답변"
    )
    
    verbose = st.checkbox(
        "상세 로그",
        value=False,
        help="에이전트의 사고 과정 표시"
    )
    
    # 대화 초기화
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        # 에이전트 메모리도 함께 리셋
        if st.session_state.agent and hasattr(st.session_state.agent, 'memory'):
            st.session_state.agent.memory.clear()
        st.session_state.agent = None
        st.session_state.agent_temp = None
        st.rerun()
    
    st.markdown("---")
    
    # 정보
    st.subheader("ℹ️ 정보")
    st.caption(f"모델: {config.LLM_MODEL_NAME}")
    if config.GROQ_API_KEY:
        st.caption(f"Fallback: {config.GROQ_MODEL_NAME} ✅")
    else:
        st.caption("Fallback: Groq 미설정 ⚠️")
    st.caption(f"청크 크기: {config.CHUNK_SIZE}")
    st.caption(f"검색 Top-K: {config.RETRIEVAL_TOP_K}")


# ==================== 메인 영역 ====================
st.title("🔬 AgenticRAG - Materials Science Research Assistant")
st.markdown("반도체 인터커넥트 재료에 대한 질문에 답변합니다.")

# Agent 초기화 (temperature 변경 시 재생성)
needs_init = st.session_state.agent is None
needs_reinit = st.session_state.agent_temp != temperature

if needs_init or needs_reinit:
    with st.spinner("🤖 에이전트 초기화 중..."):
        try:
            st.session_state.agent = create_agent(
                verbose=verbose,
                temperature=temperature
            )
            st.session_state.agent_temp = temperature
            if needs_init:
                st.success("✅ 에이전트 준비 완료!")
        except Exception:
            logging.exception("Agent initialization failed")
            st.error("에이전트 초기화에 실패했습니다. 설정과 VectorDB를 확인하세요.")
            st.info("VectorDB가 없다면 먼저 vectordb.py를 실행하세요.")
            st.stop()


# 예시 질문
if not st.session_state.messages:
    st.markdown("### 💡 예시 질문")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Cu-Mg 합금의 저항률은?", use_container_width=True):
            st.session_state.pending_query = "Cu-Mg 합금의 저항률은?"
            st.rerun()

        if st.button("Cu2O의 밴드갭은?", use_container_width=True):
            st.session_state.pending_query = "Cu2O의 밴드갭은?"
            st.rerun()

    with col2:
        if st.button("electromigration 최신 논문", use_container_width=True):
            st.session_state.pending_query = "electromigration에 관한 최신 논문을 찾아줘"
            st.rerun()

        if st.button("구리 합금 시장 동향", use_container_width=True):
            st.session_state.pending_query = "구리 합금의 시장 동향과 최신 뉴스에 대해 알려줘"
            st.rerun()


# 대화 기록 표시 (verbose 켜져 있으면 과거 사고 과정도 함께 표시)
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if verbose and message["role"] == "assistant" and message.get("steps"):
            with st.expander("🔍 사고 과정 보기"):
                for i, step in enumerate(message["steps"], 1):
                    st.markdown(f"**Step {i}**")
                    st.code(f"Tool: {step[0].tool}\nInput: {step[0].tool_input}", language="text")
                    st.text_area(f"Output {i}", str(step[1]), height=150, disabled=True, key=f"hist_{idx}_{i}")
        st.markdown(message["content"])


# 사용자 입력 (채팅 입력 또는 예시 버튼)
_pending = st.session_state.pending_query
if _pending:
    st.session_state.pending_query = None

_chat_input = st.chat_input("질문을 입력하세요...")
prompt = (_chat_input or _pending or "").strip()

if prompt:
    if len(prompt) > 2000:
        st.warning("질문이 너무 깁니다. 2000자 이내로 입력해 주세요.")
        st.stop()

    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 에이전트 응답
    with st.chat_message("assistant"):
        with st.spinner("🔍 검색 중..."):
            try:
                result = run_agent(
                    prompt,
                    agent=st.session_state.agent,
                    return_steps=True  # 항상 steps 수집 (로그 이력 보존용)
                )

                response = result["output"]
                steps = result.get("intermediate_steps", [])

                # 상세 로그 표시 (현재 응답)
                if verbose and steps:
                    with st.expander("🔍 사고 과정 보기"):
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"**Step {i}**")
                            st.code(f"Tool: {step[0].tool}\nInput: {step[0].tool_input}", language="text")
                            st.text_area(f"Output {i}", str(step[1]), height=150, disabled=True, key=f"curr_{len(st.session_state.messages)}_{i}")

                st.markdown(response)

                # 응답 저장 (steps 포함 — 나중에 verbose 로그 이력 표시에 사용)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "steps": steps
                })
                
            except Exception:
                logging.exception("Agent execution failed in Streamlit")
                error_msg = "❌ 에이전트 실행 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ==================== 푸터 ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    AgenticRAG | Powered by LangChain & Gemini | 
    <a href='https://github.com/JDeun/SKKU_RAG' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
