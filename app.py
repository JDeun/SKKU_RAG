"""
AgenticRAG Streamlit App
=========================
재료과학 연구를 위한 AgenticRAG 웹 인터페이스
"""

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
    st.caption("일반 웹 정보 검색")
    st.caption("✅")
    
    st.markdown("---")
    
    # 설정
    st.subheader("⚙️ 설정")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config.LLM_TEMPERATURE,
        step=0.1,
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
        st.session_state.agent = None
        st.rerun()
    
    st.markdown("---")
    
    # 정보
    st.subheader("ℹ️ 정보")
    st.caption(f"모델: {config.LLM_MODEL_NAME}")
    st.caption(f"청크 크기: {config.CHUNK_SIZE}")
    st.caption(f"검색 Top-K: {config.RETRIEVAL_TOP_K}")


# ==================== 메인 영역 ====================
st.title("🔬 AgenticRAG - Materials Science Research Assistant")
st.markdown("반도체 인터커넥트 재료에 대한 질문에 답변합니다.")

# Agent 초기화
if st.session_state.agent is None:
    with st.spinner("🤖 에이전트 초기화 중..."):
        try:
            st.session_state.agent = create_agent(
                verbose=verbose,
                temperature=temperature
            )
            st.success("✅ 에이전트 준비 완료!")
        except Exception as e:
            st.error(f"❌ 에이전트 초기화 실패: {e}")
            st.info("VectorDB가 없다면 먼저 vectordb.py를 실행하세요.")
            st.stop()


# 예시 질문
if not st.session_state.messages:
    st.markdown("### 💡 예시 질문")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Cu-Mg 합금의 저항률은?", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Cu-Mg 합금의 저항률은?"
            })
            st.rerun()
        
        if st.button("Cu2O의 밴드갭은?", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Cu2O의 밴드갭은?"
            })
            st.rerun()
    
    with col2:
        if st.button("electromigration 최신 논문", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "electromigration에 관한 최신 논문을 찾아줘"
            })
            st.rerun()

        if st.button("구리 합금 시장 동향", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "구리 합금의 시장 동향과 최신 뉴스에 대해 알려줘"
            })
            st.rerun()


# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
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
                    return_steps=verbose
                )
                
                response = result["output"]
                
                # 상세 로그 표시
                if verbose and "intermediate_steps" in result:
                    with st.expander("🔍 사고 과정 보기"):
                        for i, step in enumerate(result["intermediate_steps"], 1):
                            st.markdown(f"**Step {i}**")
                            st.code(f"Tool: {step[0].tool}\nInput: {step[0].tool_input}", language="text")
                            st.text_area(f"Output {i}", step[1], height=150, disabled=True)
                
                st.markdown(response)
                
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
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
    <a href='https://github.com' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
