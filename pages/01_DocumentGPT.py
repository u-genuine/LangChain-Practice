import time
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title = "DocumentGPT", 
    page_icon = "📃"
)

st.title("DocumentGPT")

# Session State 초기화
# Stremalit은 코드가 실행될 때마다 처음부터 다시 실행됨
# session_state는 새로고침 시에 데이터를 유지하는 저장소
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 디버깅용: 현재 저장된 메시지 확인
st.write(st.session_state["messages"])


# 메시지 전송 함수
def send_message(message, role, save=True):
    """
    메시지를 화면에 표시하고 session_state에 저장

    Args:
        message: 표시할 메시지 내용
        role: "human" or "ai"
        save: True면 session_state에 저장, False면 화면만 표시
    """
    # with: 컨텍스트 매니저 - 블록이 끝나면 자동으로 정리
    # 여기서는 chat_message UI 컴포넌트 생성
    with st.chat_message(role):
        st.write(message)

    if save:
        st.session_state["messages"].append({"message": message, "role": role})


# 기존 대화 내역 복원
# Streamlit은 새로고침마다 코드를 재실행하므로
# session_state에 저장된 이전 대화를 다시 그림
for message in st.session_state["messages"]:
    send_message(
        message["message"],
        message["role"],
        save=False,  # 이미 저장된 대화이므로 중복 저장 방지
    )


# 사용자 입력 처리
# chat_input: 채팅 입력창 생성
message = st.chat_input("Send a message to the ai")


if message:
    # 사용자 메시지 표시 및 저장
    send_message(message, "human")

    # AI 응답 시뮬레이션
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    # 사이드바에 전체 session_state 표시 (디버깅용)
    with st.sidebar:
        st.write(st.session_state)