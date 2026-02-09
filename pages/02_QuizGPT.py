import streamlit as st
import os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from utils import format_docs
from pathlib import Path

# 스트림릿 페이지 설정
st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤔"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=1.0,
    model="gpt-5-nano",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()] # 터미널에 실시간 답변 출력
)

# Few-shot 프롬프팅
questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """
            당신은 교사 역할을 하는 유능한 비서입니다.

            오직 아래 제공된 Context만을 바탕으로 사용자의 지식을 테스트하기 위한 퀴즈 10개를 만드세요.

            각 문제는 4개의 선택지를 가져야 하며, 그중 3개는 틀린 답이고 1개만 정답이어야 합니다.

            정답인 선택지 뒤에는 반드시 (✅) 표시를 하세요.
            
            질문 예시: 
            
            질문: 바다의 색깔은 무엇인가요?
            선택지: 빨강 | 노랑 | 초록 | 파랑(✅)

            질문: 조지아의 수도는 어디인가요?
            선택지: 바쿠 | 트빌리시(✅) | 마닐라 | 베이루트
            
            질문: 영화 '아바타'는 언제 개봉했나요?
            선택지: 2007 | 2001 | 2009(✅) | 1998
            
            질문: Julius Caesar는 누구인가요?
            선택지: 로마의 황제(✅) | 화가 | 배우 | 모델
            
            당신의 차례입니다!
            
            Context: {context}
            """
            )
        ]
    )


questions_chain = {"context": format_docs} | questions_prompt | llm

# Few-shot 프롬프팅
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 강력한 포맷팅 알고리즘입니다.

            당신은 시험 문제들을 전달받아 JSON 형식으로 변환합니다.
            선택지 중 (✅) 표시가 있는 것이 정답입니다.

            입력 예시:    

            질문: 바다의 색깔은 무엇인가요?
            선택지: 빨강 | 노랑 | 초록 | 파랑(✅)

            질문: 조지아의 수도는 어디인가요?
            선택지: 바쿠 | 트빌리시(✅) | 마닐라 | 베이루트
            
            질문: 영화 '아바타'는 언제 개봉했나요?
            선택지: 2007 | 2001 | 2009(✅) | 1998
            
            질문: Julius Caesar는 누구인가요?
            선택지: 로마의 황제(✅) | 화가 | 배우 | 모델

            출력 예시:
            
            ```json
            {{ "questions": [
                {{
                    "question": "바다의 색깔은 무엇인가요?",
                    "answers": [
                            {{ "answer": "빨강", "correct": false }},
                            {{ "answer": "노랑", "correct": false }},
                            {{ "answer": "초록", "correct": false }},
                            {{ "answer": "파랑", "correct": true }}
                    ]
                }},
                {{
                    "question": "조지아의 수도는 어디인가요?",
                    "answers": [
                            {{ "answer": "바쿠", "correct": false }},
                            {{ "answer": "트빌리시", "correct": true }},
                            {{ "answer": "마닐라", "correct": false }},
                            {{ "answer": "베이루트", "correct": false }}
                    ]
                }},
                {{
                    "question": "영화 '아바타'는 언제 개봉했나요?",
                    "answers": [
                            {{ "answer": "2007", "correct": false }},
                            {{ "answer": "2001", "correct": false }},
                            {{ "answer": "2009", "correct": true }},
                            {{ "answer": "1998", "correct": false }}
                    ]
                }},
                {{
                    "question": "율리우스 카이사르(Julius Caesar)는 누구인가요?",
                    "answers": [
                            {{ "answer": "로마의 황제", "correct": true }},
                            {{ "answer": "화가", "correct": false }},
                            {{ "answer": "배우", "correct": false }},
                            {{ "answer": "모델", "correct": false }}
                    ]
                }}
            ]
            }}
            ```
            당신 차례입니다!

            질문 목록: {context}
        """
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner = "Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    # 업로드된 파일을 로컬에 저장
    with open(file_path, "wb") as f: # write binary
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    
    # 저장된 파일 로드 & 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


# 사이드바 UI
with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.", 
        ("File", "Wikipedia Article")
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docs, .txt or .pdf file", 
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)

    else :
        topic = st.text_input("Search Wikipedia...")
        if topic:
            # 한국어 위키피디아 검색 & 상위 5개 결과로 리트리버 설정
            retriever = WikipediaRetriever(top_k_results=5, lang="ko")
            with st.status("Searching wikipedia..."):
                docs = retriever.get_relevant_documents(topic)



# 메인 화면
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )


else:
    start = st.button("Generate Quiz")

    if start:
        questions_response = questions_chain.invoke(docs)
        st.write(questions_response.content)
        formatting_response = formatting_chain.invoke({"context": questions_response.content})
        st.write(formatting_response.content)