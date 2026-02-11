import streamlit as st
import json
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

# LLM에게 알려줄 함수의 명세서 (인터페이스 정의)
function = {
    "name": "create_quiz",
    "description": "질문과 답변 목록을 가진 퀴즈를 반환하는 함수",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"}
                                },
                                "required": ["answer", "correct"]
                            }
                        }
                    },
                    "required": ["question", "answers"]
                }
            },
        },
        "required": ["questions"]
    },
}

llm = ChatOpenAI(
    temperature=1.0,
    model="gpt-4.1-nano",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()] # 터미널에 실시간 답변 출력
).bind(
    function_call = {"name": "create_quiz"},
    functions = [function]
)

# 문서에서 퀴즈 텍스트를 생성하는 프롬프트
questions_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """
            당신은 교사 역할을 하는 유능한 비서입니다.

            오직 아래 제공된 Context만을 바탕으로 사용자의 지식을 테스트하기 위한 퀴즈 10개를 만드세요.

            각 문제는 4개의 선택지를 가져야 하며, 그중 3개는 틀린 답이고 1개만 정답이어야 합니다.
            
            Context: {context}
            """
        )
    ]
)

# 문서 포맷팅 -> 퀴즈 생성 체인
questions_chain = {"context": format_docs} | questions_prompt | llm


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

# 캐시 사용 이유:
# 1. OpenAI API 호출 횟수 줄이기 위해
# 2. 동일한 주제/파일에 대해 즉각적인 응답을 주기 위해
@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    # _docs: docs는 해싱 제외
    # topic: 대신 파일명이나 검색어가 바뀐지 보고 재실행 여부 결정
    response = questions_chain.invoke(_docs)
    # | formatting_chain | output_parser

    arguments = response.additional_kwargs["function_call"]["arguments"]

    return json.loads(arguments)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
     # 한국어 위키피디아 검색 & 상위 5개 결과로 리트리버 설정
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.get_relevant_documents(term)
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
            docs = wiki_search(topic)
    
    view_answer = st.toggle("정답 확인")
           



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
    # topic이 있으면 topic을, 없으면 file.name을 캐시 구분 키값으로 사용
    response = run_quiz_chain(docs, topic if topic else file.name)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "선택지를 고르세요.", 
                [answer["answer"] for answer in question["answers"]], 
                index=None
            )
            
            # 해당 문제의 정답 데이터를 미리 변수에 할다 
            correct_answer = None
            for a in question["answers"]:
                if a["correct"]:
                    correct_answer = a["answer"]

            # 사이드바의 '정답 확인' 토글이 활성화되면 정답 제공
            if view_answer:
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                
                # 오답인 경우 정답을 함께 출력
                elif value is not None:
                    st.error(f"Wrong! 정답: {correct_answer}")
                        
        button = st.form_submit_button()