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
    # Few-shot 프롬프팅
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
                """
    You are a helpful assistant that is role playing as a teacher.
            
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
    Use (o) to signal the correct answer.
            
    Question examples:
            
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
            
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
            
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
            
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
            
    Your turn!
            
    Context: {context}
"""
            )
        ]
    )

    chain = {"context": format_docs} | prompt | llm

    start = st.button("Generate Quiz")

    if start:
        chain.invoke(docs)