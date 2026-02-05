import streamlit as st
import os
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

# 페이지 설정
st.set_page_config(page_title = "DocumentGPT", page_icon = "📃")
st.title("DocumentGPT")


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key를 입력하세요",
        type = "password"
    )

    st.divider()

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", 
        type = ["txt", "pdf", "docx"],
    )

    st.divider()

    st.markdown("""
    ### Github Repo
    (https://github.com/u-genuine/LangChain-Practice)
    """)

if not openai_api_key:
    st.info("Please add your OpenAI API Key in the siddbar to start chatting.")
    st.stop()


# 스트리밍 응답을 위한 커스텀 콜백 핸들러 클래스
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs): 
        """LLM 시작 시: 빈 공간(placeholder)생성"""
        self.message_box = st.empty() # 실시간 토큰을 채워넣을 빈 칸

    def on_llm_new_token(self, token, *args, **kwargs):
        """새 토큰 생성될 때마다 호출됨"""
        self.message += token
        self.message_box.markdown(self.message) # 빈 칸에 마크다운 형식으로 텍스트 누적

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai") # 답변 완료 시 세션에 최종 저장



def save_message(message, role):
    """메시지를 session_state에 저장"""
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save = True):
    """메시지를 화면에 표시하고 선택적으로 저장"""
    # st.chat_message: 채팅 메시지 스타일 UI 생성 (아바타 + 말풍선)
    with st.chat_message(role):  # 말풍선 내부에 콘텐츠 배치
        st.markdown(message) 
    if save:
        save_message(message, role)


def paint_history():
    """저장된 대화 기록 화면에 표시"""
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save = False)


def format_docs(docs):
    """문서 리스트를 하나의 텍스트로 변환"""
    return "\n\n".join(doc.page_content for doc in docs)


# Streamlit 캐싱: 같은 파일이면 재실행 시 embed_file 건너뜀
# 파일 내용이 변경되면 자동으로 재실행
@st.cache_data(show_spinner = "Embedding file...")
def embed_file(file, api_key):
    """파일을 저장하고 임베딩하여 retriever 반환"""
    file_content = file.read()

    # 폴더가 없으면 생성
    cache_dir_path = "./.cache/files"
    if not os.path.exists(cache_dir_path):
        os.makedirs(cache_dir_path)

    file_path = f"{cache_dir_path}/{file.name}"

    # 업로드된 파일을 로컬에 저장
    with open(file_path, "wb") as f: # write binary
        f.write(file_content)
    

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    
    # 저장된 파일 로드
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    cache_embeddings_path = f"./.cache/embeddings/{file.name}"
    if not os.path.exists(cache_embeddings_path):
        os.makedirs(cache_embeddings_path)

    embeddings = OpenAIEmbeddings(api_key = api_key)
    cache_dir = LocalFileStore(cache_embeddings_path)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


# 세션 스테이트 초기화 (재실행 시에도 유지됨)
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory( # 버퍼 크기는 어떻게되지?
        return_messages=True,
        memory_key="history"
    )

memory = st.session_state["memory"] 


# 모델 설정
llm = ChatOpenAI(
    temperature=0.1,
    streaming = True, # 스트리밍 활성화
    callbacks = [ChatCallbackHandler()], # 커스텀 콜백 등록
    openai_api_key = openai_api_key # 입력받은 키 적용
)


# UI 레이아웃
st.markdown(
    """
Welcome!

Use this Chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)



# 메인 로직
if file:
    retriever = embed_file(file, openai_api_key)
    send_message("I'm ready! Ask away!", "ai", save = False)
    paint_history() # 이전 대화 복원

    message = st.chat_input("Ask anything about your file....")
    if message:
        send_message(message, "human")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        chain = ({
                "context": retriever | RunnableLambda(format_docs),  
                "question": RunnablePassthrough(),
                "history": lambda x: memory.load_memory_variables({})["history"]
            } 
            | prompt 
            | llm
        )

        # AI 메시지 블록 안에서 invoke 실행
        # 이거 위에 send_message 함수를 안쓰는 이유?
        with st.chat_message("ai"): 
            response = chain.invoke(message) # ChatCallbackHandler가 이 블록 안에서 실시간 출력

            with st.expander("Memory Debug (기억 데이터 확인)"):
                st.write(memory.load_memory_variables({}))
        
        memory.save_context(
            {"input": message},
            {"output": response.content}            
        )

else:
    # 파일 없으면 대화 내역 초기화
    st.session_state["messages"] = []
    st.session_state["memory"].clear()