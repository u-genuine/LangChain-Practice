import streamlit as st
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

# 초기화 - session_state에 memory 저장
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )

memory = st.session_state["memory"]

# 페이지 설정
st.set_page_config(
    page_title = "DocumentGPT", 
    page_icon = "📃"
)

# 스트리밍 응답을 위한 커스텀 콜백 핸들러
class ChatCallbackHandler(BaseCallbackHandler):
    """LLM 토큰 생성 시마다 실시간으로 화면에 표시"""
    message = ""

    def on_llm_start(self, *args, **kwargs):
        """LLM 시작 시: 빈 공간(placeholder)생성"""
        self.message_box = st.empty() # 나중에 업데이트할 빈 공간

    def on_llm_end(self, *args, **kwargs):
        """LLM 종료 시: 완성된 메시지를 session_state에 저장"""
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        """새 토큰 생성될 때마다 호출됨"""
        self.message += token # 누적
        self.message_box.markdown(self.message) # 실시간 업데이트


llm = ChatOpenAI(
    temperature=0.1,
    streaming = True, # 스트리밍 활성화
    callbacks = [ChatCallbackHandler()] # 커스텀 콜백 등록
)

# Streamlit 캐싱: 같은 파일이면 재실행 시 embed_file 건너뜀
# 파일 내용이 변경되면 자동으로 재실행
@st.cache_data(show_spinner = "Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    # 업로드된 파일을 로컬에 저장
    with open(file_path, "wb") as f: # write binary
        f.write(file_content)
    

    # 파일별 임베딩 캐시 저장소
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size = 600,
        chunk_overlap = 100,
    )
    
    # 저장된 파일 로드
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    """메시지를 session_state에 저장"""
    st.session_state["messages"].append({"message": message, "role": role})



def send_message(message, role, save = True):
    """메시지를 화면에 표시하고 선택적으로 저장"""
    # st.chat_message: 채팅 메시지 스타일 UI 생성 (아바타 + 말풍선)
    with st.chat_message(role): 
        st.markdown(message)
    if save:
        save_message(message, role)



def paint_history():
    """저장된 대화 내역 복원"""
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save = False)



def format_docs(docs):
    """문서 리스트를 하나의 텍스트로 변환"""
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)



st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this Chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", 
        type = ["txt", "pdf", "docx"],
    )

if file:
    # 파일 업로드 시: 임베딩 후 채팅 활성화
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save = False)
    paint_history()

    message = st.chat_input("Ask anything about your file....")
    if message:
        send_message(message, "human")

        chain = ({
                "context": retriever | RunnableLambda(format_docs),  
                "question": RunnablePassthrough(),
                "history": lambda x: memory.load_memory_variables({})["history"]
            } 
            | prompt 
            | llm
        )

        # AI 메시지 블록 안에서 invoke 실행
        # → ChatCallbackHandler가 이 블록 안에서 실시간 출력
        with st.chat_message("ai"): 
            response = chain.invoke(message) 

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