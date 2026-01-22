import streamlit as st
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS

# 페이지 설정
st.set_page_config(
    page_title = "DocumentGPT", 
    page_icon = "📃"
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



def send_message(message, role, save = True):
    """메시지를 화면에 표시하고 선택적으로 session_state에 저장"""
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})



def paint_history():
    """저장된 대화 내역을 화면에 복원"""
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save = False)


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
        send_message("lalala", "ai")

else:
    # 파일 없으면 대화 내역 초기화
    st.session_state["messages"] = []