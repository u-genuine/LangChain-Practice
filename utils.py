import streamlit as st
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS

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

def format_docs(docs):
    """문서 리스트를 하나의 텍스트로 변환"""
    return "\n\n".join(doc.page_content for doc in docs)