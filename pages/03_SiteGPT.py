from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def parse_page(soup):
    # 불필요한 UI 요소 제거
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    
    # 텍스트 추출 후 공백, 특수문자 제거
    return (
        str(soup.get_text())
        .replace("\n"," ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    # 토큰 기반 텍스트 분할
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        filter_urls=[r"^(.*\/blog\/).*"], # 특정 경로(/blog/)만 수집하도록 필터링
        parsing_function = parse_page
        )
    
    loader.requests_per_second = 5 # 대상 서버 부하 방지를 위한 속도 제한
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)


with st.sidebar:
    url = st.text_input("URL을 입력하세요", placeholder="https://example.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 입력해주세요")
    else:
        docs = load_website(url)
        st.write(docs)