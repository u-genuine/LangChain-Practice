from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )
memory = st.session_state["memory"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwaghs):
        self.message += token
        self.message_box.markdown(self.message)
    
# Map 단계 LLM은 스트리밍 X
silent_llm = ChatOpenAI(
    temperature=0.1
)

# 최종 답변에 사용할 LLM 스트리밍 O
llm = ChatOpenAI(
    temperature=0.1,
    streaming = True,
    callbacks = [ChatCallbackHandler()]
)


# 메시지를 session_state에 저장
def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )

# 메시지를 화면에 표시
def draw_message(message, role, save = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message ,role)

def draw_history():
    for message in st.session_state["messages"]:
        draw_message(message["message"], message["role"], False)


answers_prompt = ChatPromptTemplate.from_template("""
    주어진 context만을 이용해서 사용자의 질문에 답변하세요. 답변할 수 없다면, 지어내지 말고 모른다고 하세요. 

    그리고 각 답변을 0부터 5까지의 점수로 평가해주세요. 0점은 사용자에게 쓸모없음, 5점은 사용자에게 매우 유용함을 의미합니다.

    답변과 점수를 모두 포함해주세요.

    Conext: {context}
                                                  
    예제:

    질문: 달까지의 거리는?
    답변: 달은 384,400 km 떨어져있습니다.
    점수: 5

    질문: 태양까지의 거리는?
    답변: 모름
    점수: 0

    Question: {question}
"""
)

def get_answers(inputs):
    """
    [Map 단계]
    검색된 여러 문서 조각(docs)를 순회하며 각각 질문에 대한 개별 답변 생성
    """
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | silent_llm

    return {
        "question": question, 
        "answers": [
            {
                # 각 문서 조각마다 LLM에게 답변과 점수 요청
                "answer": answers_chain.invoke({
                    "question": question,"context": doc.page_content
                    }).content,
                "source": doc.metadata["source"],
                "date":doc.metadata["lastmod"]
            } for doc in docs
        ]
    }

choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        먼저 생성된 Answers만을 사용하여 사용자의 질문에 답변하세요.

        더 높은 점수를 가진 답변들을 사용하세요. (더 유용합니다)
        
        최신의 자료를 우선시하고, 출처도 링크의 형태로 남겨주세요.

        Answers: {answers}
        """
    ),
    ("human", "{question}")
])

def choose_answer(inputs):
    """
    [Rerank 단계]
    Map 단계에서 생성된 여러 답변 중 가장 우수한 것을 골라 최종 답변 생성
    """
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    
    # 여러 답변을 하나로 합쳐서 최종 프롬프트에 전달할 컨텍스트 생성
    condenced = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n" 
        for answer in answers
    )
   
    return choose_chain.invoke({
        "answers": condenced,
        "question": question
    })

# BeautifulSoup 객체: HTML 태그를 파이썬 객체처럼 다루게 해주는 도구
def parse_page(soup):
    """
    SitemapLoacer가 웹페이지를 읽어오면 BeaurifulSoup 객체를 생성해 이 함수로 전달
    soup은 soup.find() 같은 메서듣로 특정 태그를 찾거나 제거할 수 있는 상태
    """
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
        filter_urls=[r"^(.*\/research\/).*"], # 특정 경로(/research/)만 수집하도록 필터링
        parsing_function = parse_page # soup 처리 함수 적용
    )
    
    loader.requests_per_second = 5 # 대상 서버 부하 방지를 위한 속도 제한
    docs = loader.load_and_split(text_splitter=splitter)

    # 벡터 저장소 생성: 텍스트를 숫자로 변환(Embedding)하여 FAISS에 저장
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever() # 질문과 유사한 문서를 찾아주는 리트리버 반환

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)


with st.sidebar:
    url = st.text_input("URL을 입력하세요", placeholder="https://example.com")

if not url:
    url = "https://deepmind.google/sitemap.xml"
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 입력해주세요")
    else:
        # 사이트 맵 로드 & 벡터스토어 빌드
        retriever = load_website(url) 

        draw_history()

        query = st.chat_input("Ask a question to the website.")

        if query:
            # with chat_container:
            draw_message(query, "human")
            
            # Map-Rerank 체인
            chain = (
                {
                    # 질문과 관련된 문서들을 리트리버가 찾아옴
                    "docs": retriever,
                    # 입력된 질문을 그대로 다음 함수로 전달
                    "question": RunnablePassthrough(),
                } 
                # Map-Rerank: 각 문서에서 답을 찾고 점수를 매겨 최적의 답변 선별
                | RunnableLambda(get_answers) 
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                result = chain.invoke(query)

                with st.expander("Memory Debug (기억 데이터 확인)"):
                    st.write(memory.load_memory_variables({}))
            
            memory.save_context(
                {"input": query},
                {"output": result.content.replace("$", "\$")}
            )
