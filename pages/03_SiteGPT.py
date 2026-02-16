from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )
memory = st.session_state["memory"]

# UI에 표시할 메시지 리스트 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 유사 질문 캐싱을 위한 리스트 초기화
if "qa_cache" not in st.session_state:
    st.session_state["qa_cache"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_new_token(self, token, *args, **kwaghs):
        self.message += token
        self.message_box.markdown(self.message)
    

# 스트리밍 비활성화 LLM
silent_llm = ChatOpenAI(
    temperature=0.1
)

# 스트리밍 활성화 LLM
llm = ChatOpenAI(
    temperature=0.1,
    streaming = True,
    callbacks = [ChatCallbackHandler()]
)


# 대화 내용을 session_state에 저장
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# 화면에 말풍선을 그리고 필요시 저장
def draw_message(message, role, save = True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message ,role)

# 기존 대화 내역을 화면에 복원
def draw_history():
    for message in st.session_state["messages"]:
        draw_message(message["message"], message["role"], False)


# [1단계] 각 문서 조각에서 답변을 추출하기 위한 프롬프트
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

# [2단계] 추출된 답변들 중 최적의 답변을 고르기 위한 프롬프트
choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        Answers 중 가장 점수가 높고 최신인 정보를 사용하여 사용자의 질문에 답변하세요.
        출처도 링크의 형태로 남겨주세요.
        이전 대화 맥락이 있다면 이를 고려하여 자연스럽게 답변하세요

        Answers: {answers}
        """
    ),
    MessagesPlaceholder(variable_name="history"),
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
        "history": memory.load_memory_variables({})["history"],
        "question": question
    })

# BeautifulSoup 객체: HTML 태그를 파이썬 객체처럼 다루게 해주는 도구
def parse_page(soup):
    """
    SitemapLoader가 수집한 HTML에서 불필요한 태그를 제거하는 필터
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


@st.cache_data(show_spinner="웹사이트 로딩 중...")
def load_website(url):
    """웹사이트 내용을 읽어 임베딩 후 검색 엔진 반환"""
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
    return vector_store.as_retriever() # 질문과 유사한 문서를 찾아주는 검색 엔진 반환

@st.cache_data(show_spinner="답변 생성 중...")
def get_cached_answer(query, _retriever):
    """동일 질문 캐싱을 적용한 답변 생성 체인 실행"""
    # Map-Rerank: 각 문서에서 답을 찾고 점수를 매겨 최적의 답변 선별
    chain = (
        {
            "docs": _retriever, # 질문과 관련된 문서들을 검색 엔진이 찾아옴
            "question": RunnablePassthrough(), # 입력된 질문을 그대로 다음 함수로 전달
        } 
        | RunnableLambda(get_answers) 
        | RunnableLambda(choose_answer)
    )
    return chain.invoke(query)


# 유사한 질문 판별하기 위한 프롬프트
find_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     당신은 유사한 질문을 판별하는 판독관입니다.
     아래 제공된 이전 질문 리스트 중에서 사용자의 현재 질문과 의미상 동일하거나 매우 유사한 질문이 있는지 확인하세요.

     - 유사한 질문이 있다면, 해당 질문에 대한 '답변 내용'만 정확히 출력하세요
     - 유사한 질문이 없다면, 반드시 '없음'이라고만 대답하세요
     
     이전 질문 리스트: {cache_list}
"""),
    ("human", "{question}")
])

def find_similar_question(query):
    """이전 질문 내역에서 의미상 비슷한 질문이 있는지 LLM에게 확인"""

    cache_list = "\n".join([
        f"질문: {item['query']} -> 답변: {item['answer']}" 
        for item in st.session_state["qa_cache"]
    ])

    find_chain = find_prompt | silent_llm
    response = find_chain.invoke({
        "cache_list": cache_list,
        "question": query
    })

    answer = response.content
    if answer == "없음":
        return None
    return answer


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)

with st.sidebar:
    url = st.text_input("URL을 입력하세요", placeholder="https://example.com")

if ".xml" not in url:
    with st.sidebar:
        st.error("Sitemap URL을 입력해주세요")

else:
    # 사이트 맵 로드 & 벡터스토어 빌드
    retriever = load_website(url) 
    draw_history()

    query = st.chat_input("Ask a question to the website.")

    if query:
        draw_message(query, "human")

        # 유사 질문이 있는지 확인
        similar_answer = find_similar_question(query)

        if similar_answer:
            # 비슷한 질문이 있었다면 저장된 답변 출력
            draw_message(similar_answer, "ai")
        
        else:
            # 새로운 질문이면 Map-Rerank 체인 실행
            with st.chat_message("ai"):
                result = get_cached_answer(query, retriever)
            
            save_message(result.content.replace("$", "\$"), "ai")
            st.session_state["qa_cache"].append({
                "query": query,
                "answer": result.content.replace("$", "\$")
            })
            memory.save_context(
                {"input": query},
                {"output": result.content.replace("$", "\$")}
            )

        with st.expander("Memory Debug (기억 데이터 확인)"):
            st.write(memory.load_memory_variables({}))