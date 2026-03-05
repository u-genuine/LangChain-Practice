from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(dotenv_path="./env/.env")
openai.api_key = os.getenv('OPENAI_API_KEY')

index_name = "recipes"

# Pinecone 초기화 & 인덱스 연결
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)

# OpenAI 임베딩 모델 초기화 (벡터 변환에 사용)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Pinecone 벡터 스토어 연결 - 이미 저장된 인덱스 불러옴
vector_store = PineconeVectorStore(index_name=index_name,embedding=embeddings)

# FastAPI 앱 초기화 & OpenAPI 문서 설정
# /docs 경로에서 Swagger UI로 확인 가능
app = FastAPI(
    title="레시피 GPT. 당신만의 흑백요리사",
    description="재료를 몇 가지 알려주시면 레시피를 추천해드립니다.",
    servers=[
        # GPTs Actio이 API를 호출할 때 사용할 서버 주소 (cloudflared Tunnel URL)
        {"url": "https://cycle-industries-kitty-apps.trycloudflare.com"}
    ]
)

# 응답 스키마 - Pinecone에서 꺼낸 Document 본문만 반환
class Document(BaseModel):
    page_content: str

# GET /recipe - 재료를 받아 유사한 레시피 목록을 반환하는 엔드포인트
# GPTs가 ingredient 쿼리 파라미터와 함께 이 엔드포인트를 호출
@app.get(
    "/recipes", 
    summary="재료를 입력받아 레시피 목록을 반환합니다.",
    description="재료를 쿼리 파라미터로 전달하면 해당 재료가 포함된 레시피 목록을 반환합니다.",
    response_description="레시피 내용과 조리 방법이 담긴 Document 객체",
    response_model=list[Document], # Document Object를 만들기 위해 pydantic 사용
    openapi_extra={
        "x-openai-isConsequential": False
            # False: 한 번 허용/항상 허용/거부 버튼 표시 (기본 값)
            # True: 항상 허용 없이 허용/거부만 제공 → 부작용 있는 작업에 사용
    }
)
def get_recipe(ingredient: str):
    # 입력된 재료와 유사한 레시피를 벡터 유사도 검색으로 조회
    docs = vector_store.similarity_search(ingredient)
    return docs

# 가짜 DB - code: username 매핑
# 실제 서비스라면 DB에서 조회하고 JWT 등으로 토큰 발급
user_token_db = {
    "ABCDEF": "nico"
}

# GET /autorizae - OAuth 로그인 페이지 반환
# ChatGPT가 사용자를 이 페이지로 리다이렉트시킴
# include_in_schema=False → Swagger UI & OpenAPI 스펙에서 숨김 (GPT Action에 노출 불필요)
@app.get(
    "/authorize",
    response_class=HTMLResponse, # 가상의 HTML 페이지를 반환
    include_in_schema=False
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    # client_id: ChatGPT가 보낸 앱 식별자
    # redirect_uri: 로그인 완료 후 돌아갈 ChatGPT 측 주소
    # state: CSRF 방지용 난수값, 그대로 redirect_uri에 붙여서 돌려줘야 함
    print(client_id, redirect_uri, state)
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <!-- 클릭 시 ChatGPT로 code와 state를 담아 리다이렉트 -->
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """

# POST /token - code를 access_token으로 교환
# ChatGPT가 /authorize에서 받은 code를 가지고 이 엔드포인트로 POST 요청을 보냄
# 이후 ChatGPT → 우리 서버로 보내는 모든 요청엔 이 access_token이 Authorization 헤더에 담김
@app.post(
    "/token",
    include_in_schema=False
)
def handle_token(code = Form(...)): 
    print(code)
    return {
        "access_token": user_token_db[code] # code로 사용자 조회 후 토큰 반환
    }