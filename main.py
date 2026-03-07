from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from database import Session, User, Token
import uuid
import hashlib

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
        {"url": "https://that-success-mouth-publishers.trycloudflare.com"}
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
def get_recipe(request: Request, ingredient: str):
    print(request.headers)
    # 입력된 재료와 유사한 레시피를 벡터 유사도 검색으로 조회
    docs = vector_store.similarity_search(ingredient)
    return docs

# GET /autorizae - OAuth 로그인 페이지 반환
# ChatGPT가 사용자를 이 페이지로 redirect
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
        <body>
            <h1>레시피 GPT 로그인</h1>
            <form action="/login?redirect_uri={redirect_uri}&state={state}" method="post">
                <input type="text" name="username" placeholder="ID" /><br/>
                <input type="password" name="password" placeholder="PW" /><br/>
                <button type="submit">로그인</button>
            </form>
        </body>
    </html>
    """

@app.post("/login", response_class=HTMLResponse, include_in_schema=False)
def handle_login(
    redirect_uri: str,
    state: str,
    username: str = Form(...),
    password: str = Form(...)
):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    with Session() as session:
        user = session.query(User).filter(
            User.username == username,
            User.password == hashed_pw
        ).first()

        if not user:
            return "<h1>아이디 또는 비밀번호가 일치하지 않습니다</h1>"
        
        # code 발급 후 tokens 테이블에 저장
        code = str(uuid.uuid4())
        token = Token(code=code, access_token=None, username=username)
        session.add(token)
        session.commit()

    return f"""
    <html>
        <meta http-equiv="refresh" content="0;url={redirect_uri}?code={code}&state={state}" />
    </html>
"""

# POST /token - code를 access_token으로 교환
# ChatGPT가 /authorize에서 받은 code를 가지고 이 엔드포인트로 POST 요청을 보냄
# 이후 ChatGPT → 우리 서버로 보내는 모든 요청엔 이 access_token이 Authorization 헤더에 담김
@app.post("/token", include_in_schema=False)
def handle_token(code = Form(...)): 
    with Session() as session:
        token = session.query(Token).filter(Token.code == code).first()

        if not token:
            raise HTTPException(status_code=400, detail="Invalid code")

        # access_token 발급 후 DB 업데이트
        access_token = str(uuid.uuid4())
        token.access_token = access_token
        session.commit()

        return {"access_token": access_token}

# Todo
# 유저들이 인증하도록 DB 구축 (O)
# 유저들이 각자 마음에 드는 레시피를 따로 표시할 수 있도록
# 좋아하는 레시피 즐겨찾기
# 즐겨찾기 레시피 리스트를 가져올 수 있도록
# 유저의 레시피를 저장할 url 만들어야 함 GPT에도 액션 추가
# 특정 유저의 레시피를 나열해 줄 또 다른 url, endpoint 추가
# 링크 초대된 사람만 볼 수 있도록 배포해서 공유