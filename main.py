from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# FastAPI 앱 초기화 & OpenAPI 문서 설정
# /docs 경로에서 Swagger UI로 확인 가능
app = FastAPI(
    title="Nicolacus Maximus Quote Giver",
    description="Get a real quote said by Nicolas Maximus himself.",
    servers=[
        # Custom GPT Action에서 바라볼 서버 주소 (cloudflared 터널)
        {"url": "https://hopefully-democrats-effective-virtually.trycloudflare.com"}
    ]
)

# 응답 스키마 정의 - OpenAPI 문서에 자동 반영되고 Custom GPT가 구조를 인식하는데 사용됨
class Quote(BaseModel):
    quote: str = Field(description="The quote that Nicolacus Maximus said.")
    year: int = Field(description="The year when Nicolacus Maximus said the quote.")


# GET /quote - 명언 1개를 반환하는 엔드포인트
# Custom GPT가 이 엔드포인트를 호출해 명언 가져옴
@app.get(
    "/quote", 
    summary="Returns a random quote by Nicolacus Maximus",
    description="Upon receiving a GET request this endpoint will return a real quote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said", # Quote Object를 만들기 위해 pydantic 사용
    response_model=Quote,
    openapi_extra={
        # False: 한 번 허용/항상 허용/거부 버튼 표시 (기본 값)
        # True: 항상 허용 없이 허용/거부만 제공 → 부작용 있는 작업에 사용
        "x-openai-isConsequential": True 
    }
)
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950
    }

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
# ChatGPT가 /authorizae에서 받은 code를 가지고 이 엔드포인트로 POST 요청을 보냄
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