from fastapi import FastAPI
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
def get_quote():
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950
    }