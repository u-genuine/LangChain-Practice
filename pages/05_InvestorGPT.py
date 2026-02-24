import os
from dotenv import load_dotenv
import openai
import streamlit as st
import requests
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema.messages import SystemMessage

load_dotenv(dotenv_path="./env/.env")
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

# [스키마 1] LLM에게 이 도구를 쓸 때 어떤 값을 넣어야 하는지 알려주는 설계도 역할
# 검색어(query) 하나를 받는 스키마
class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

# [도구 1] 주식 심볼 검색
# 스키마 -> 입력 규격 / 툴 -> 실제 실행 로직
# LLM이 description을 읽고 언제 이 도구를 쓸지 스스로 판단함 
class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It makes a query as an argument.
    Example query: Stock Market Symbol for Apple Company
"""
    # 위에서 만든 스키마 연결 -> LLM이 잘못된 입력을 넣으면 자동으로 걸러줌
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


# [스키마 2] 심볼(symbol) 하나만 받는 스키마
class CompanyOverviewArgsSchema(BaseModel):
    symbol : str = Field(description="Symbol of the company. /nExample: AAPL, TSLA")

# [도구 2] 기업 재무 정보 조회
# 도구 1에서 심볼을 얻은 뒤 Alpha Vantage API로 재무 데이터를 가져옴
# 투자 판단에 필요한 지표들을 반환
class CompanyOverviewTool(BaseTool):
    name="CompanyOverview"
    description="""
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    # 위에서 만든 스키마 연결
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


# [도구 3] 손익계산서 조회
# 매출, 영업이익, 순이익 등 기간별 실적 데이터 가져옴
class CompanyIncomeStatmentTool(BaseTool):
    name="CompanyIncomeStatment"
    description="""
    Use this to get the income statement of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]

# [도구 4] 주간 성과 조회
class CompanyStockPerformanceTool(BaseTool):
    name="CompanyStockPerformance"
    description="""
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )

        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]
    
agent = initialize_agent(
    llm=llm,
    verbose=True, # agent가 하는 작업 과정 & 추론을 볼 수 있음

    # OpenAI의 Function Calling 기능을 활용하는 에이전트 유형
    # 별도의 문자열 파싱 없이, 정의된 스키마에 맞춰 여러 input을 구조회된 형태로 받을 수 있음
    agent=AgentType.OPENAI_FUNCTIONS, 

    # 파싱 에러 처리
    # LLM 답변이 지정된 양식을 벗어나서 Output Parser가 읽지 못할 경우, LLM에게 교정을 유도함
    handle_parsing_errors=True, 

    # 도구들을 조립해서 "기업명 -> 심볼 검색 -> 재무 정보 조회"
    # LLM이 알아서 순서를 판단하고 도구를 골라서 씀
    tools = [
        StockMarketSymbolSearchTool(), # 기업명으로 주식 심볼 검색
        CompanyOverviewTool(), # 주식 심볼로 기업 재무 정보 조회
        CompanyIncomeStatmentTool(), # 손익계산서 조회
        CompanyStockPerformanceTool()
    ],
    agent_kwargs={
        "system_message": SystemMessage(content="""
            You are a hedge fund manager.
                                        
            You evaluate a company and provide your opinion and reasons why the stock is buy or not.
                                        
            Consider the performance of a stock, the company overview and the income statement.
                                        
            Be assertive in your judgement and recommend the stock or advise the user against it.
        """)
    }
)

# prompt = "Give me financial information on Apple's stock, considering its financial and income statements and stock performance and help me analyze if it's a potential good investment."




st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈"
)

st.markdown(
    """
    # InvestorGPT

    Welcome to InvestorGPT.

    Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write the name of the company you are interested on.")

if company:
    result = agent.invoke(company)
    st.write(result["output"])
    