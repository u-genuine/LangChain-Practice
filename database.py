from sqlalchemy import ForeignKey, create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker

# DB 연결
DATABASE_URL = "mysql+pymysql://root:root@localhost/recipe_gpt"

# DB 연결 정보를 담고 연결을 관리하는 객체
# application.yml의 spring.datasource.url과 같은 역할
engine = create_engine(DATABASE_URL)

# 이 Base를 상속받은 클래스를 DB 테이블로 인식
# Spring Boot의 @Entity 역할
Base = declarative_base()

# DB 작업(조회/저장/삭제)을 수행하는 세션을 만드는 팩토리
# Spring Boot의 EntityManager 역할
# 사용 시: with Session() as session: ...
Session = sessionmaker(bind=engine)

# User 테이블 정의
class User(Base):
    __tablename__ = "users"
    username = Column(String(50), primary_key=True)
    password = Column(String(255))
    
# tokens 테이블 = code, access_token, username 매핑
# 일회용이 아닌거로 일단 구현
class Token(Base):
    __tablename__ = "tokens"
    code = Column(String(255), primary_key=True)
    access_token = Column(String(255), nullable=True) # /token 호출 시 발급
    username = Column(String(50), ForeignKey("users.username")) # 어떤 유저의 토큰인지

# 정의된 테이블이 DB에 없으면 자동으로 생성
# 이미 존재하는 테이블은 건들지 않음
Base.metadata.create_all(engine)