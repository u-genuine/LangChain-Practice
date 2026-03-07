import hashlib
from database import Session, User

with Session() as session:
    user = User(
        username="u_genuine",
        password = hashlib.sha256("1234".encode()).hexdigest(),
    )
    session.add(user)
    session.commit()
    print("유저 추가 완료")