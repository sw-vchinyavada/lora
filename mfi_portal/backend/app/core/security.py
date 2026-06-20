"""Authentication and security."""
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel

from .config import ALGORITHM, SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES

def verify_password(plain: str, hashed: str) -> bool:
    if isinstance(plain, str):
        plain = plain.encode("utf-8")
    if isinstance(hashed, str):
        hashed = hashed.encode("utf-8")
    return bcrypt.checkpw(plain, hashed)


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


class TokenData(BaseModel):
    username: str
    role: str = "officer"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "officer")
        if username is None:
            return None
        return TokenData(username=username, role=role)
    except JWTError:
        return None
