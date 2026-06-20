"""Auth schemas."""
from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    email: str | None = None
    full_name: str | None = None
    role: str = "officer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str | None
    full_name: str | None
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class LoginRequest(BaseModel):
    username: str
    password: str
