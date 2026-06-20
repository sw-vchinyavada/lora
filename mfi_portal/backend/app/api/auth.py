"""Auth API."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import LoginRequest, Token, UserResponse, UserCreate
from app.core.security import verify_password, get_password_hash, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


def _make_token_response(user: User) -> Token:
    token = create_access_token({"sub": user.username, "role": user.role})
    return Token(
        access_token=token,
        user=UserResponse(id=user.id, username=user.username, email=user.email, full_name=user.full_name, role=user.role),
    )


@router.post("/login", response_model=Token)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account inactive")
    user.last_login = datetime.utcnow()
    db.commit()
    return _make_token_response(user)


@router.post("/token", response_model=Token)
def login_form(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 form login (for Swagger / API clients)."""
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account inactive")
    return _make_token_response(user)


@router.post("/register", response_model=UserResponse)
def register(data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    user = User(
        username=data.username,
        email=data.email,
        full_name=data.full_name,
        hashed_password=get_password_hash(data.password),
        role=data.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, username=user.username, email=user.email, full_name=user.full_name, role=user.role)
