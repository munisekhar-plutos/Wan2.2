from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .config import settings
from .models import Base

# Setup standard engine. SQLite will be used by default (e.g. sqlite:///api_jobs.db)
# connect_args={"check_same_thread": False} is required only for SQLite
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all database tables on application startup."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI Dependency to yield database sessions with clean lifecycles."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
