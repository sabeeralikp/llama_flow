from sqlalchemy import Boolean, Column, Integer, String
from database.database import Base


class BaseChatBot(Base):
    __tablename__ = "base_chatbot"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    vector_db = Column(String)
    collection = Column(String)
    embed_model_provider = Column(String)
    embed_model = Column(String)
    llm_provider = Column(String)
    llm = Column(String)
    load_in_4bit = Column(Boolean)
    chunking_strategy = Column(String)
    semantic_splitting_buffer_size = Column(Integer)
    semantic_splitting_breakpoint_percentile_threshold = Column(Integer)
    retriver_top_k = Column(Integer)
